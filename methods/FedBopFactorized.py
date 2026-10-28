import math
import torch
from core.FederatedLearningClass import *
from utils.logger import *


class FedBopFactorized(FederatedLearningClass):
    """
    FedBop with bounded client-factorized server memory.

    Client side is unchanged from FedBop12:
      * ordinary Bop on binary parameters,
      * the same temporal state-majority sparse event,
      * persistent Adam for non-binary trainables,
      * ternary client-momentum consensus only at gamma-decay boundaries,
      * the same minimum-feasible client momentum projection after server update.

    Only the SERVER binary aggregation is changed.

    Let v_{k,i} in {0,1} be the sparse event from client k at coordinate i.
    FedBop12's aligned server recurrence is

        A_i^{r+1} = beta_K A_i^r + tau * c_i^r / b,

    where c_i^r = sum_k v_{k,i}^r,
          beta_K = (1-gamma)^K,
          b = floor(K/2) + 0.5.

    We factor A_i/tau into per-client normalized aligned contributions

        A_i / tau = sum_k h_{k,i},

    and update each client contribution independently:

        h_{k,i}^{r+1}
          = clip(beta_K h_{k,i}^r + v_{k,i}^r / b, -1, +1).

    The global coordinate flips iff

        sum_k h_{k,i}^{r+1} > 1.

    Thus:
      * round 1 is exactly the same zero-history strict-majority boundary as
        FedBop12;
      * repeated evidence from one client increases that client's confidence but
        is bounded at one Bop threshold and can never grow without limit;
      * support from different clients can accumulate asynchronously;
      * v=0 is not a negative vote; it only lets old evidence decay by beta_K.

    On an accepted global bit flip, the aligned factor state changes sign:

        h_{k,i} <- -h_{k,i}

    which is the exact aligned-state consequence of preserving the corresponding
    virtual momentum while W changes sign. No reset, majority gate, signed
    no-event rule, or new training hyperparameter is introduced.

    The per-client server factors are diagnostic/aggregation state only; they are
    not claimed to reconstruct the clients' true private Bop momentum.
    """

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.contributors_percent = (
            float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        )

        # Bop hyperparameters. One gamma is used consistently on client and server,
        # with the same decay schedule as the existing FedBop experiments.
        self.gamma = self.get_arg(float, "gamma", 1e-4)
        self.tau = self.get_arg(float, "tau", 1e-8)
        self.decay_factor = self.get_arg(float, "decay_factor", 1e-1)
        self.decay_rounds = self.get_arg(int, "decay_rounds", 200)

        # Low-bit client-momentum consensus used ONLY at gamma-decay boundaries.
        # Five ternary symbols fit in one uint8 (3**5=243), plus one FP32 scale
        # per binary layer. These are codec parameters, not training policies.
        self.ternary_deadzone_mult = self.get_arg(float, "ternary_deadzone_mult", 0.5)
        self.ternary_min_scale_mult = self.get_arg(float, "ternary_min_scale_mult", 1e-3)

        # Adam for real-valued trainable parameters (BN beta in BinaryNet).
        self.stats_lr = self.get_arg(float, "stats_lr", 1e-2)
        self.stats_optimizer = None
        self.stats_param_ids = None

        # Client-private Bop state.
        self.m_buffers = None
        self.round_global_weights = None
        self.temporal_opposite_counts = {}
        self.temporal_state_steps = {}
        self.round_local_vote_masks = {}
        self.has_local_update = False

        # Server-side bounded client-factorized aligned memory.
        # Layout:
        #   factor_buffers[layer_name][client_id] -> flat CPU float32 h in [-1,1]
        # h is dimensionless: h = a/tau, where a is an aligned server evidence
        # contribution for that client. For K=10 and BinaryNet, float32 storage is
        # about 414 MB on the server.
        self.factor_buffers = {}

        # Prepared on a decay round after receiving all client momentum sidecars,
        # then broadcast together with the decayed gamma. This is CLIENT-EMA
        # consensus only; factor_buffers is deliberately preserved across decay.
        self.pending_shared_m_ternary = {}
        self.pending_sync_round = -1

        # Round/gamma bookkeeping.
        self._scheduled_local_round = 1
        self._diag_client_round = 0
        self._last_decay_round = None

        # Lightweight diagnostics only; they do not affect optimization.
        self.diag = self.get_arg(int, "diag", 1)
        self.diag_layer_every = self.get_arg(int, "diag_layer_every", 20)
        self._diag_steps = 0
        self._diag_raw_flips = {}
        self._diag_grad_abs = {}

        # Extended server-side diagnostics. These states NEVER participate in
        # optimization or model updates; they only compare sparse event/flip sets
        # across clients and consecutive rounds.
        self.diag_extended = self.get_arg(int, "diag_extended", 1)
        self._diag_prev_client_events = {}   # (client_id, layer) -> (round, sorted idx)
        self._diag_prev_server_flips = {}    # layer -> (round, bool mask)

    def get_name(self):
        return "FedBopFactorizedV3"

    def init_method(self, server=None):
        if server is not None:
            self.server = server

    def select_clients_to_train(self, all_clients):
        if self.contributors_percent != 1.0:
            return self.select_random_clients(all_clients, self.contributors_percent)
        return super().select_clients_to_train(all_clients)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _log(self, message):
        if not self.diag:
            return
        text = f"[FedBopFactorizedV3BoundedClientMemory] {message}"
        try:
            logger.log_info(text)
        except Exception:
            try:
                logger.log_debug(text)
            except Exception:
                print(text, flush=True)

    def _layer_log_enabled(self, round_number):
        if not self.diag:
            return False
        if round_number <= 3:
            return True
        return self.diag_layer_every > 0 and round_number % self.diag_layer_every == 0

    @torch.no_grad()
    def _sign_binary(self, tensor):
        return torch.where(tensor >= 0, torch.ones_like(tensor), -torch.ones_like(tensor))

    def _is_binary_param_tensor(self, name, tensor):
        if not torch.is_tensor(tensor):
            return False
        if not torch.is_floating_point(tensor):
            return False
        if tensor.dim() < 2:
            return False

        lname = name.lower()
        if "weight" not in lname:
            return False

        skip_tokens = (
            "bn",
            "batchnorm",
            "norm",
            "gn",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        )
        return not any(token in lname for token in skip_tokens)

    def _prepare_stats_optimizer(self, model):
        params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and not self._is_binary_param_tensor(name, p.data)
        ]
        ids = tuple(id(p) for p in params)

        if self.stats_optimizer is None or self.stats_param_ids != ids:
            self.stats_optimizer = torch.optim.Adam(
                params,
                lr=self.stats_lr,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0.0,
                amsgrad=False,
            )
            self.stats_param_ids = ids
            self._log(
                f"STATS_OPT Adam params={len(params)} lr={self.stats_lr:.3e} "
                f"persistent_state=1 outer_SGD=disabled"
            )

    @torch.no_grad()
    def _prepare_m_buffer(self, key, reference):
        if self.m_buffers is None:
            self.m_buffers = {}

        m = self.m_buffers.get(key)
        if m is None or m.shape != reference.shape:
            m = torch.zeros_like(reference)
        elif m.device != reference.device or m.dtype != reference.dtype:
            m = m.to(device=reference.device, dtype=reference.dtype)

        self.m_buffers[key] = m
        return m

    @torch.no_grad()
    def _create_client_state(self, model):
        self.m_buffers = {}
        self.round_global_weights = {}
        self.temporal_opposite_counts = {}
        self.temporal_state_steps = {}
        self.round_local_vote_masks = {}

        for name, param in model.named_parameters():
            if self._is_binary_param_tensor(name, param.data):
                self.m_buffers[name] = torch.zeros_like(param.data)

    def _is_decay_round(self, round_number):
        try:
            r = int(round_number)
        except Exception:
            return False
        return (
            self.decay_rounds > 0
            and r > 0
            and r < int(self.num_of_rounds)
            and (r % int(self.decay_rounds) == 0)
        )

    @torch.no_grad()
    def _pack_ternary_base3(self, q):
        """Pack q in {-1,0,+1}: five trits per uint8 (3**5 = 243)."""
        q = q.detach().to(device="cpu", dtype=torch.int16).view(-1)
        trits = q + 1  # -1->0, 0->1, +1->2
        n = int(trits.numel())
        pad = (-n) % 5
        if pad:
            # q=0 corresponds to trit 1, so padded coordinates decode to zero.
            trits = torch.cat([trits, torch.ones(pad, dtype=torch.int16)])
        trits = trits.view(-1, 5).to(torch.int32)
        powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32).view(1, 5)
        packed = (trits * powers).sum(dim=1).to(torch.uint8)
        return packed, n

    @torch.no_grad()
    def _unpack_ternary_base3(self, packed, numel):
        packed = torch.as_tensor(packed, dtype=torch.uint8, device="cpu").view(-1)
        values = packed.to(torch.int32).view(-1, 1)
        powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32).view(1, 5)
        trits = torch.remainder(torch.div(values, powers, rounding_mode="floor"), 3)
        return (trits.to(torch.int8) - 1).view(-1)[: int(numel)]

    @torch.no_grad()
    def _quantize_momentum_ternary(self, m):
        """Bop-aware ternary momentum codec with one FP32 scale per layer."""
        x = m.detach().to(device="cpu", dtype=torch.float32).view(-1)
        deadzone = max(0.0, float(self.ternary_deadzone_mult)) * float(self.tau)
        active = x.abs() > deadzone
        q = torch.zeros(x.numel(), dtype=torch.int8)
        active_n = int(active.sum().item())

        if active_n > 0:
            q[active] = torch.sign(x[active]).to(torch.int8)
            scale = float(x[active].abs().mean().item())
        else:
            scale = float(self.tau)

        min_scale = max(1e-30, float(self.ternary_min_scale_mult) * float(self.tau))
        scale = max(scale, min_scale)
        packed, numel = self._pack_ternary_base3(q)
        recon = q.to(torch.float32) * float(scale)

        mse = float(((recon - x) ** 2).mean().item()) if x.numel() else 0.0
        mean_abs = float(x.abs().mean().item()) if x.numel() else 0.0
        rel_rmse = math.sqrt(mse) / max(1e-30, mean_abs)

        entry = {
            "packed": packed,
            "numel": int(numel),
            "scale": torch.tensor(float(scale), dtype=torch.float32),
        }
        diag = {
            "nonzero": active_n,
            "packed_bytes": int(packed.numel()),
            "scale": float(scale),
            "deadzone": float(deadzone),
            "rel_rmse": float(rel_rmse),
        }
        return entry, recon, diag

    @torch.no_grad()
    def _decode_ternary_entry(self, entry, device="cpu", dtype=torch.float32):
        q = self._unpack_ternary_base3(entry["packed"], entry["numel"])
        x = q.to(torch.float32) * float(entry["scale"])
        return x.to(device=device, dtype=dtype)

    @torch.no_grad()
    def _prepare_decay_consensus(self, clients_models, global_model, round_number):
        """Average client Bop EMA states on every gamma-decay boundary.

        The ordinary sparse event aggregation has already selected the global
        binary model for this round. Here we ONLY build a shared CLIENT momentum
        state for the next gamma regime. The server event momentum remains intact.
        """
        self.pending_shared_m_ternary = {}
        self.pending_sync_round = int(round_number)

        k_clients = len(clients_models)
        total_numel = 0
        total_uplink_bytes = 0
        total_downlink_bytes = 0
        total_mean_abs = 0.0
        total_sync_abs = 0.0
        total_sync_nonzero = 0
        total_rel_rmse = 0.0
        total_client_mad = 0.0
        sync_layers = 0

        for key, global_tensor in global_model.items():
            if not torch.is_tensor(global_tensor):
                continue
            if not self._is_binary_param_tensor(key, global_tensor):
                continue

            n = int(global_tensor.numel())
            m_sum = torch.zeros(n, dtype=torch.float32, device="cpu")
            received = 0

            for _, packet in clients_models:
                entry = packet.get("momentum_ternary_model", {}).get(key)
                if entry is None:
                    continue
                decoded = self._decode_ternary_entry(entry).view(-1)
                if decoded.numel() != n:
                    raise ValueError(
                        f"Decay-consensus momentum size mismatch for {key}: "
                        f"{decoded.numel()} != {n}"
                    )
                m_sum.add_(decoded)
                received += 1
                total_uplink_bytes += int(torch.as_tensor(entry["packed"]).numel()) + 4

            if received != k_clients:
                raise RuntimeError(
                    f"Decay round {round_number}: expected momentum sidecar for "
                    f"layer {key} from all {k_clients} participating clients; got {received}."
                )

            mean_m = m_sum / float(received)

            # Direct measure of client optimizer-state desynchronization before
            # consensus: mean absolute deviation from the client mean. This is a
            # diagnostic only and is especially useful around gamma transitions.
            layer_abs_dev = 0.0
            for _, packet in clients_models:
                entry = packet["momentum_ternary_model"][key]
                decoded = self._decode_ternary_entry(entry).view(-1)
                layer_abs_dev += float((decoded - mean_m).abs().sum().item())
            total_client_mad += layer_abs_dev / float(received)

            shared_entry, shared_m, qdiag = self._quantize_momentum_ternary(mean_m)
            self.pending_shared_m_ternary[key] = shared_entry

            total_numel += n
            total_downlink_bytes += int(qdiag["packed_bytes"]) + 4
            total_mean_abs += float(mean_m.abs().sum().item())
            total_sync_abs += float(shared_m.abs().sum().item())
            total_sync_nonzero += int(qdiag["nonzero"])
            total_rel_rmse += float(qdiag["rel_rmse"])
            sync_layers += 1

        if self.diag:
            self._log(
                f"DECAY_M_CONSENSUS round={round_number} K={k_clients} "
                f"gamma_before={self.gamma:.3e} layers={sync_layers} "
                f"mean|m_mean|={total_mean_abs/max(1,total_numel):.3e} "
                f"client_m_MAD={total_client_mad/max(1,total_numel):.3e} "
                f"mean|m_shared|={total_sync_abs/max(1,total_numel):.3e} "
                f"shared_nonzero_rate={total_sync_nonzero/max(1,total_numel):.6e} "
                f"mean_rel_rmse={total_rel_rmse/max(1,sync_layers):.3e} "
                f"ternary_uplink_total_MiB={total_uplink_bytes/(1024.0**2):.3f} "
                f"ternary_uplink_per_client_MiB={(total_uplink_bytes/max(1,k_clients))/(1024.0**2):.3f} "
                f"ternary_downlink_MiB={total_downlink_bytes/(1024.0**2):.3f} "
                f"server_event_m_preserved=1"
            )

    def _infer_round_number(self, clients_models=None):
        if clients_models:
            for _, packet in clients_models:
                diag = packet.get("diag", {})
                if "round" in diag:
                    try:
                        return int(diag["round"])
                    except Exception:
                        pass
        try:
            return int(self.server.round_number)
        except Exception:
            return -1

    # -------------------------------------------------------------------------
    # Client local optimization
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def after_backward(self, client_train_dict):
        model = client_train_dict["client_model"]

        if self.m_buffers is None:
            self._create_client_state(model)

        self._prepare_stats_optimizer(model)
        self._diag_steps += 1

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not self._is_binary_param_tensor(name, param.data):
                continue

            # Stored kernel itself is binary; Bop owns this parameter.
            param.data.copy_(self._sign_binary(param.data))
            m = self._prepare_m_buffer(name, param.data)
            grad = param.grad.detach()

            # Original Bop EMA.
            m.mul_(1.0 - self.gamma).add_(grad, alpha=self.gamma)

            # For w in {-1,+1}, the ordinary Bop flip condition is equivalent to
            # w*m > tau.
            flip_mask = (m.abs() > self.tau) & (
                torch.sign(m) == self._sign_binary(param.data)
            )

            if self.diag:
                if name not in self._diag_raw_flips:
                    self._diag_raw_flips[name] = 0
                    self._diag_grad_abs[name] = 0.0
                self._diag_raw_flips[name] += int(flip_mask.sum().item())
                self._diag_grad_abs[name] += float(grad.abs().mean().item())

            # Apply actual local Bop transition.
            param.data[flip_mask] = -param.data[flip_mask]
            param.data.copy_(self._sign_binary(param.data))

            # Fixed one-bit semantic for the whole run:
            # Did the realized binary state spend a strict majority of local
            # minibatches opposite to the round-start global state w0?
            w0_cpu = None if self.round_global_weights is None else self.round_global_weights.get(name)
            if w0_cpu is not None:
                w0 = self._sign_binary(
                    w0_cpu.to(device=param.device, dtype=param.dtype)
                )
                count = self.temporal_opposite_counts.get(name)
                if (
                    count is None
                    or count.shape != param.shape
                    or count.device != param.device
                ):
                    count = torch.zeros_like(param.data, dtype=torch.int16)

                count.add_((self._sign_binary(param.data) != w0).to(torch.int16))
                self.temporal_opposite_counts[name] = count
                self.temporal_state_steps[name] = self.temporal_state_steps.get(name, 0) + 1

            # Prevent the framework optimizer from touching binary parameters.
            param.grad = None

        # Real-valued trainables are optimized ONCE by Adam.
        if self.stats_optimizer is not None:
            self.stats_optimizer.step()
            self.stats_optimizer.zero_grad(set_to_none=True)

        # Important: ClientExp interprets False as "do not call the outer optimizer".
        # This prevents the non-binary parameters from receiving an additional SGD
        # update after Adam and matches the intended Bop optimization split.
        return False

    @torch.no_grad()
    def pack_client_model(self, raw_model, global_model, id):
        tensors = {}
        stats_model = {}

        round_number = int(self._scheduled_local_round or (self._diag_client_round + 1))
        self._diag_client_round = round_number
        sync_round = self._is_decay_round(round_number)
        momentum_ternary_model = {}
        sync_payload_bytes = 0
        sync_nonzero = 0
        sync_rel_rmse_sum = 0.0
        sync_layers = 0

        total_binary = 0
        total_endpoint = 0
        total_votes = 0
        total_intersection = 0
        total_union = 0
        layer_diag = {}

        for key, tensor in raw_model.items():
            if not torch.is_tensor(tensor):
                continue

            if not self._is_binary_param_tensor(key, tensor):
                stats_model[key] = tensor.detach().cpu()
                continue

            local_weight = self._sign_binary(tensor.detach().cpu()).view(-1)
            global_weight = self._sign_binary(global_model[key].detach().cpu()).view(-1)
            endpoint_flip = local_weight != global_weight

            steps = int(self.temporal_state_steps.get(key, 0))
            counts = self.temporal_opposite_counts.get(key)

            if counts is None or counts.numel() != tensor.numel() or steps <= 0:
                # Framework-safety fallback only; normal FedWork path always has history.
                vote_mask = endpoint_flip.clone()
                fallback = 1
            else:
                c = counts.detach().cpu().view(-1).to(torch.int32)
                # Strict majority: c > steps-c. A tie produces no event.
                vote_mask = c > (int(steps) - c)
                fallback = 0

            self.round_local_vote_masks[key] = vote_mask.clone()
            flip_indices = torch.nonzero(vote_mask, as_tuple=False).view(-1)
            tensors[key] = {"idx": flip_indices.tolist()}

            # On a gamma-decay boundary only, add a low-bit snapshot of the
            # client-private Bop EMA. This does not change the ordinary event path.
            if sync_round:
                m = self._prepare_m_buffer(key, tensor)
                ternary_entry, _, ternary_diag = self._quantize_momentum_ternary(m)
                momentum_ternary_model[key] = ternary_entry
                sync_payload_bytes += int(ternary_diag["packed_bytes"]) + 4
                sync_nonzero += int(ternary_diag["nonzero"])
                sync_rel_rmse_sum += float(ternary_diag["rel_rmse"])
                sync_layers += 1

            # Preserve the round-start reference for post-server projection.
            self.round_global_weights[key] = global_weight.view_as(tensor).clone()

            if self.diag:
                endpoint_n = int(endpoint_flip.sum().item())
                vote_n = int(vote_mask.sum().item())
                inter_n = int((vote_mask & endpoint_flip).sum().item())
                union_n = int((vote_mask | endpoint_flip).sum().item())
                total_binary += tensor.numel()
                total_endpoint += endpoint_n
                total_votes += vote_n
                total_intersection += inter_n
                total_union += union_n
                layer_diag[key] = {
                    "n": tensor.numel(),
                    "steps": steps,
                    "endpoint": endpoint_n,
                    "votes": vote_n,
                    "jaccard": inter_n / max(1, union_n),
                    "fallback": fallback,
                }

        if self.diag:
            self._log(
                f"CLIENT_STATE_MAJ round={round_number} id={id} gamma={self.gamma:.3e} "
                f"binary={total_binary} endpoint={total_endpoint} "
                f"endpoint_rate={total_endpoint/max(1,total_binary):.6e} "
                f"events={total_votes} event_rate={total_votes/max(1,total_binary):.6e} "
                f"event_endpoint_jaccard={total_intersection/max(1,total_union):.6f}"
            )
            if self._layer_log_enabled(round_number):
                for key, d in layer_diag.items():
                    self._log(
                        f"CLIENT_LAYER round={round_number} id={id} layer={key} "
                        f"n={d['n']} steps={d['steps']} endpoint={d['endpoint']} "
                        f"events={d['votes']} jaccard={d['jaccard']:.4f} "
                        f"fallback={d['fallback']}"
                    )

        packet = {
            "client_id": id,
            "tensors": tensors,
            "stats_model": stats_model,
        }
        if sync_round:
            packet["momentum_ternary_model"] = momentum_ternary_model
            if self.diag:
                self._log(
                    f"DECAY_M_UPLINK round={round_number} id={id} "
                    f"layers={sync_layers} payload_MiB={sync_payload_bytes/(1024.0**2):.3f} "
                    f"nonzero_rate={sync_nonzero/max(1,total_binary):.6e} "
                    f"mean_rel_rmse={sync_rel_rmse_sum/max(1,sync_layers):.3e}"
                )
        if self.diag:
            packet["diag"] = {
                "round": round_number,
                "steps": self._diag_steps,
                "endpoint_flips": total_endpoint,
                "transmitted_votes": total_votes,
                "total_binary": total_binary,
            }

        self._diag_steps = 0
        self._diag_raw_flips = {}
        self._diag_grad_abs = {}
        self.has_local_update = True
        return packet

    def unpack_client_model(self, packed_model):
        return packed_model

    # -------------------------------------------------------------------------
    # Diagnostic helpers (server-only; no effect on optimization)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _diag_sparse_intersection(self, a, b):
        """Exact intersection size for two sorted unique int64 CPU index tensors."""
        if a is None or b is None or a.numel() == 0 or b.numel() == 0:
            return 0
        a = a.to(device="cpu", dtype=torch.long).view(-1)
        b = b.to(device="cpu", dtype=torch.long).view(-1)
        # torch.nonzero-generated event indices are sorted; searchsorted avoids
        # allocating dense N-bit masks for every client.
        pos = torch.searchsorted(b, a)
        valid = pos < b.numel()
        if not bool(valid.any()):
            return 0
        av = a[valid]
        pv = pos[valid]
        return int((b[pv] == av).sum().item())

    # -------------------------------------------------------------------------
    # Server aggregation: bounded client-factorized memory
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def aggregate(self, clients_models, global_model):
        if len(clients_models) == 0:
            return global_model

        round_number = self._infer_round_number(clients_models)
        k_clients = len(clients_models)
        majority = (k_clients // 2) + 1

        gamma = float(self.gamma)
        if not (0.0 < gamma < 1.0):
            raise ValueError("FedBopFactorizedV3 requires 0 < gamma < 1")

        beta = 1.0 - gamma
        beta_k = beta ** k_clients

        # Same strict-majority calibration used by FedBop12.
        # K=10 -> majority=6, b=5.5.
        decision_boundary = float(majority) - 0.5
        event_increment = 1.0 / decision_boundary  # normalized by tau

        # Stable identity list for the participating clients.
        client_ids = [
            str(packet.get("client_id", client_label))
            for client_label, packet in clients_models
        ]
        if len(set(client_ids)) != len(client_ids):
            raise RuntimeError("FedBopFactorizedV3 received duplicate client ids")

        total_binary = 0
        total_union = 0
        total_votes = 0
        total_flips = 0
        total_instant = 0

        # Factor-state diagnostics.
        total_factor_entries = 0
        total_factor_abs = 0.0
        total_factor_sum = 0.0
        total_factor_positive = 0
        total_factor_negative = 0
        total_factor_saturated_pos = 0
        total_factor_saturated_neg = 0
        total_flip_score = 0.0

        # Extended diagnostic accumulators retained from FedBop12Diag.
        support_hist_all = torch.zeros(k_clients + 1, dtype=torch.int64)
        support_hist_flips = torch.zeros(k_clients + 1, dtype=torch.int64)
        pair_intersections = 0
        pair_unions = 0
        same_client_repeat_intersection = 0
        same_client_repeat_union = 0
        same_client_current_events = 0
        server_flip_repeat_intersection = 0
        server_flip_repeat_union = 0
        server_current_flips = 0

        for key, global_tensor in global_model.items():
            if not torch.is_tensor(global_tensor):
                continue
            if not self._is_binary_param_tensor(key, global_tensor):
                continue

            # Current global sign is only needed to apply the accepted transition;
            # h itself is already aligned to the current global sign.
            w = self._sign_binary(global_tensor.detach().cpu()).view(-1)
            n = int(w.numel())

            vote_count = torch.zeros(n, dtype=torch.int16, device="cpu")
            event_idx_by_client = {}

            # Decode sparse events and retain identity.
            for client_label, packet in clients_models:
                cid = str(packet.get("client_id", client_label))
                entry = packet.get("tensors", {}).get(key)
                if entry is None:
                    idx = torch.empty(0, dtype=torch.long, device="cpu")
                else:
                    idx = torch.as_tensor(
                        entry.get("idx", []), dtype=torch.long, device="cpu"
                    ).view(-1)

                event_idx_by_client[cid] = idx

                if self.diag and self.diag_extended:
                    state_key = (cid, key)
                    prev = self._diag_prev_client_events.get(state_key)
                    if prev is not None and int(prev[0]) == int(round_number) - 1:
                        prev_idx = prev[1]
                        inter = self._diag_sparse_intersection(idx, prev_idx)
                        same_client_repeat_intersection += inter
                        same_client_repeat_union += (
                            int(idx.numel()) + int(prev_idx.numel()) - inter
                        )
                        same_client_current_events += int(idx.numel())
                    self._diag_prev_client_events[state_key] = (
                        int(round_number), idx.clone()
                    )

                if idx.numel() > 0:
                    vote_count[idx] += 1

            layer_factors = self.factor_buffers.setdefault(key, {})

            # Predict/decay every participating client's factor, then add this
            # round's sparse event contribution and clip to the Bop-derived bound.
            for cid in client_ids:
                h = layer_factors.get(cid)
                if h is None or h.numel() != n:
                    h = torch.zeros(n, dtype=torch.float32, device="cpu")
                else:
                    h = h.to(device="cpu", dtype=torch.float32)

                h.mul_(beta_k)

                idx = event_idx_by_client[cid]
                if idx.numel() > 0:
                    h[idx] = h[idx] + float(event_increment)

                h.clamp_(min=-1.0, max=1.0)
                layer_factors[cid] = h

            # Aggregate identity-preserving client contributions.
            score = torch.zeros(n, dtype=torch.float32, device="cpu")
            for cid in client_ids:
                score.add_(layer_factors[cid])

            # Normalized Bop condition:
            #   A = tau * score, so A > tau <=> score > 1.
            flip_mask_cpu = score > 1.0
            instant_mask = vote_count >= majority

            if self.diag and self.diag_extended:
                support_hist_all += torch.bincount(
                    vote_count.to(torch.long), minlength=k_clients + 1
                )[: k_clients + 1]

                if bool(flip_mask_cpu.any()):
                    support_hist_flips += torch.bincount(
                        vote_count[flip_mask_cpu].to(torch.long),
                        minlength=k_clients + 1,
                    )[: k_clients + 1]

                c64 = vote_count.to(torch.int64)
                pair_intersections += int(((c64 * (c64 - 1)) // 2).sum().item())
                kmc = int(k_clients) - c64
                total_pairs = int(k_clients * (k_clients - 1) // 2)
                pair_unions += int(
                    (total_pairs - ((kmc * (kmc - 1)) // 2)).sum().item()
                )

                prev_flip = self._diag_prev_server_flips.get(key)
                if prev_flip is not None and int(prev_flip[0]) == int(round_number) - 1:
                    prev_mask = prev_flip[1]
                    inter = int((flip_mask_cpu & prev_mask).sum().item())
                    union = int((flip_mask_cpu | prev_mask).sum().item())
                    server_flip_repeat_intersection += inter
                    server_flip_repeat_union += union
                    server_current_flips += int(flip_mask_cpu.sum().item())
                self._diag_prev_server_flips[key] = (
                    int(round_number), flip_mask_cpu.clone()
                )

            # Apply accepted global binary transition.
            if bool(flip_mask_cpu.any()):
                mask_dev = flip_mask_cpu.to(
                    device=global_tensor.device, dtype=torch.bool
                )
                global_tensor.view(-1)[mask_dev] *= -1
            global_tensor.copy_(self._sign_binary(global_tensor))

            # Preserve the corresponding virtual momentum across W -> -W:
            # aligned contribution h therefore changes sign exactly.
            if bool(flip_mask_cpu.any()):
                for cid in client_ids:
                    h = layer_factors[cid]
                    h[flip_mask_cpu] = -h[flip_mask_cpu]
                    layer_factors[cid] = h

            if self.diag:
                total_binary += n
                total_union += int((vote_count > 0).sum().item())
                total_votes += int(vote_count.sum().item())
                total_flips += int(flip_mask_cpu.sum().item())
                total_instant += int(instant_mask.sum().item())

                if bool(flip_mask_cpu.any()):
                    total_flip_score += float(score[flip_mask_cpu].sum().item())

                for cid in client_ids:
                    h = layer_factors[cid]
                    total_factor_entries += int(h.numel())
                    total_factor_abs += float(h.abs().sum().item())
                    total_factor_sum += float(h.sum().item())
                    total_factor_positive += int((h > 0).sum().item())
                    total_factor_negative += int((h < 0).sum().item())
                    total_factor_saturated_pos += int((h >= 1.0).sum().item())
                    total_factor_saturated_neg += int((h <= -1.0).sum().item())

        # Non-binary aggregation is unchanged from FedBop12.
        for key, global_tensor in global_model.items():
            if not torch.is_tensor(global_tensor):
                continue
            if self._is_binary_param_tensor(key, global_tensor):
                continue

            received = []
            for _, packet in clients_models:
                stats = packet.get("stats_model", {})
                if key in stats:
                    received.append(stats[key])
            if not received:
                continue

            if torch.is_floating_point(global_tensor):
                acc = torch.zeros_like(
                    global_tensor, dtype=torch.float32, device=global_tensor.device
                )
                for x in received:
                    acc.add_(x.to(device=global_tensor.device, dtype=torch.float32))
                global_tensor.copy_((acc / len(received)).to(dtype=global_tensor.dtype))
            else:
                value = received[0].to(
                    device=global_tensor.device, dtype=global_tensor.dtype
                )
                for x in received[1:]:
                    value = torch.maximum(
                        value,
                        x.to(device=global_tensor.device, dtype=global_tensor.dtype),
                    )
                global_tensor.copy_(value)

        # Client private-momentum ternary consensus remains exactly as in FedBop12.
        if self._is_decay_round(round_number):
            self._prepare_decay_consensus(clients_models, global_model, round_number)

        if self.diag:
            factor_den = max(1, total_factor_entries)
            self._log(
                f"SERVER_FACTORIZED round={round_number} K={k_clients} "
                f"gamma={self.gamma:.3e} beta_k={beta_k:.9f} "
                f"tau={self.tau:.3e} b={decision_boundary:.3f} "
                f"event_increment={event_increment:.6e} "
                f"union_rate={total_union/max(1,total_binary):.6e} "
                f"mean_support={total_votes/max(1,total_union):.3f} "
                f"instant_majority_rate={total_instant/max(1,total_binary):.6e} "
                f"global_flip_rate={total_flips/max(1,total_binary):.6e} "
                f"mean_abs_h={total_factor_abs/factor_den:.6e} "
                f"mean_h={total_factor_sum/factor_den:.6e} "
                f"positive_h_rate={total_factor_positive/factor_den:.6e} "
                f"negative_h_rate={total_factor_negative/factor_den:.6e} "
                f"sat_pos_rate={total_factor_saturated_pos/factor_den:.6e} "
                f"sat_neg_rate={total_factor_saturated_neg/factor_den:.6e} "
                f"mean_score_on_flips={total_flip_score/max(1,total_flips):.6f}"
            )

            if self.diag_extended:
                flip_hist_total = int(support_hist_flips.sum().item())
                weighted_flip_support = int(
                    (
                        support_hist_flips
                        * torch.arange(k_clients + 1, dtype=torch.int64)
                    ).sum().item()
                )
                low_support_flips = int(support_hist_flips[:majority].sum().item())
                zero_support_flips = int(support_hist_flips[0].item())

                micro_pair_jaccard = pair_intersections / max(1, pair_unions)
                same_client_repeat_jaccard = (
                    same_client_repeat_intersection / max(1, same_client_repeat_union)
                )
                same_client_repeat_rate = (
                    same_client_repeat_intersection / max(1, same_client_current_events)
                )
                server_flip_repeat_jaccard = (
                    server_flip_repeat_intersection / max(1, server_flip_repeat_union)
                )
                server_flip_repeat_rate = (
                    server_flip_repeat_intersection / max(1, server_current_flips)
                )

                hist_all_text = ",".join(
                    f"c{i}={int(support_hist_all[i].item())}"
                    for i in range(k_clients + 1)
                )
                hist_flip_text = ",".join(
                    f"c{i}={int(support_hist_flips[i].item())}"
                    for i in range(k_clients + 1)
                )

                self._log(
                    f"EVENT_ID_DIAG round={round_number} "
                    f"cross_client_micro_jaccard={micro_pair_jaccard:.6e} "
                    f"same_client_repeat_jaccard={same_client_repeat_jaccard:.6e} "
                    f"same_client_repeat_rate={same_client_repeat_rate:.6e}"
                )
                self._log(
                    f"FACTORIZED_FLIP_DIAG round={round_number} "
                    f"mean_current_support_on_flips="
                    f"{weighted_flip_support/max(1,flip_hist_total):.6f} "
                    f"low_support_flip_rate="
                    f"{low_support_flips/max(1,flip_hist_total):.6e} "
                    f"zero_support_flip_rate="
                    f"{zero_support_flips/max(1,flip_hist_total):.6e} "
                    f"consecutive_reflip_rate={server_flip_repeat_rate:.6e} "
                    f"consecutive_reflip_jaccard={server_flip_repeat_jaccard:.6e}"
                )
                self._log(
                    f"EVENT_SUPPORT_HIST round={round_number} {hist_all_text}"
                )
                self._log(
                    f"FACTORIZED_FLIP_SUPPORT_HIST round={round_number} {hist_flip_text}"
                )

        return global_model

    # -------------------------------------------------------------------------
    # Server -> client model/state update
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def pack_server_model(self, raw_model):
        bin_model = {}
        stats_model = {}

        for key, tensor in raw_model.items():
            if not torch.is_tensor(tensor):
                continue
            if self._is_binary_param_tensor(key, tensor):
                bin_model[key] = self._sign_binary(tensor.detach().cpu())
            else:
                stats_model[key] = tensor.detach().cpu()

        try:
            round_number = int(self.server.round_number)
        except Exception:
            round_number = -1

        old_gamma = self.gamma
        did_decay = False
        if (
            round_number > 0
            and round_number < int(self.num_of_rounds)
            and self.decay_rounds > 0
            and round_number % self.decay_rounds == 0
            and self._last_decay_round != round_number
        ):
            self.gamma *= self.decay_factor
            self._last_decay_round = round_number
            did_decay = True

        if self.diag and did_decay:
            self._log(
                f"GAMMA_DECAY round={round_number} "
                f"gamma={old_gamma:.3e}->{self.gamma:.3e}"
            )

        momentum_sync = (
            int(round_number) == int(self.pending_sync_round)
            and len(self.pending_shared_m_ternary) > 0
        )

        next_round = int(round_number) + 1 if round_number >= 0 else 1
        payload = {
            "bin_model": bin_model,
            "stats_model": stats_model,
            "gamma": self.gamma,
            "next_round": next_round,
            "momentum_sync": bool(momentum_sync),
            "momentum_sync_round": int(round_number) if momentum_sync else -1,
        }
        if momentum_sync:
            payload["shared_m_ternary"] = self.pending_shared_m_ternary
        return payload

    @torch.no_grad()
    def unpack_server_model(self, packed_model, current_model):
        bin_model = packed_model["bin_model"]
        stats_model = packed_model["stats_model"]
        momentum_sync = bool(packed_model.get("momentum_sync", False))
        momentum_sync_round = int(packed_model.get("momentum_sync_round", -1))
        shared_m_ternary = packed_model.get("shared_m_ternary", {})
        self.gamma = float(packed_model["gamma"])
        self._scheduled_local_round = int(
            packed_model.get("next_round", self._diag_client_round + 1)
        )

        if self.m_buffers is None:
            self.m_buffers = {}
        if self.round_global_weights is None:
            self.round_global_weights = {}

        total_binary = 0
        total_projected = 0
        total_delta = 0.0
        max_post_violation = 0.0
        sync_numel = 0
        sync_abs_private_to_shared = 0.0
        sync_abs_shared = 0.0
        sync_missing_layers = 0

        for key, server_tensor in bin_model.items():
            if key not in current_model:
                continue

            client_tensor = current_model[key]
            w_new = self._sign_binary(
                server_tensor.to(device=client_tensor.device, dtype=client_tensor.dtype)
            )
            m = self._prepare_m_buffer(key, client_tensor)

            do_projection = False

            if momentum_sync:
                # All clients receive exactly the same reconstructed low-bit EMA.
                # We intentionally do NOT mix this client-gradient EMA into the
                # server's event EMA; only client private states are synchronized.
                entry = shared_m_ternary.get(key)
                if entry is None:
                    sync_missing_layers += 1
                    raise RuntimeError(
                        f"Missing shared momentum for layer {key} on decay sync round "
                        f"{momentum_sync_round}."
                    )
                shared_m = self._decode_ternary_entry(
                    entry, device=m.device, dtype=m.dtype
                ).view_as(m)
                sync_numel += int(m.numel())
                sync_abs_private_to_shared += float((m - shared_m).abs().sum().item())
                sync_abs_shared += float(shared_m.abs().sum().item())
                m.copy_(shared_m)
                do_projection = True

            elif self.has_local_update and key in self.round_global_weights:
                do_projection = True

            if do_projection:
                # -----------------------------------------------------------------
                # MINIMUM FEASIBLE PROJECTION
                # -----------------------------------------------------------------
                # Bop flips a binary weight w when w*m > tau. After the server has
                # accepted w_new, an optimizer state is immediately compatible with
                # that decision iff
                #
                #                   w_new * x <= tau.
                #
                # We do NOT invent a reflected state and do NOT reset x to +/-tau.
                # We solve the parameter-free convex problem
                #
                #       min_x  1/2 ||x - m||_2^2
                #       s.t.   w_new * x <= tau.
                #
                # Since w_new is +/-1, the feasible set is a 1-D half-space. Its
                # exact orthogonal projection is
                #
                #       m* = m - [w_new*m - tau]_+ w_new.
                #
                # Hence every already-feasible coordinate is preserved EXACTLY and
                # only an actual Bop violation is moved, by the minimum possible
                # Euclidean distance, to the non-triggering boundary w_new*m*=tau.
                aligned = w_new * m
                excess = torch.clamp(aligned - self.tau, min=0.0)
                violation = excess > 0
                projected = m - excess * w_new

                if self.diag and violation.any():
                    total_projected += int(violation.sum().item())
                    total_delta += float(
                        (projected[violation] - m[violation]).abs().sum().item()
                    )

                m.copy_(projected)

                if self.diag:
                    total_binary += m.numel()
                    post_excess = torch.clamp(w_new * m - self.tau, min=0.0)
                    if post_excess.numel() > 0:
                        max_post_violation = max(
                            max_post_violation, float(post_excess.max().item())
                        )

            # Server binary state is authoritative for the next local round.
            client_tensor.copy_(w_new)
            self.round_global_weights[key] = w_new.detach().cpu().clone()
            self.temporal_opposite_counts[key] = torch.zeros_like(
                client_tensor, dtype=torch.int16
            )
            self.temporal_state_steps[key] = 0

        # Synchronize the small non-binary state, but KEEP Adam moments persistent.
        for key, server_tensor in stats_model.items():
            if key not in current_model:
                continue
            target = current_model[key]
            target.copy_(
                server_tensor.to(device=target.device, dtype=target.dtype)
            )

        if self.diag and momentum_sync:
            self._log(
                f"DECAY_M_DOWNLINK round={momentum_sync_round} gamma_after={self.gamma:.3e} "
                f"numel={sync_numel} missing_layers={sync_missing_layers} "
                f"mean|m_private-m_shared|={sync_abs_private_to_shared/max(1,sync_numel):.3e} "
                f"mean|m_shared|={sync_abs_shared/max(1,sync_numel):.3e} "
                f"post_sync_projected={total_projected} "
                f"post_sync_projection_rate={total_projected/max(1,total_binary):.6e} "
                f"max_post_violation={max_post_violation:.3e} "
                f"adam_state_preserved=1"
            )
        elif self.diag and self.has_local_update:
            self._log(
                f"MIN_PROJ round={self._diag_client_round} gamma={self.gamma:.3e} "
                f"projected={total_projected} "
                f"projection_rate={total_projected/max(1,total_binary):.6e} "
                f"mean_abs_delta={total_delta/max(1,total_projected):.3e} "
                f"max_post_violation={max_post_violation:.3e} "
                f"adam_state_preserved=1"
            )

        self.has_local_update = False
        return current_model
