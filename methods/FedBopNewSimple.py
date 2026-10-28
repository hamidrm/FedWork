import math
import torch

from core.FederatedLearningClass import *
from utils.logger import *


class FedBopNewSimple(FederatedLearningClass):
    """
    Standalone final FedBop variant.

    This class inherits ONLY from FederatedLearningClass.  It does not import or
    inherit from FedBopNew or any other method class.

    Design
    ------
    Client:
      1) Ordinary Bop on binary parameters:
             m <- (1-gamma)m + gamma*g
             flip w iff w*m > tau
      2) During each local round, count the realized binary-state occupancy
         relative to the round-start global state w0.
      3) Send one sparse event for coordinate j iff the opposite state occupied
         a strict majority of local minibatches.
      4) Non-binary trainables are optimized once by persistent Adam.
      5) After receiving the server state, apply the exact minimum feasible
         projection to client-private Bop momentum:
             m* = m - [w_new*m - tau]_+ w_new

    Server:
      1) Persistent Bop-style event memory.
      2) Same server dynamics as the successful FedBopNew annealed formulation.
      3) The old half-integer b=5.5 notation is removed algebraically:
             D = 2*floor(K/2) + 1
         and
             c / (floor(K/2)+0.5) == 2c / D
      4) Server update:
             beta_K = (1-gamma)^K
             beta_ref_K = (1-initial_gamma)^K
             event_gain = (1-beta_K)/(1-beta_ref_K)

             M <- beta_K*M
                  + w*tau*(2c/D)*event_gain

         Global binary coordinate flips iff
             w*M > tau

    Sync / decay policy:
      - gamma decay is unchanged: every decay_rounds rounds.
      - client Bop momentum consensus occurs ONLY ONCE, at the first decay
        boundary (round == decay_rounds).
      - later decay boundaries do NOT upload/download ternary momentum sidecars.
      - server event memory is never overwritten by client momentum consensus.

    With decay_rounds=200 and num_rounds=1000:
        sync  : 200
        decay : 200, 400, 600, 800
    """

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.contributors_percent = (
            float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        )

        # ------------------------------------------------------------------
        # Bop / annealing hyperparameters
        # ------------------------------------------------------------------
        self.gamma = self.get_arg(float, "gamma", 1e-4)
        self.initial_gamma = float(self.gamma)
        self.tau = self.get_arg(float, "tau", 1e-8)

        self.decay_factor = self.get_arg(float, "decay_factor", 1e-1)
        self.decay_rounds = self.get_arg(int, "decay_rounds", 200)

        # ------------------------------------------------------------------
        # One-shot ternary client-momentum synchronization codec
        # ------------------------------------------------------------------
        self.ternary_deadzone_mult = self.get_arg(
            float, "ternary_deadzone_mult", 0.5
        )
        self.ternary_min_scale_mult = self.get_arg(
            float, "ternary_min_scale_mult", 1e-3
        )

        # ------------------------------------------------------------------
        # Real-valued trainables
        # ------------------------------------------------------------------
        self.stats_lr = self.get_arg(float, "stats_lr", 1e-2)
        self.stats_optimizer = None
        self.stats_param_ids = None

        # ------------------------------------------------------------------
        # Client-private state
        # ------------------------------------------------------------------
        self.m_buffers = None
        self.round_global_weights = None
        self.temporal_opposite_counts = {}
        self.temporal_state_steps = {}
        self.round_local_vote_masks = {}
        self.has_local_update = False

        # ------------------------------------------------------------------
        # Server-private event memory
        # ------------------------------------------------------------------
        self.server_m_buffers = {}

        # ------------------------------------------------------------------
        # One-shot client momentum consensus prepared by the server
        # ------------------------------------------------------------------
        self.pending_shared_m_ternary = {}
        self.pending_sync_round = -1

        # ------------------------------------------------------------------
        # Round/gamma bookkeeping
        # ------------------------------------------------------------------
        self._scheduled_local_round = 1
        self._diag_client_round = 0
        self._last_decay_round = None

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        self.diag = self.get_arg(int, "diag", 1)
        self.diag_extended = self.get_arg(int, "diag_extended", 1)
        self.diag_layer_every = self.get_arg(int, "diag_layer_every", 20)

        self._diag_steps = 0
        self._diag_raw_flips = {}
        self._diag_grad_abs = {}

        self._diag_prev_client_events = {}
        self._diag_prev_server_flips = {}

    # =========================================================================
    # Framework interface / general helpers
    # =========================================================================

    def get_name(self):
        return "FedBopFinal"

    def init_method(self, server=None):
        if server is not None:
            self.server = server

    def select_clients_to_train(self, all_clients):
        if self.contributors_percent != 1.0:
            return self.select_random_clients(
                all_clients,
                self.contributors_percent,
            )
        return super().select_clients_to_train(all_clients)

    def _log(self, message):
        if not self.diag:
            return

        text = f"[FedBopFinalOneShotSync] {message}"

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
        return (
            self.diag_layer_every > 0
            and round_number % self.diag_layer_every == 0
        )

    @torch.no_grad()
    def _sign_binary(self, tensor):
        return torch.where(
            tensor >= 0,
            torch.ones_like(tensor),
            -torch.ones_like(tensor),
        )

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

    def _is_local_bn_running_state(self, key):
        """
        Client BN running statistics stay local.

        The server still receives/aggregates them for its own global evaluation,
        but clients do not overwrite their private running statistics on downlink.
        """
        lname = key.lower()
        return any(
            token in lname
            for token in (
                "running_mean",
                "running_var",
                "num_batches_tracked",
            )
        )

    def _prepare_stats_optimizer(self, model):
        params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad
            and not self._is_binary_param_tensor(name, p.data)
        ]

        ids = tuple(id(p) for p in params)

        if (
            self.stats_optimizer is None
            or self.stats_param_ids != ids
        ):
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
                f"STATS_OPT Adam params={len(params)} "
                f"lr={self.stats_lr:.3e} "
                f"persistent_state=1 outer_SGD=disabled"
            )

    @torch.no_grad()
    def _prepare_m_buffer(self, key, reference):
        if self.m_buffers is None:
            self.m_buffers = {}

        m = self.m_buffers.get(key)

        if m is None or m.shape != reference.shape:
            m = torch.zeros_like(reference)
        elif (
            m.device != reference.device
            or m.dtype != reference.dtype
        ):
            m = m.to(
                device=reference.device,
                dtype=reference.dtype,
            )

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
            and r % int(self.decay_rounds) == 0
        )

    def _is_sync_round(self, round_number):
        """
        One-shot client-momentum consensus at the FIRST decay boundary only.
        """
        try:
            r = int(round_number)
        except Exception:
            return False

        return (
            self.decay_rounds > 0
            and r == int(self.decay_rounds)
            and r > 0
            and r < int(self.num_of_rounds)
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

    # =========================================================================
    # Ternary codec / one-shot client momentum consensus
    # =========================================================================

    @torch.no_grad()
    def _pack_ternary_base3(self, q):
        """
        Pack q in {-1,0,+1}: five trits per uint8 because 3**5 = 243.
        """
        q = q.detach().to(
            device="cpu",
            dtype=torch.int16,
        ).view(-1)

        trits = q + 1
        n = int(trits.numel())

        pad = (-n) % 5
        if pad:
            # zero momentum is trit 1
            trits = torch.cat(
                [
                    trits,
                    torch.ones(
                        pad,
                        dtype=torch.int16,
                    ),
                ]
            )

        trits = trits.view(-1, 5).to(torch.int32)

        powers = torch.tensor(
            [1, 3, 9, 27, 81],
            dtype=torch.int32,
        ).view(1, 5)

        packed = (
            trits * powers
        ).sum(dim=1).to(torch.uint8)

        return packed, n

    @torch.no_grad()
    def _unpack_ternary_base3(self, packed, numel):
        packed = torch.as_tensor(
            packed,
            dtype=torch.uint8,
            device="cpu",
        ).view(-1)

        values = packed.to(
            torch.int32
        ).view(-1, 1)

        powers = torch.tensor(
            [1, 3, 9, 27, 81],
            dtype=torch.int32,
        ).view(1, 5)

        trits = torch.remainder(
            torch.div(
                values,
                powers,
                rounding_mode="floor",
            ),
            3,
        )

        return (
            trits.to(torch.int8) - 1
        ).view(-1)[: int(numel)]

    @torch.no_grad()
    def _quantize_momentum_ternary(self, m):
        """
        Bop-aware ternary momentum codec with one FP32 scale per binary layer.
        """
        x = m.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).view(-1)

        deadzone = (
            max(
                0.0,
                float(self.ternary_deadzone_mult),
            )
            * float(self.tau)
        )

        active = x.abs() > deadzone

        q = torch.zeros(
            x.numel(),
            dtype=torch.int8,
        )

        active_n = int(active.sum().item())

        if active_n > 0:
            q[active] = torch.sign(
                x[active]
            ).to(torch.int8)

            scale = float(
                x[active].abs().mean().item()
            )
        else:
            scale = float(self.tau)

        min_scale = max(
            1e-30,
            float(self.ternary_min_scale_mult)
            * float(self.tau),
        )

        scale = max(scale, min_scale)

        packed, numel = self._pack_ternary_base3(q)

        recon = (
            q.to(torch.float32)
            * float(scale)
        )

        mse = (
            float(
                ((recon - x) ** 2).mean().item()
            )
            if x.numel()
            else 0.0
        )

        mean_abs = (
            float(x.abs().mean().item())
            if x.numel()
            else 0.0
        )

        rel_rmse = (
            math.sqrt(mse)
            / max(1e-30, mean_abs)
        )

        entry = {
            "packed": packed,
            "numel": int(numel),
            "scale": torch.tensor(
                float(scale),
                dtype=torch.float32,
            ),
        }

        diag = {
            "nonzero": active_n,
            "packed_bytes": int(
                packed.numel()
            ),
            "scale": float(scale),
            "deadzone": float(deadzone),
            "rel_rmse": float(rel_rmse),
        }

        return entry, recon, diag

    @torch.no_grad()
    def _decode_ternary_entry(
        self,
        entry,
        device="cpu",
        dtype=torch.float32,
    ):
        q = self._unpack_ternary_base3(
            entry["packed"],
            entry["numel"],
        )

        x = (
            q.to(torch.float32)
            * float(entry["scale"])
        )

        return x.to(
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def _prepare_momentum_consensus(
        self,
        clients_models,
        global_model,
        round_number,
    ):
        """
        Build one shared low-bit CLIENT Bop momentum state.

        This is called only at the first decay boundary.  It does NOT modify
        server_m_buffers.
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

            if not self._is_binary_param_tensor(
                key,
                global_tensor,
            ):
                continue

            n = int(global_tensor.numel())

            m_sum = torch.zeros(
                n,
                dtype=torch.float32,
                device="cpu",
            )

            received = 0

            for _, packet in clients_models:
                entry = packet.get(
                    "momentum_ternary_model",
                    {},
                ).get(key)

                if entry is None:
                    continue

                decoded = self._decode_ternary_entry(
                    entry
                ).view(-1)

                if decoded.numel() != n:
                    raise ValueError(
                        f"Momentum-consensus size mismatch "
                        f"for {key}: {decoded.numel()} != {n}"
                    )

                m_sum.add_(decoded)
                received += 1

                total_uplink_bytes += (
                    int(
                        torch.as_tensor(
                            entry["packed"]
                        ).numel()
                    )
                    + 4
                )

            if received != k_clients:
                raise RuntimeError(
                    f"Sync round {round_number}: expected "
                    f"momentum sidecar for layer {key} "
                    f"from all {k_clients} participating clients; "
                    f"got {received}."
                )

            mean_m = (
                m_sum / float(received)
            )

            # Diagnostic: mean absolute deviation of client states.
            layer_abs_dev = 0.0

            for _, packet in clients_models:
                entry = packet[
                    "momentum_ternary_model"
                ][key]

                decoded = self._decode_ternary_entry(
                    entry
                ).view(-1)

                layer_abs_dev += float(
                    (
                        decoded - mean_m
                    ).abs().sum().item()
                )

            total_client_mad += (
                layer_abs_dev
                / float(received)
            )

            shared_entry, shared_m, qdiag = (
                self._quantize_momentum_ternary(
                    mean_m
                )
            )

            self.pending_shared_m_ternary[
                key
            ] = shared_entry

            total_numel += n

            total_downlink_bytes += (
                int(qdiag["packed_bytes"]) + 4
            )

            total_mean_abs += float(
                mean_m.abs().sum().item()
            )

            total_sync_abs += float(
                shared_m.abs().sum().item()
            )

            total_sync_nonzero += int(
                qdiag["nonzero"]
            )

            total_rel_rmse += float(
                qdiag["rel_rmse"]
            )

            sync_layers += 1

        if self.diag:
            self._log(
                f"M_SYNC_CONSENSUS "
                f"round={round_number} "
                f"K={k_clients} "
                f"gamma_before={self.gamma:.3e} "
                f"layers={sync_layers} "
                f"mean|m_mean|="
                f"{total_mean_abs/max(1,total_numel):.3e} "
                f"client_m_MAD="
                f"{total_client_mad/max(1,total_numel):.3e} "
                f"mean|m_shared|="
                f"{total_sync_abs/max(1,total_numel):.3e} "
                f"shared_nonzero_rate="
                f"{total_sync_nonzero/max(1,total_numel):.6e} "
                f"mean_rel_rmse="
                f"{total_rel_rmse/max(1,sync_layers):.3e} "
                f"ternary_uplink_total_MiB="
                f"{total_uplink_bytes/(1024.0**2):.3f} "
                f"ternary_uplink_per_client_MiB="
                f"{(total_uplink_bytes/max(1,k_clients))/(1024.0**2):.3f} "
                f"ternary_downlink_MiB="
                f"{total_downlink_bytes/(1024.0**2):.3f} "
                f"server_event_m_preserved=1"
            )

    # =========================================================================
    # Client local optimization
    # =========================================================================

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

            if not self._is_binary_param_tensor(
                name,
                param.data,
            ):
                continue

            # Binary parameter is owned by Bop.
            param.data.copy_(
                self._sign_binary(param.data)
            )

            m = self._prepare_m_buffer(
                name,
                param.data,
            )

            grad = param.grad.detach()

            # Ordinary Bop EMA.
            m.mul_(
                1.0 - self.gamma
            ).add_(
                grad,
                alpha=self.gamma,
            )

            # w*m > tau
            flip_mask = (
                (m.abs() > self.tau)
                & (
                    torch.sign(m)
                    == self._sign_binary(
                        param.data
                    )
                )
            )

            if self.diag:
                if name not in self._diag_raw_flips:
                    self._diag_raw_flips[name] = 0
                    self._diag_grad_abs[name] = 0.0

                self._diag_raw_flips[name] += int(
                    flip_mask.sum().item()
                )

                self._diag_grad_abs[name] += float(
                    grad.abs().mean().item()
                )

            # Realized local Bop state transition.
            param.data[
                flip_mask
            ] = -param.data[flip_mask]

            param.data.copy_(
                self._sign_binary(param.data)
            )

            # Temporal realized-state occupancy relative to round-start w0.
            w0_cpu = (
                None
                if self.round_global_weights is None
                else self.round_global_weights.get(
                    name
                )
            )

            if w0_cpu is not None:
                w0 = self._sign_binary(
                    w0_cpu.to(
                        device=param.device,
                        dtype=param.dtype,
                    )
                )

                count = (
                    self.temporal_opposite_counts.get(
                        name
                    )
                )

                if (
                    count is None
                    or count.shape != param.shape
                    or count.device != param.device
                ):
                    count = torch.zeros_like(
                        param.data,
                        dtype=torch.int16,
                    )

                count.add_(
                    (
                        self._sign_binary(
                            param.data
                        )
                        != w0
                    ).to(torch.int16)
                )

                self.temporal_opposite_counts[
                    name
                ] = count

                self.temporal_state_steps[
                    name
                ] = (
                    self.temporal_state_steps.get(
                        name,
                        0,
                    )
                    + 1
                )

            # Prevent outer optimizer from touching binary weights.
            param.grad = None

        # Non-binary trainables are updated ONCE by persistent Adam.
        if self.stats_optimizer is not None:
            self.stats_optimizer.step()
            self.stats_optimizer.zero_grad(
                set_to_none=True
            )

        # ClientExp: False means suppress outer optimizer.step().
        return False

    # =========================================================================
    # Client -> server packing
    # =========================================================================

    @torch.no_grad()
    def pack_client_model(
        self,
        raw_model,
        global_model,
        id,
    ):
        tensors = {}
        stats_model = {}

        round_number = int(
            self._scheduled_local_round
            or (
                self._diag_client_round
                + 1
            )
        )

        self._diag_client_round = round_number

        # One-shot sidecar only.
        sync_round = self._is_sync_round(
            round_number
        )

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

            if not self._is_binary_param_tensor(
                key,
                tensor,
            ):
                # Server receives non-binary state for aggregation/evaluation.
                stats_model[
                    key
                ] = tensor.detach().cpu()
                continue

            local_weight = self._sign_binary(
                tensor.detach().cpu()
            ).view(-1)

            global_weight = self._sign_binary(
                global_model[
                    key
                ].detach().cpu()
            ).view(-1)

            endpoint_flip = (
                local_weight
                != global_weight
            )

            steps = int(
                self.temporal_state_steps.get(
                    key,
                    0,
                )
            )

            counts = (
                self.temporal_opposite_counts.get(
                    key
                )
            )

            if (
                counts is None
                or counts.numel() != tensor.numel()
                or steps <= 0
            ):
                # Safety fallback only.
                vote_mask = endpoint_flip.clone()
                fallback = 1
            else:
                c = (
                    counts.detach()
                    .cpu()
                    .view(-1)
                    .to(torch.int32)
                )

                # Strict temporal majority.
                vote_mask = (
                    c
                    > (
                        int(steps) - c
                    )
                )

                fallback = 0

            self.round_local_vote_masks[
                key
            ] = vote_mask.clone()

            flip_indices = torch.nonzero(
                vote_mask,
                as_tuple=False,
            ).view(-1)

            tensors[key] = {
                "idx": flip_indices.tolist(),
            }

            if sync_round:
                m = self._prepare_m_buffer(
                    key,
                    tensor,
                )

                (
                    ternary_entry,
                    _,
                    ternary_diag,
                ) = self._quantize_momentum_ternary(
                    m
                )

                momentum_ternary_model[
                    key
                ] = ternary_entry

                sync_payload_bytes += (
                    int(
                        ternary_diag[
                            "packed_bytes"
                        ]
                    )
                    + 4
                )

                sync_nonzero += int(
                    ternary_diag[
                        "nonzero"
                    ]
                )

                sync_rel_rmse_sum += float(
                    ternary_diag[
                        "rel_rmse"
                    ]
                )

                sync_layers += 1

            # Preserve round-start reference for post-server projection.
            self.round_global_weights[
                key
            ] = global_weight.view_as(
                tensor
            ).clone()

            if self.diag:
                endpoint_n = int(
                    endpoint_flip.sum().item()
                )

                vote_n = int(
                    vote_mask.sum().item()
                )

                inter_n = int(
                    (
                        vote_mask
                        & endpoint_flip
                    ).sum().item()
                )

                union_n = int(
                    (
                        vote_mask
                        | endpoint_flip
                    ).sum().item()
                )

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
                    "jaccard": (
                        inter_n
                        / max(1, union_n)
                    ),
                    "fallback": fallback,
                }

        if self.diag:
            self._log(
                f"CLIENT_STATE_MAJ "
                f"round={round_number} "
                f"id={id} "
                f"gamma={self.gamma:.3e} "
                f"binary={total_binary} "
                f"endpoint={total_endpoint} "
                f"endpoint_rate="
                f"{total_endpoint/max(1,total_binary):.6e} "
                f"events={total_votes} "
                f"event_rate="
                f"{total_votes/max(1,total_binary):.6e} "
                f"event_endpoint_jaccard="
                f"{total_intersection/max(1,total_union):.6f}"
            )

            if self._layer_log_enabled(
                round_number
            ):
                for key, d in layer_diag.items():
                    self._log(
                        f"CLIENT_LAYER "
                        f"round={round_number} "
                        f"id={id} "
                        f"layer={key} "
                        f"n={d['n']} "
                        f"steps={d['steps']} "
                        f"endpoint={d['endpoint']} "
                        f"events={d['votes']} "
                        f"jaccard={d['jaccard']:.4f} "
                        f"fallback={d['fallback']}"
                    )

        packet = {
            "client_id": id,
            "tensors": tensors,
            "stats_model": stats_model,
        }

        if sync_round:
            packet[
                "momentum_ternary_model"
            ] = momentum_ternary_model

            if self.diag:
                self._log(
                    f"M_SYNC_UPLINK "
                    f"round={round_number} "
                    f"id={id} "
                    f"layers={sync_layers} "
                    f"payload_MiB="
                    f"{sync_payload_bytes/(1024.0**2):.3f} "
                    f"nonzero_rate="
                    f"{sync_nonzero/max(1,total_binary):.6e} "
                    f"mean_rel_rmse="
                    f"{sync_rel_rmse_sum/max(1,sync_layers):.3e}"
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

    # =========================================================================
    # Server-only diagnostic helper
    # =========================================================================

    @torch.no_grad()
    def _diag_sparse_intersection(
        self,
        a,
        b,
    ):
        if (
            a is None
            or b is None
            or a.numel() == 0
            or b.numel() == 0
        ):
            return 0

        a = a.to(
            device="cpu",
            dtype=torch.long,
        ).view(-1)

        b = b.to(
            device="cpu",
            dtype=torch.long,
        ).view(-1)

        pos = torch.searchsorted(
            b,
            a,
        )

        valid = pos < b.numel()

        if not bool(valid.any()):
            return 0

        av = a[valid]
        pv = pos[valid]

        return int(
            (
                b[pv] == av
            ).sum().item()
        )

    # =========================================================================
    # Server aggregation
    # =========================================================================

    @torch.no_grad()
    def aggregate(
        self,
        clients_models,
        global_model,
    ):
        if len(clients_models) == 0:
            return global_model

        round_number = (
            self._infer_round_number(
                clients_models
            )
        )

        k_clients = len(clients_models)

        gamma = float(self.gamma)

        if not (0.0 < gamma < 1.0):
            raise ValueError(
                "FedBopFinal requires 0 < gamma < 1"
            )

        # --------------------------------------------------------------
        # Same FedBopNew annealed server memory.
        # --------------------------------------------------------------
        beta_k = (
            1.0 - gamma
        ) ** k_clients

        beta_ref_k = (
            1.0
            - float(self.initial_gamma)
        ) ** k_clients

        if not (
            0.0 < beta_ref_k < 1.0
        ):
            raise ValueError(
                "Invalid initial_gamma for FedBopFinal"
            )

        # D replaces b = floor(K/2)+0.5 exactly:
        #       c/b = 2c/D
        separator = (
            2 * (k_clients // 2)
            + 1
        )

        event_gain = (
            1.0 - beta_k
        ) / (
            1.0 - beta_ref_k
        )

        total_binary = 0
        total_union = 0
        total_votes = 0
        total_flips = 0
        total_instant = 0

        support_hist_all = torch.zeros(
            k_clients + 1,
            dtype=torch.int64,
        )

        support_hist_flips = torch.zeros(
            k_clients + 1,
            dtype=torch.int64,
        )

        pair_intersections = 0
        pair_unions = 0

        same_client_repeat_intersection = 0
        same_client_repeat_union = 0
        same_client_current_events = 0

        server_flip_repeat_intersection = 0
        server_flip_repeat_union = 0
        server_current_flips = 0

        for key, global_tensor in global_model.items():
            if not torch.is_tensor(
                global_tensor
            ):
                continue

            if not self._is_binary_param_tensor(
                key,
                global_tensor,
            ):
                continue

            w = self._sign_binary(
                global_tensor.detach().cpu()
            ).view(-1)

            n = int(w.numel())

            vote_count = torch.zeros(
                n,
                dtype=torch.int16,
                device="cpu",
            )

            for client_label, packet in clients_models:
                entry = packet.get(
                    "tensors",
                    {},
                ).get(key)

                if entry is None:
                    continue

                idx = torch.as_tensor(
                    entry.get("idx", []),
                    dtype=torch.long,
                    device="cpu",
                ).view(-1)

                if (
                    self.diag
                    and self.diag_extended
                ):
                    client_id = packet.get(
                        "client_id",
                        client_label,
                    )

                    state_key = (
                        str(client_id),
                        key,
                    )

                    prev = (
                        self._diag_prev_client_events.get(
                            state_key
                        )
                    )

                    if (
                        prev is not None
                        and int(prev[0])
                        == int(round_number) - 1
                    ):
                        prev_idx = prev[1]

                        inter = (
                            self._diag_sparse_intersection(
                                idx,
                                prev_idx,
                            )
                        )

                        same_client_repeat_intersection += (
                            inter
                        )

                        same_client_repeat_union += (
                            int(idx.numel())
                            + int(prev_idx.numel())
                            - inter
                        )

                        same_client_current_events += int(
                            idx.numel()
                        )

                    self._diag_prev_client_events[
                        state_key
                    ] = (
                        int(round_number),
                        idx.clone(),
                    )

                if idx.numel() > 0:
                    vote_count[
                        idx
                    ] += 1

            server_m = (
                self.server_m_buffers.get(
                    key
                )
            )

            if (
                server_m is None
                or server_m.shape != w.shape
            ):
                server_m = torch.zeros_like(
                    w,
                    dtype=torch.float32,
                )
            else:
                server_m = server_m.to(
                    device="cpu",
                    dtype=torch.float32,
                )

            support_strength = (
                2.0
                * vote_count.to(
                    torch.float32
                )
                / float(separator)
            )

            aligned_fresh = (
                float(self.tau)
                * support_strength
                * float(event_gain)
            )

            server_m.mul_(
                beta_k
            ).add_(
                w.to(torch.float32)
                * aligned_fresh
            )

            self.server_m_buffers[
                key
            ] = server_m

            # Same global Bop trigger as FedBopNew.
            flip_mask_cpu = (
                w.to(torch.float32)
                * server_m
            ) > float(self.tau)

            # Current-round strict majority; diagnostics only.
            instant_mask = (
                2
                * vote_count.to(
                    torch.int32
                )
            ) > int(k_clients)

            if (
                self.diag
                and self.diag_extended
            ):
                support_hist_all += torch.bincount(
                    vote_count.to(
                        torch.long
                    ),
                    minlength=k_clients + 1,
                )[: k_clients + 1]

                if bool(
                    flip_mask_cpu.any()
                ):
                    support_hist_flips += torch.bincount(
                        vote_count[
                            flip_mask_cpu
                        ].to(torch.long),
                        minlength=k_clients + 1,
                    )[: k_clients + 1]

                c64 = vote_count.to(
                    torch.int64
                )

                pair_intersections += int(
                    (
                        (
                            c64
                            * (c64 - 1)
                        )
                        // 2
                    ).sum().item()
                )

                kmc = (
                    int(k_clients)
                    - c64
                )

                total_pairs = int(
                    k_clients
                    * (k_clients - 1)
                    // 2
                )

                pair_unions += int(
                    (
                        total_pairs
                        - (
                            (
                                kmc
                                * (kmc - 1)
                            )
                            // 2
                        )
                    ).sum().item()
                )

                prev_flip = (
                    self._diag_prev_server_flips.get(
                        key
                    )
                )

                if (
                    prev_flip is not None
                    and int(prev_flip[0])
                    == int(round_number) - 1
                ):
                    prev_mask = prev_flip[1]

                    inter = int(
                        (
                            flip_mask_cpu
                            & prev_mask
                        ).sum().item()
                    )

                    union = int(
                        (
                            flip_mask_cpu
                            | prev_mask
                        ).sum().item()
                    )

                    server_flip_repeat_intersection += (
                        inter
                    )

                    server_flip_repeat_union += (
                        union
                    )

                    server_current_flips += int(
                        flip_mask_cpu.sum().item()
                    )

                self._diag_prev_server_flips[
                    key
                ] = (
                    int(round_number),
                    flip_mask_cpu.clone(),
                )

            if bool(
                flip_mask_cpu.any()
            ):
                mask = flip_mask_cpu.to(
                    device=global_tensor.device,
                    dtype=torch.bool,
                )

                global_tensor.view(-1)[
                    mask
                ] *= -1

            global_tensor.copy_(
                self._sign_binary(
                    global_tensor
                )
            )

            if self.diag:
                total_binary += n

                total_union += int(
                    (
                        vote_count > 0
                    ).sum().item()
                )

                total_votes += int(
                    vote_count.sum().item()
                )

                total_flips += int(
                    flip_mask_cpu.sum().item()
                )

                total_instant += int(
                    instant_mask.sum().item()
                )

        # --------------------------------------------------------------
        # Non-binary aggregation.
        # Floating tensors: mean.
        # Integer state/counters: max.
        # --------------------------------------------------------------
        for key, global_tensor in global_model.items():
            if not torch.is_tensor(
                global_tensor
            ):
                continue

            if self._is_binary_param_tensor(
                key,
                global_tensor,
            ):
                continue

            received = []

            for _, packet in clients_models:
                stats = packet.get(
                    "stats_model",
                    {},
                )

                if key in stats:
                    received.append(
                        stats[key]
                    )

            if not received:
                continue

            if torch.is_floating_point(
                global_tensor
            ):
                acc = torch.zeros_like(
                    global_tensor,
                    dtype=torch.float32,
                    device=global_tensor.device,
                )

                for x in received:
                    acc.add_(
                        x.to(
                            device=global_tensor.device,
                            dtype=torch.float32,
                        )
                    )

                global_tensor.copy_(
                    (
                        acc
                        / float(
                            len(received)
                        )
                    ).to(
                        dtype=global_tensor.dtype
                    )
                )

            else:
                value = received[
                    0
                ].to(
                    device=global_tensor.device,
                    dtype=global_tensor.dtype,
                )

                for x in received[1:]:
                    value = torch.maximum(
                        value,
                        x.to(
                            device=global_tensor.device,
                            dtype=global_tensor.dtype,
                        ),
                    )

                global_tensor.copy_(
                    value
                )

        # --------------------------------------------------------------
        # One-shot client-state consensus only.
        # --------------------------------------------------------------
        if self._is_sync_round(
            round_number
        ):
            self._prepare_momentum_consensus(
                clients_models,
                global_model,
                round_number,
            )

        if self.diag:
            self._log(
                f"SERVER_BOP_FINAL "
                f"round={round_number} "
                f"K={k_clients} "
                f"gamma={gamma:.3e} "
                f"initial_gamma={self.initial_gamma:.3e} "
                f"beta_K={beta_k:.9f} "
                f"separator={separator} "
                f"event_gain={event_gain:.6e} "
                f"sync_now="
                f"{int(self._is_sync_round(round_number))} "
                f"sync_policy=first_decay_only "
                f"decay_every={self.decay_rounds} "
                f"tau={self.tau:.3e} "
                f"union_rate="
                f"{total_union/max(1,total_binary):.6e} "
                f"mean_support="
                f"{total_votes/max(1,total_union):.3f} "
                f"instant_majority_rate="
                f"{total_instant/max(1,total_binary):.6e} "
                f"global_flip_rate="
                f"{total_flips/max(1,total_binary):.6e}"
            )

        if (
            self.diag
            and self.diag_extended
        ):
            cross_client_micro_jaccard = (
                pair_intersections
                / max(1, pair_unions)
            )

            same_client_repeat_jaccard = (
                same_client_repeat_intersection
                / max(
                    1,
                    same_client_repeat_union,
                )
            )

            same_client_repeat_rate = (
                same_client_repeat_intersection
                / max(
                    1,
                    same_client_current_events,
                )
            )

            server_flip_repeat_rate = (
                server_flip_repeat_intersection
                / max(
                    1,
                    server_current_flips,
                )
            )

            server_flip_repeat_jaccard = (
                server_flip_repeat_intersection
                / max(
                    1,
                    server_flip_repeat_union,
                )
            )

            flip_hist_total = int(
                support_hist_flips.sum().item()
            )

            weighted_support = sum(
                i
                * int(
                    support_hist_flips[
                        i
                    ].item()
                )
                for i in range(
                    k_clients + 1
                )
            )

            mean_current_support_on_flips = (
                weighted_support
                / max(
                    1,
                    flip_hist_total,
                )
            )

            majority_start = (
                k_clients // 2
            ) + 1

            low_support_flips = int(
                support_hist_flips[
                    :majority_start
                ].sum().item()
            )

            zero_support_flips = int(
                support_hist_flips[
                    0
                ].item()
            )

            self._log(
                f"EVENT_ID_DIAG "
                f"round={round_number} "
                f"cross_client_micro_jaccard="
                f"{cross_client_micro_jaccard:.6e} "
                f"same_client_repeat_jaccard="
                f"{same_client_repeat_jaccard:.6e} "
                f"same_client_repeat_rate="
                f"{same_client_repeat_rate:.6e}"
            )

            self._log(
                f"SERVER_FLIP_DIAG "
                f"round={round_number} "
                f"mean_current_support_on_flips="
                f"{mean_current_support_on_flips:.6f} "
                f"low_support_flip_rate="
                f"{low_support_flips/max(1,flip_hist_total):.6e} "
                f"zero_support_flip_rate="
                f"{zero_support_flips/max(1,flip_hist_total):.6e} "
                f"consecutive_reflip_rate="
                f"{server_flip_repeat_rate:.6e} "
                f"consecutive_reflip_jaccard="
                f"{server_flip_repeat_jaccard:.6e}"
            )

            hist_all_text = ",".join(
                f"c{i}="
                f"{int(support_hist_all[i].item())}"
                for i in range(
                    k_clients + 1
                )
            )

            hist_flip_text = ",".join(
                f"c{i}="
                f"{int(support_hist_flips[i].item())}"
                for i in range(
                    k_clients + 1
                )
            )

            self._log(
                f"EVENT_SUPPORT_HIST "
                f"round={round_number} "
                f"{hist_all_text}"
            )

            self._log(
                f"FLIP_SUPPORT_HIST "
                f"round={round_number} "
                f"{hist_flip_text}"
            )

        return global_model

    # =========================================================================
    # Server -> client packing
    # =========================================================================

    @torch.no_grad()
    def pack_server_model(
        self,
        raw_model,
    ):
        bin_model = {}
        stats_model = {}

        for key, tensor in raw_model.items():
            if not torch.is_tensor(tensor):
                continue

            if self._is_binary_param_tensor(
                key,
                tensor,
            ):
                bin_model[key] = self._sign_binary(
                    tensor.detach().cpu()
                )
            else:
                stats_model[
                    key
                ] = tensor.detach().cpu()

        try:
            round_number = int(
                self.server.round_number
            )
        except Exception:
            round_number = -1

        old_gamma = float(self.gamma)
        did_decay = False

        # Decay occurs AFTER this round's aggregation, for the next round.
        if (
            round_number > 0
            and round_number < int(self.num_of_rounds)
            and self.decay_rounds > 0
            and (
                round_number
                % int(self.decay_rounds)
                == 0
            )
            and self._last_decay_round
            != round_number
        ):
            self.gamma *= self.decay_factor

            self._last_decay_round = (
                round_number
            )

            did_decay = True

        if (
            self.diag
            and did_decay
        ):
            self._log(
                f"GAMMA_DECAY "
                f"round={round_number} "
                f"gamma="
                f"{old_gamma:.3e}"
                f"->{self.gamma:.3e}"
            )

        momentum_sync = (
            int(round_number)
            == int(self.pending_sync_round)
            and len(
                self.pending_shared_m_ternary
            )
            > 0
        )

        next_round = (
            int(round_number) + 1
            if round_number >= 0
            else 1
        )

        payload = {
            "bin_model": bin_model,
            "stats_model": stats_model,
            "gamma": self.gamma,
            "next_round": next_round,
            "momentum_sync": bool(
                momentum_sync
            ),
            "momentum_sync_round": (
                int(round_number)
                if momentum_sync
                else -1
            ),
        }

        if momentum_sync:
            payload[
                "shared_m_ternary"
            ] = self.pending_shared_m_ternary

        return payload

    # =========================================================================
    # Server -> client unpacking / exact minimum feasible projection
    # =========================================================================

    @torch.no_grad()
    def unpack_server_model(
        self,
        packed_model,
        current_model,
    ):
        bin_model = packed_model[
            "bin_model"
        ]

        stats_model = packed_model[
            "stats_model"
        ]

        momentum_sync = bool(
            packed_model.get(
                "momentum_sync",
                False,
            )
        )

        momentum_sync_round = int(
            packed_model.get(
                "momentum_sync_round",
                -1,
            )
        )

        shared_m_ternary = (
            packed_model.get(
                "shared_m_ternary",
                {},
            )
        )

        self.gamma = float(
            packed_model[
                "gamma"
            ]
        )

        self._scheduled_local_round = int(
            packed_model.get(
                "next_round",
                self._diag_client_round + 1,
            )
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

            client_tensor = current_model[
                key
            ]

            w_new = self._sign_binary(
                server_tensor.to(
                    device=client_tensor.device,
                    dtype=client_tensor.dtype,
                )
            )

            m = self._prepare_m_buffer(
                key,
                client_tensor,
            )

            do_projection = False

            if momentum_sync:
                entry = shared_m_ternary.get(
                    key
                )

                if entry is None:
                    sync_missing_layers += 1

                    raise RuntimeError(
                        f"Missing shared momentum "
                        f"for layer {key} on sync round "
                        f"{momentum_sync_round}."
                    )

                shared_m = self._decode_ternary_entry(
                    entry,
                    device=m.device,
                    dtype=m.dtype,
                ).view_as(m)

                sync_numel += int(
                    m.numel()
                )

                sync_abs_private_to_shared += float(
                    (
                        m - shared_m
                    ).abs().sum().item()
                )

                sync_abs_shared += float(
                    shared_m.abs().sum().item()
                )

                m.copy_(
                    shared_m
                )

                do_projection = True

            elif (
                self.has_local_update
                and key in self.round_global_weights
            ):
                do_projection = True

            if do_projection:
                # Exact orthogonal projection onto:
                #     w_new * x <= tau
                aligned = (
                    w_new * m
                )

                excess = torch.clamp(
                    aligned - self.tau,
                    min=0.0,
                )

                violation = (
                    excess > 0
                )

                projected = (
                    m
                    - excess * w_new
                )

                if (
                    self.diag
                    and bool(
                        violation.any()
                    )
                ):
                    total_projected += int(
                        violation.sum().item()
                    )

                    total_delta += float(
                        (
                            projected[
                                violation
                            ]
                            - m[
                                violation
                            ]
                        ).abs().sum().item()
                    )

                m.copy_(
                    projected
                )

                if self.diag:
                    total_binary += int(
                        m.numel()
                    )

                    post_excess = torch.clamp(
                        w_new * m
                        - self.tau,
                        min=0.0,
                    )

                    if post_excess.numel() > 0:
                        max_post_violation = max(
                            max_post_violation,
                            float(
                                post_excess.max().item()
                            ),
                        )

            # Server binary state is authoritative for next local round.
            client_tensor.copy_(
                w_new
            )

            self.round_global_weights[
                key
            ] = (
                w_new.detach()
                .cpu()
                .clone()
            )

            self.temporal_opposite_counts[
                key
            ] = torch.zeros_like(
                client_tensor,
                dtype=torch.int16,
            )

            self.temporal_state_steps[
                key
            ] = 0

        # Keep client BN running stats local.  Other small real-valued state
        # follows the global/server state.  Adam moments remain private and
        # persistent because the Adam optimizer object is not recreated here.
        for key, server_tensor in stats_model.items():
            if key not in current_model:
                continue

            if self._is_local_bn_running_state(
                key
            ):
                continue

            target = current_model[
                key
            ]

            target.copy_(
                server_tensor.to(
                    device=target.device,
                    dtype=target.dtype,
                )
            )

        if (
            self.diag
            and momentum_sync
        ):
            self._log(
                f"M_SYNC_DOWNLINK "
                f"round={momentum_sync_round} "
                f"gamma_after={self.gamma:.3e} "
                f"numel={sync_numel} "
                f"missing_layers={sync_missing_layers} "
                f"mean|m_private-m_shared|="
                f"{sync_abs_private_to_shared/max(1,sync_numel):.3e} "
                f"mean|m_shared|="
                f"{sync_abs_shared/max(1,sync_numel):.3e} "
                f"post_sync_projected={total_projected} "
                f"post_sync_projection_rate="
                f"{total_projected/max(1,total_binary):.6e} "
                f"max_post_violation="
                f"{max_post_violation:.3e} "
                f"adam_state_preserved=1"
            )

        elif (
            self.diag
            and self.has_local_update
        ):
            self._log(
                f"MIN_PROJ "
                f"round={self._diag_client_round} "
                f"gamma={self.gamma:.3e} "
                f"projected={total_projected} "
                f"projection_rate="
                f"{total_projected/max(1,total_binary):.6e} "
                f"mean_abs_delta="
                f"{total_delta/max(1,total_projected):.3e} "
                f"max_post_violation="
                f"{max_post_violation:.3e} "
                f"adam_state_preserved=1"
            )

        self.has_local_update = False

        return current_model
