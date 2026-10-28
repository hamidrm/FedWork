import math
import pickle
import torch
from core.FederatedLearningClass import *
from utils.logger import *


class FedBop_wolmaj(FederatedLearningClass):
    """
    Final communication candidate for the harsh K=10, Dirichlet alpha=0.1 setup.
    It preserves the successful FedBop binary/optimizer arithmetic and global-BN
    recalibration, synchronizes Bop momentum every 200 rounds by default, and
    quantizes only the WIRE representation of BN beta and calibration moments.

    Controlled FedBop12 variant for testing Bop-consistent server event annealing while keeping the client rule, aggregation evidence, projection, sync cadence, and gamma schedule otherwise unchanged.

    Binary path
    -----------
    1) Each client trains binary weights with ordinary Bop.
    2) During a local round, each coordinate records whether its REALIZED binary
       state is opposite to the round-start global state. The client transmits one
       sparse event iff that opposite state occupied a strict majority of local
       minibatches.
    3) The server feeds client event support into one persistent Bop-style EMA and
       flips the global binary coordinate when the server Bop condition fires.
    4) After receiving the server-selected binary state, a client keeps its private
       Bop momentum unless that momentum would immediately violate the accepted
       global state. If it violates it, we apply the exact minimum-Euclidean-change
       projection onto the Bop-feasible half-space.

    There is deliberately NO reflection, NO canonical +/-tau reset, and NO
    phase-dependent rule switching.

    In this variant, client optimizer-state synchronization is deliberately
    DECOUPLED from gamma decay so that the two effects can be measured
    independently. Gamma still follows decay_rounds. Separately, every
    sync_rounds rounds, clients transmit a low-bit ternary snapshot of their
    private Bop EMA; the server averages these snapshots and broadcasts one shared
    low-bit EMA. The server's own event-EMA is NOT overwritten.

    If a synchronization round is also a gamma-decay round, clients train and the
    server aggregates with the OLD gamma; the server then forms the client-EMA
    consensus, decays gamma, and broadcasts the shared momentum together with the
    NEW gamma for the next local round.

    Server event annealing
    ----------------------
    The original FedBop12 recalibrated event_scale after every gamma decay so that
    the *effective* current-round event injection stayed constant. This variant
    deliberately removes that compensation while keeping the original phase-1
    calibration. A fixed pseudo-gradient scale is calibrated once using the
    initial gamma. Therefore, as gamma decays, server memory retention increases
    and the effective strength of fresh event evidence decreases in the same
    Bop-like direction:

        A_{r+1} = beta_K A_r + (tau*c/b) * event_gain
        event_gain = (1-beta_K) / (1-beta_ref_K)

    where beta_ref_K uses the initial gamma. Thus event_gain=1 in the first
    phase, approximately 0.1 after one 10x gamma decay, approximately 0.01 after
    two, etc. No new training hyperparameter is introduced.

    Real-valued path
    ----------------
    Non-binary trainable parameters (e.g. BN beta) are optimized only by Adam,
    matching the Bop setup. The framework's outer optimizer step is suppressed by
    returning False from after_backward(), so these parameters are not accidentally
    updated a second time by SGD. Adam moments persist across federated rounds.
    """

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.contributors_percent = (
            float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        )

        # Bop hyperparameters. One gamma is used consistently on client and server,
        # with the same decay schedule as the existing FedBop experiments.
        self.gamma = self.get_arg(float, "gamma", 1e-4)
        # Immutable reference gamma used only to calibrate the server pseudo-gradient
        # scale once. Unlike FedBop12, we do NOT re-normalize fresh evidence after
        # later gamma decays.
        self.initial_gamma = float(self.gamma)
        self.tau = self.get_arg(float, "tau", 1e-8)
        self.decay_factor = self.get_arg(float, "decay_factor", 1e-1)
        self.decay_rounds = self.get_arg(int, "decay_rounds", 200)

        # Client-momentum consensus cadence is independent of gamma decay.
        # Final-candidate severe non-IID schedule (for num_of_rounds=700):
        #   sync at 200,400,600
        #   decay at 200,400,600
        # (No sync is sent at the terminal round because _is_sync_round requires
        #  r < num_of_rounds.)
        self.sync_rounds = self.get_arg(int, "sync_rounds", 200)

        # Low-bit client-momentum consensus codec.
        # Five ternary symbols fit in one uint8 (3**5=243), plus one FP32 scale
        # per binary layer. These are codec parameters, not training policies.
        self.ternary_deadzone_mult = self.get_arg(float, "ternary_deadzone_mult", 0.5)
        self.ternary_min_scale_mult = self.get_arg(float, "ternary_min_scale_mult", 1e-3)

        # Adam for real-valued trainable parameters (BN beta in BinaryNet).
        self.stats_lr = self.get_arg(float, "stats_lr", 1e-2)
        self.stats_optimizer = None
        self.stats_param_ids = None

        # ------------------------------------------------------------------
        # Communication codecs for the real-valued inference state.
        # ------------------------------------------------------------------
        # Only the WIRE representation is quantized. Clients/server reconstruct
        # FP32 tensors before optimization/inference. Supported modes:
        #   fp32 : reference path
        #   fp16 : conservative 16-bit path (default)
        #   int8 : per-tensor symmetric INT8 + one FP32 scale
        #
        # BN beta is sent as the locally Adam-optimized ABSOLUTE beta on the
        # uplink, then averaged on the server. Downlink sends the aggregated
        # absolute beta. This exactly matches FedAvg-style model aggregation in
        # fp32 mode and avoids any ambiguity from the client's quantized round-
        # start copy.
        # BN calibration sends per-client first/second RAW MOMENTS (mean, E[x^2])
        # plus exact counts; this is algebraically equivalent to (sum, sumsq, n)
        # but is numerically safer for FP16 because raw sums can overflow.
        self.beta_comm_mode = self.get_arg(str, "beta_comm_mode", "fp16").lower()
        self.bn_stats_comm_mode = self.get_arg(str, "bn_stats_comm_mode", "fp16").lower()
        self._valid_comm_modes = ("fp32", "fp16", "int8")
        if self.beta_comm_mode not in self._valid_comm_modes:
            raise ValueError(
                f"beta_comm_mode must be one of {self._valid_comm_modes}; "
                f"got {self.beta_comm_mode!r}"
            )
        if self.bn_stats_comm_mode not in self._valid_comm_modes:
            raise ValueError(
                f"bn_stats_comm_mode must be one of {self._valid_comm_modes}; "
                f"got {self.bn_stats_comm_mode!r}"
            )

        # Current BinaryNet has fixed BN gamma=1 and client-private running stats.
        # They are not optimizer state and are not needed on the ordinary model
        # uplink/downlink once evaluation BN is recalibrated from global-model
        # activation moments. Keep a switch only for compatibility/debugging.
        self.omit_legacy_bn_buffers = self.get_arg(int, "omit_legacy_bn_buffers", 1)

        # Client-private Bop state.
        self.m_buffers = None
        self.round_global_weights = None
        self.temporal_opposite_counts = {}
        self.temporal_state_steps = {}
        self.round_local_vote_masks = {}
        self.has_local_update = False

        # Server-private Bop state.
        self.server_m_buffers = {}

        # Server -> client binary-wire state.  The first broadcast is an
        # absolute 1-bit snapshot.  Every later broadcast is encoded as the
        # sparse set of coordinates whose authoritative GLOBAL binary state
        # flipped since the previous broadcast.  Those indices use the exact
        # same Gap+Rice codec as the client uplink.  Keeping this state private
        # to the server changes only the wire representation; it never enters
        # Bop optimization or aggregation.
        self._server_downlink_prev_binary = {}

        # Prepared on an independent momentum-sync round after receiving all
        # client momentum sidecars. On rounds that are also decay boundaries it is
        # broadcast together with the decayed gamma; otherwise gamma is unchanged.
        # This is CLIENT-EMA consensus only; server_m_buffers retains its own
        # event-history semantics.
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

        # ------------------------------------------------------------------
        # Global-BN recalibration (evaluation-side only)
        # ------------------------------------------------------------------
        # The state-swap diagnostic showed that the dominant late-round
        # sawtooth comes from server BN running-stat mismatch, not from the
        # sparse global binary flips.  To test the strongest remedy with ONE
        # run, the server recalibrates BN immediately before every global
        # evaluation using the CURRENT fixed global model and a tiny balanced
        # calibration slice: by default one local training minibatch/client.
        #
        # This does NOT touch W, beta, Bop EMA, Adam state, event aggregation,
        # sync cadence, or gamma decay.  It only replaces the server's
        # evaluation running_mean/running_var.
        self.bn_recalibrate = self.get_arg(int, "bn_recalibrate", 1)
        self.bn_calib_batches_per_client = self.get_arg(
            int, "bn_calib_batches_per_client", 1
        )
        self.bn_calib_unbiased_var = self.get_arg(int, "bn_calib_unbiased_var", 1)
        self._server_eval_wrapped = False
        self._original_server_evaluate_model = None

    def get_name(self):
        return "FedBop without Local Majority"

    def init_method(self, server=None):
        if server is not None:
            self.server = server
            self._install_bn_recalibrated_evaluation(server)

    # -------------------------------------------------------------------------
    # Server-side global BN recalibration
    # -------------------------------------------------------------------------

    def _is_center_bn_module(self, module):
        """Recognize BinaryNet CenterOnlyBatchNorm without importing arch code."""
        return (
            hasattr(module, "running_mean")
            and hasattr(module, "running_var")
            and hasattr(module, "num_batches_tracked")
            and hasattr(module, "momentum")
            and torch.is_tensor(getattr(module, "running_mean", None))
            and torch.is_tensor(getattr(module, "running_var", None))
            and "batchnorm" in module.__class__.__name__.lower()
        )

    @torch.no_grad()
    def _recalibrate_server_bn(self):
        """Re-estimate BN moments on the CURRENT fixed global model.

        Simulator implementation of the distributed protocol:
          1) every client forwards a tiny TRAIN calibration slice through the
             same fixed current global (W,beta),
          2) it forms per-BN-channel (count, mean, E[x^2]),
          3) mean/E[x^2] are quantized according to bn_stats_comm_mode,
          4) the server decodes and count-weights the moments,
          5) the resulting global mean/variance are installed for evaluation.

        Raw data and activation maps are never part of the conceptual wire
        payload. In this in-process FedWork experiment the server can access
        client loaders directly, so quantize/dequantize is simulated explicitly
        to measure the same numerical effect as distributed communication.
        """
        if not int(self.bn_recalibrate):
            return
        if int(self.bn_calib_batches_per_client) <= 0:
            return
        if not hasattr(self, "server") or self.server is None:
            return

        server = self.server
        clients = getattr(server, "clients", None)
        if not clients:
            return

        model = server.global_model
        device = getattr(server, "executer", "cpu")

        bn_modules = []
        for name, module in model.named_modules():
            if self._is_center_bn_module(module):
                bn_modules.append((name, module))
        if not bn_modules:
            return

        training_flags = {id(m): bool(m.training) for m in model.modules()}
        old_momenta = {name: module.momentum for name, module in bn_modules}

        # Hooks write to the accumulator belonging to whichever client is being
        # forwarded right now. This lets us quantize each client's sufficient
        # statistics BEFORE server aggregation, matching a real FL uplink.
        active_accum = {}
        handles = []

        def make_hook(layer_name):
            def hook(_module, inputs):
                if not inputs:
                    return
                x = inputs[0].detach()
                if x.dim() < 2:
                    return
                reduce_dims = tuple(d for d in range(x.dim()) if d != 1)
                xd = x.to(dtype=torch.float64)
                layer_sum = xd.sum(dim=reduce_dims).cpu()
                layer_sumsq = (xd * xd).sum(dim=reduce_dims).cpu()
                count = int(x.numel() // x.shape[1])

                st = active_accum.get(layer_name)
                if st is None:
                    active_accum[layer_name] = {
                        "sum": layer_sum,
                        "sumsq": layer_sumsq,
                        "count": count,
                    }
                else:
                    st["sum"].add_(layer_sum)
                    st["sumsq"].add_(layer_sumsq)
                    st["count"] += count
            return hook

        before_mean = 0.0
        before_var = 0.0
        before_n = 0
        for name, module in bn_modules:
            before_mean += float(module.running_mean.detach().abs().sum().item())
            before_var += float(module.running_var.detach().abs().sum().item())
            before_n += int(module.running_mean.numel())
            handles.append(module.register_forward_pre_hook(make_hook(name)))

        # Server-side decoded pooled moment accumulators.
        pooled = {}
        total_batches = 0
        total_examples = 0
        total_stats_payload_bytes = 0
        contributing_clients = 0

        try:
            model.eval()
            for _, module in bn_modules:
                module.train(True)

            for client_name in sorted(clients.keys()):
                client = clients[client_name]
                loader = getattr(client, "dataset", None)
                if loader is None:
                    continue

                active_accum.clear()
                used = 0
                client_examples = 0
                for batch in loader:
                    if used >= int(self.bn_calib_batches_per_client):
                        break
                    if not isinstance(batch, (tuple, list)) or len(batch) < 1:
                        continue
                    inputs = batch[0].to(device)
                    try:
                        labels = batch[1].to(device) if len(batch) > 1 else None
                        if labels is not None:
                            inputs, _ = self.client_training_get_data(inputs, labels)
                    except Exception:
                        pass
                    model(inputs)
                    client_examples += int(inputs.shape[0])
                    total_examples += int(inputs.shape[0])
                    total_batches += 1
                    used += 1

                if used <= 0:
                    continue
                contributing_clients += 1

                # Client -> server calibration payload. We transmit normalized
                # moments rather than raw sum/sumsq so FP16 cannot overflow just
                # because spatial count is large. The server reconstructs the
                # exact same pooled moments by count weighting.
                for layer_name, st in active_accum.items():
                    count = int(st["count"])
                    if count <= 0:
                        continue
                    mean = (st["sum"] / float(count)).to(torch.float32)
                    second = (st["sumsq"] / float(count)).to(torch.float32)

                    mean_entry, mean_bytes = self._encode_wire_tensor(
                        mean, self.bn_stats_comm_mode
                    )
                    second_entry, second_bytes = self._encode_wire_tensor(
                        second, self.bn_stats_comm_mode
                    )
                    # One exact int64 count/layer.
                    total_stats_payload_bytes += int(mean_bytes + second_bytes + 8)

                    mean_dec = self._decode_wire_tensor(
                        mean_entry, device="cpu", dtype=torch.float64
                    )
                    second_dec = self._decode_wire_tensor(
                        second_entry, device="cpu", dtype=torch.float64
                    )

                    pst = pooled.get(layer_name)
                    if pst is None:
                        pooled[layer_name] = {
                            "weighted_mean": mean_dec * float(count),
                            "weighted_second": second_dec * float(count),
                            "count": count,
                        }
                    else:
                        pst["weighted_mean"].add_(mean_dec * float(count))
                        pst["weighted_second"].add_(second_dec * float(count))
                        pst["count"] += count
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        delta_mean_sum = 0.0
        delta_var_sum = 0.0
        delta_n = 0
        calibrated_layers = 0
        for name, module in bn_modules:
            st = pooled.get(name)
            if st is None or int(st["count"]) <= 0:
                continue

            count = int(st["count"])
            mean64 = st["weighted_mean"] / float(count)
            second64 = st["weighted_second"] / float(count)
            var64 = second64 - mean64 * mean64
            var64.clamp_(min=0.0)
            if int(self.bn_calib_unbiased_var) and count > 1:
                var64.mul_(float(count) / float(count - 1))

            mean_new = mean64.to(
                device=module.running_mean.device,
                dtype=module.running_mean.dtype,
            )
            var_new = var64.to(
                device=module.running_var.device,
                dtype=module.running_var.dtype,
            )

            delta_mean_sum += float(
                (module.running_mean.detach() - mean_new).abs().sum().item()
            )
            delta_var_sum += float(
                (module.running_var.detach() - var_new).abs().sum().item()
            )
            delta_n += int(mean_new.numel())

            module.running_mean.copy_(mean_new)
            module.running_var.copy_(var_new)
            try:
                module.num_batches_tracked.fill_(int(total_batches))
            except Exception:
                pass
            calibrated_layers += 1

        for name, module in bn_modules:
            module.momentum = old_momenta[name]
        for m in model.modules():
            m.train(training_flags[id(m)])

        if self.diag:
            try:
                round_number = int(server.round_number)
            except Exception:
                round_number = -1
            per_client_kib = (
                total_stats_payload_bytes / max(1, contributing_clients) / 1024.0
            )
            self._log(
                f"BN_GLOBAL_RECALIB round={round_number} "
                f"layers={calibrated_layers} clients={contributing_clients} "
                f"batches={total_batches} examples={total_examples} "
                f"batches_per_client={int(self.bn_calib_batches_per_client)} "
                f"stats_comm={self.bn_stats_comm_mode} "
                f"stats_uplink_total_KiB={total_stats_payload_bytes/1024.0:.3f} "
                f"stats_uplink_per_client_KiB={per_client_kib:.3f} "
                f"mean_abs_delta_running_mean={delta_mean_sum/max(1,delta_n):.6e} "
                f"mean_abs_delta_running_var={delta_var_sum/max(1,delta_n):.6e} "
                f"moment_repr=mean_second fixed_global_model=1 test_data_used=0"
            )

    def _install_bn_recalibrated_evaluation(self, server):
        """Wrap Server.evaluate_model once; training/aggregation are untouched."""
        if self._server_eval_wrapped:
            return
        original = server.evaluate_model
        self._original_server_evaluate_model = original

        def evaluate_with_global_bn_recalibration(*args, **kwargs):
            self._recalibrate_server_bn()
            return original(*args, **kwargs)

        server.evaluate_model = evaluate_with_global_bn_recalibration
        self._server_eval_wrapped = True
        if self.diag:
            self._log(
                "BN_GLOBAL_RECALIB_INSTALL enabled=1 "
                f"batches_per_client={int(self.bn_calib_batches_per_client)} "
                f"beta_comm={self.beta_comm_mode} stats_comm={self.bn_stats_comm_mode} "
                f"sync_every={self.sync_rounds} decay_every={self.decay_rounds}"
            )

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
        text = f"[FedBopFinalQuant200] {message}"
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

    def _is_bn_beta_key(self, key):
        # BinaryNet CenterOnlyBatchNorm exposes trainable beta as bn*.bias.
        lname = str(key).lower()
        return lname.endswith(".bias") and "bn" in lname

    def _is_legacy_bn_buffer_key(self, key):
        lname = str(key).lower()
        return (
            "running_mean" in lname
            or "running_var" in lname
            or "num_batches_tracked" in lname
            # CenterOnlyBatchNorm fixed gamma=1 buffer. Binary weights are
            # already excluded by _is_binary_param_tensor.
            or ("bn" in lname and lname.endswith(".weight"))
        )

    @torch.no_grad()
    def _encode_wire_tensor(self, tensor, mode):
        """Quantize one floating tensor for simulated wire transmission.

        Returns a self-contained CPU entry and a payload-byte estimate. Metadata
        keys/shape descriptors are intentionally excluded from the byte estimate;
        the dominant numeric payload and scale are counted.
        """
        mode = str(mode).lower()
        x = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        n = int(x.numel())

        if mode == "fp32":
            return {"mode": "fp32", "data": x}, 4 * n

        if mode == "fp16":
            # A single scale protects second moments from FP16 overflow. For
            # ordinary beta/means the scale is normally exactly 1.0.
            max_abs = float(x.abs().max().item()) if n else 0.0
            scale = max(1.0, max_abs / 60000.0)
            q = (x / scale).to(torch.float16)
            return {"mode": "fp16", "data": q, "scale": float(scale)}, 2 * n + 4

        if mode == "int8":
            max_abs = float(x.abs().max().item()) if n else 0.0
            scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
            q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
            return {"mode": "int8", "data": q, "scale": float(scale)}, n + 4

        raise ValueError(f"Unsupported wire mode: {mode}")

    @torch.no_grad()
    def _decode_wire_tensor(self, entry, device="cpu", dtype=torch.float32):
        mode = str(entry.get("mode", "fp32")).lower()
        data = entry["data"]
        if mode == "fp32":
            x = data.to(device=device, dtype=torch.float32)
        elif mode == "fp16":
            x = data.to(device=device, dtype=torch.float32) * float(entry.get("scale", 1.0))
        elif mode == "int8":
            x = data.to(device=device, dtype=torch.float32) * float(entry.get("scale", 1.0))
        else:
            raise ValueError(f"Unsupported wire mode in packet: {mode}")
        return x.to(dtype=dtype)

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

    def _is_sync_round(self, round_number):
        try:
            r = int(round_number)
        except Exception:
            return False
        return (
            self.sync_rounds > 0
            and r > 0
            and r < int(self.num_of_rounds)
            and (r % int(self.sync_rounds) == 0)
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
    def _prepare_momentum_consensus(self, clients_models, global_model, round_number):
        """Average client Bop EMA states on an independent momentum-sync boundary.

        The ordinary sparse event aggregation has already selected the global
        binary model for this round. Here we ONLY build a shared CLIENT momentum
        state for the next local round. The server event momentum remains intact.
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
                        f"Momentum-consensus size mismatch for {key}: "
                        f"{decoded.numel()} != {n}"
                    )
                m_sum.add_(decoded)
                received += 1
                total_uplink_bytes += int(torch.as_tensor(entry["packed"]).numel()) + 4

            if received != k_clients:
                raise RuntimeError(
                    f"Sync round {round_number}: expected momentum sidecar for "
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
                f"M_SYNC_CONSENSUS round={round_number} K={k_clients} "
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


    # -------------------------------------------------------------------------
    # Gap + Rice codec for sparse binary events
    # -------------------------------------------------------------------------

    @staticmethod
    def _indices_to_gaps(indices):
        """
        Convert sorted unique non-negative indices to non-negative gaps.

        Example:
            [12, 15, 19, 35] -> [12, 2, 3, 15]

        For i > 0:
            gap[i] = idx[i] - idx[i-1] - 1
        """
        idx = torch.as_tensor(indices, dtype=torch.long, device="cpu").view(-1)

        if idx.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        if bool((idx < 0).any()):
            raise ValueError("Gap-Rice codec received a negative index.")

        if idx.numel() > 1:
            if bool((idx[1:] <= idx[:-1]).any()):
                raise ValueError(
                    "Gap-Rice codec requires strictly increasing unique indices."
                )

        gaps = idx.clone()
        if idx.numel() > 1:
            gaps[1:] = idx[1:] - idx[:-1] - 1

        return gaps


    @staticmethod
    def _gaps_to_indices(gaps):
        """
        Inverse of _indices_to_gaps().
        """
        gaps = torch.as_tensor(
            gaps,
            dtype=torch.long,
            device="cpu"
        ).view(-1)

        if gaps.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        if bool((gaps < 0).any()):
            raise ValueError("Gap-Rice decoder produced a negative gap.")

        idx = torch.empty_like(gaps)
        idx[0] = gaps[0]

        if gaps.numel() > 1:
            # idx[i] = idx[i-1] + gap[i] + 1
            idx[1:] = gaps[1:] + 1
            idx = torch.cumsum(idx, dim=0)

        return idx


    @staticmethod
    def _choose_rice_k(gaps):
        """
        Select the Rice parameter k that minimizes the exact code length:

            B(k) = sum_i floor(gap_i / 2^k) + N * (k + 1)

        This includes:
          - q_i unary bits: q_i + 1
          - k remainder bits per value

        The search is exact over all useful k values.
        """
        gaps = torch.as_tensor(
            gaps,
            dtype=torch.long,
            device="cpu"
        ).view(-1)

        n = int(gaps.numel())
        if n == 0:
            return 0, 0

        max_gap = int(gaps.max().item())

        # No optimum can require k much larger than the bit width of max_gap.
        max_k = max(0, int(max_gap).bit_length())

        best_k = 0
        best_bits = None

        for k in range(max_k + 1):
            quotient_bits = int((gaps >> k).sum().item())
            total_bits = quotient_bits + n * (k + 1)

            if best_bits is None or total_bits < best_bits:
                best_bits = total_bits
                best_k = k

        return int(best_k), int(best_bits)


    @staticmethod
    def _rice_encode_gaps(gaps, k):
        """
        Encode non-negative integer gaps using Rice coding.

        Codeword for x:
            q = x >> k
            r = x & ((1 << k) - 1)

            [ q times '1' ][ '0' ][ r in exactly k bits ]

        Bits are packed MSB-first into a bytes object.
        Returns:
            payload_bytes, exact_valid_bit_count
        """
        gaps = torch.as_tensor(
            gaps,
            dtype=torch.long,
            device="cpu"
        ).view(-1)

        k = int(k)
        if k < 0:
            raise ValueError("Rice parameter k must be non-negative.")

        if gaps.numel() == 0:
            return b"", 0

        out = bytearray()
        current_byte = 0
        used_bits = 0
        total_bits = 0

        def write_bits(value, count):
            nonlocal current_byte, used_bits, total_bits

            value = int(value)
            count = int(count)

            while count > 0:
                free = 8 - used_bits
                take = min(free, count)

                shift = count - take
                mask = (1 << take) - 1
                chunk = (value >> shift) & mask

                current_byte = (current_byte << take) | chunk
                used_bits += take
                total_bits += take
                count -= take

                if used_bits == 8:
                    out.append(current_byte & 0xFF)
                    current_byte = 0
                    used_bits = 0

        for x_tensor in gaps:
            x = int(x_tensor.item())
            if x < 0:
                raise ValueError("Rice coding only supports non-negative integers.")

            q = x >> k
            r = x & ((1 << k) - 1) if k > 0 else 0

            # Unary q: q ones followed by one zero.
            # Write long unary runs in chunks to avoid one Python iteration/bit.
            q_left = q
            while q_left >= 64:
                write_bits((1 << 64) - 1, 64)
                q_left -= 64

            if q_left > 0:
                write_bits((1 << q_left) - 1, q_left)

            write_bits(0, 1)

            if k > 0:
                write_bits(r, k)

        # Pad the final partial byte with zero bits on the right.
        if used_bits > 0:
            current_byte <<= (8 - used_bits)
            out.append(current_byte & 0xFF)

        return bytes(out), int(total_bits)


    @staticmethod
    def _rice_decode_gaps(payload, k, count, num_bits):
        """
        Decode a Rice-coded bytes payload back to a CPU int64 gap tensor.
        """
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"Rice payload must be bytes-like, got {type(payload).__name__}."
            )

        payload = bytes(payload)
        k = int(k)
        count = int(count)
        num_bits = int(num_bits)

        if k < 0 or count < 0 or num_bits < 0:
            raise ValueError("Invalid Rice codec metadata.")

        if count == 0:
            if num_bits != 0:
                raise ValueError(
                    "Rice packet has zero values but non-zero valid bit count."
                )
            return torch.empty(0, dtype=torch.long)

        if num_bits > len(payload) * 8:
            raise ValueError(
                "Rice packet declares more valid bits than are present."
            )

        bit_pos = 0
        values = []

        def read_bit():
            nonlocal bit_pos

            if bit_pos >= num_bits:
                raise ValueError("Unexpected end of Rice bitstream.")

            byte_index = bit_pos >> 3
            bit_in_byte = 7 - (bit_pos & 7)
            bit = (payload[byte_index] >> bit_in_byte) & 1
            bit_pos += 1
            return bit

        def read_fixed(nbits):
            value = 0
            for _ in range(int(nbits)):
                value = (value << 1) | read_bit()
            return value

        for _ in range(count):
            q = 0

            while True:
                bit = read_bit()
                if bit == 0:
                    break
                q += 1

            r = read_fixed(k) if k > 0 else 0
            x = (q << k) | r
            values.append(x)

        return torch.tensor(values, dtype=torch.long)


    @classmethod
    def _encode_sparse_indices_gap_rice(cls, indices):
        """
        Encode sorted sparse indices.

        Wire format is a compact tuple:
            ("gr1", k, count, num_bits, payload_bytes)

        Tuple metadata is deliberately compact because the framework profiler
        measures the serialized object that is actually transmitted.
        """
        gaps = cls._indices_to_gaps(indices)
        k, predicted_bits = cls._choose_rice_k(gaps)
        payload, num_bits = cls._rice_encode_gaps(gaps, k)

        if int(num_bits) != int(predicted_bits):
            raise RuntimeError(
                f"Rice bit-count mismatch: predicted={predicted_bits}, "
                f"encoded={num_bits}"
            )

        return (
            "gr1",
            int(k),
            int(gaps.numel()),
            int(num_bits),
            payload,
        )


    @classmethod
    def _decode_sparse_indices_gap_rice(cls, entry):
        """
        Decode the compact ("gr1", ...) wire tuple back to sorted indices.
        """
        if not isinstance(entry, (tuple, list)) or len(entry) != 5:
            raise ValueError("Invalid Gap-Rice sparse-index entry.")

        codec, k, count, num_bits, payload = entry

        if codec != "gr1":
            raise ValueError(f"Unsupported sparse-index codec: {codec!r}")

        gaps = cls._rice_decode_gaps(
            payload=payload,
            k=k,
            count=count,
            num_bits=num_bits,
        )

        return cls._gaps_to_indices(gaps)

    # ---------------------------------------------------------------------
    # Dense 1-bit binary snapshot codec (downlink bootstrap / safe fallback)
    # ---------------------------------------------------------------------

    @staticmethod
    def _encode_binary_snapshot_1bit(tensor):
        """Pack signs into one bit/coordinate, MSB-first.

        Wire format:
            ("bp1", numel, payload_bytes)

        +1 is encoded as bit 1 and -1 as bit 0.  Zero follows FedBop's
        existing tie rule (>=0 -> +1).  This codec is used only when a client
        has no delta base yet (initial broadcast), or when an absolute 1-bit
        snapshot is smaller on the actual pickle wire than a Rice delta.
        """
        x = torch.as_tensor(tensor, device="cpu").contiguous().view(-1)
        n = int(x.numel())
        if n == 0:
            return ("bp1", 0, b"")

        bits = (x >= 0).to(torch.uint8)
        pad = (-n) % 8
        if pad:
            bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8)])

        bits = bits.view(-1, 8).to(torch.int16)
        weights = torch.tensor(
            [128, 64, 32, 16, 8, 4, 2, 1],
            dtype=torch.int16,
        ).view(1, 8)
        packed = (bits * weights).sum(dim=1).to(torch.uint8).contiguous()
        return ("bp1", n, packed.numpy().tobytes())

    @staticmethod
    def _decode_binary_snapshot_1bit(entry, device="cpu", dtype=torch.float32):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError("Invalid 1-bit binary snapshot entry.")

        codec, numel, payload = entry
        if codec != "bp1":
            raise ValueError(f"Unsupported binary snapshot codec: {codec!r}")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"Binary snapshot payload must be bytes-like, got {type(payload).__name__}."
            )

        n = int(numel)
        if n < 0:
            raise ValueError("Binary snapshot numel must be non-negative.")
        payload = bytes(payload)
        expected_bytes = (n + 7) // 8
        if len(payload) != expected_bytes:
            raise ValueError(
                f"Binary snapshot payload length mismatch: {len(payload)} != {expected_bytes}."
            )
        if n == 0:
            return torch.empty(0, device=device, dtype=dtype)

        # bytearray gives torch a writable buffer and avoids the non-writable
        # bytes warning from torch.frombuffer.  Expansion is done in eight
        # vectorized strided assignments, not one Python operation per bit.
        packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        out = torch.empty(int(packed.numel()) * 8, dtype=torch.int8)
        for bit in range(8):
            out[bit::8] = ((packed >> (7 - bit)) & 1).to(torch.int8)
        signs = out[:n].to(torch.float32).mul_(2.0).sub_(1.0)
        return signs.to(device=device, dtype=dtype)

    @classmethod
    def _encode_server_binary_wire(cls, current_sign, previous_sign=None):
        """Encode one authoritative global binary tensor losslessly.

        If there is no synchronized previous server state, send a dense 1-bit
        snapshot.  Otherwise encode only sign-flip indices with Gap+Rice.  For
        safety, compare the *actual default-pickle size* of the Rice delta and
        the 1-bit snapshot and use the smaller entry.  This can only change
        communication cost, never the reconstructed global binary state.

        Returns:
            entry, diag
        """
        cur = torch.as_tensor(
            current_sign, dtype=torch.float32, device="cpu"
        ).contiguous().view(-1)
        cur = torch.where(cur >= 0, torch.ones_like(cur), -torch.ones_like(cur))
        snapshot = cls._encode_binary_snapshot_1bit(cur)
        snapshot_pickle_bytes = len(pickle.dumps(snapshot))

        if previous_sign is None:
            return snapshot, {
                "mode": "snapshot",
                "events": 0,
                "rice_k": 0,
                "rice_valid_bits": 0,
                "rice_data_bytes": 0,
                "snapshot_data_bytes": len(snapshot[2]),
                "selected_pickle_bytes": snapshot_pickle_bytes,
            }

        prev = torch.as_tensor(
            previous_sign, dtype=torch.float32, device="cpu"
        ).contiguous().view(-1)
        prev = torch.where(prev >= 0, torch.ones_like(prev), -torch.ones_like(prev))
        if int(prev.numel()) != int(cur.numel()):
            return snapshot, {
                "mode": "snapshot",
                "events": 0,
                "rice_k": 0,
                "rice_valid_bits": 0,
                "rice_data_bytes": 0,
                "snapshot_data_bytes": len(snapshot[2]),
                "selected_pickle_bytes": snapshot_pickle_bytes,
            }

        flip_indices = torch.nonzero(cur != prev, as_tuple=False).view(-1).to(torch.long)
        rice = cls._encode_sparse_indices_gap_rice(flip_indices)
        rice_pickle_bytes = len(pickle.dumps(rice))

        if rice_pickle_bytes <= snapshot_pickle_bytes:
            _, k, count, valid_bits, rice_payload = rice
            return rice, {
                "mode": "rice_delta",
                "events": int(count),
                "rice_k": int(k),
                "rice_valid_bits": int(valid_bits),
                "rice_data_bytes": len(rice_payload),
                "snapshot_data_bytes": len(snapshot[2]),
                "selected_pickle_bytes": rice_pickle_bytes,
            }

        return snapshot, {
            "mode": "snapshot_fallback",
            "events": int(flip_indices.numel()),
            "rice_k": int(rice[1]),
            "rice_valid_bits": int(rice[3]),
            "rice_data_bytes": len(rice[4]),
            "snapshot_data_bytes": len(snapshot[2]),
            "selected_pickle_bytes": snapshot_pickle_bytes,
        }


    @torch.no_grad()
    def pack_client_model(self, raw_model, global_model, id):
        tensors = {}
        # Generic compatibility path for any non-BN, non-binary state.
        stats_model = {}
        # Trainable CenterOnly-BN beta is the only real-valued model parameter
        # in BinaryNet. Uplink sends quantized DELTA beta, not absolute beta.
        beta_model = {}
        beta_payload_bytes = 0
        beta_numel = 0

        round_number = int(self._scheduled_local_round or (self._diag_client_round + 1))
        self._diag_client_round = round_number
        sync_round = self._is_sync_round(round_number)
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

        # Gap-Rice communication diagnostics.
        rice_data_bytes = 0
        rice_valid_bits = 0
        fixed_index_bits = 0
        rice_layers = 0

        layer_diag = {}

        for key, tensor in raw_model.items():
            if not torch.is_tensor(tensor):
                continue

            if not self._is_binary_param_tensor(key, tensor):
                if self._is_bn_beta_key(key) and key in global_model:
                    local_beta = tensor.detach().to(device="cpu", dtype=torch.float32)
                    entry, nbytes = self._encode_wire_tensor(local_beta, self.beta_comm_mode)
                    beta_model[key] = entry
                    beta_payload_bytes += int(nbytes)
                    beta_numel += int(local_beta.numel())
                    continue

                if int(self.omit_legacy_bn_buffers) and self._is_legacy_bn_buffer_key(key):
                    # running mean/var are replaced by global-model recalibration;
                    # num_batches_tracked and fixed BN gamma=1 are not model updates.
                    continue

                stats_model[key] = tensor.detach().cpu()
                continue

            local_weight = self._sign_binary(tensor.detach().cpu()).view(-1)
            global_weight = self._sign_binary(global_model[key].detach().cpu()).view(-1)
            endpoint_flip = local_weight != global_weight

            steps = int(self.temporal_state_steps.get(key, 0))
            counts = self.temporal_opposite_counts.get(key)

            if counts is None or counts.numel() != tensor.numel() or steps <= 0:
                vote_mask = endpoint_flip.clone()
                fallback = 1
            else:
                c = counts.detach().cpu().view(-1).to(torch.int32)
                vote_mask = endpoint_flip.clone()
                fallback = 0

            self.round_local_vote_masks[key] = vote_mask.clone()
            flip_indices = torch.nonzero(
                vote_mask,
                as_tuple=False
            ).view(-1).to(device="cpu", dtype=torch.long)

            rice_entry = self._encode_sparse_indices_gap_rice(
                flip_indices
            )
            tensors[key] = rice_entry

            # Logical codec diagnostics.  The profiler still measures the
            # complete serialized packet, including tuple/dict metadata.
            _, rice_k, rice_count, rice_bits, rice_payload = rice_entry

            index_width = max(
                1,
                int(max(1, tensor.numel() - 1)).bit_length()
            )

            layer_fixed_bits = int(flip_indices.numel()) * index_width

            layer_ratio = (
                layer_fixed_bits / max(1, int(rice_bits))
            )

            self._log(
                f"GAP_RICE_LAYER "
                f"round={round_number} "
                f"id={id} "
                f"layer={key} "
                f"numel={tensor.numel()} "
                f"events={rice_count} "
                f"k={rice_k} "
                f"valid_bits={rice_bits} "
                f"data_bytes={len(rice_payload)} "
                f"fixed_index_bits={layer_fixed_bits} "
                f"codec_ratio={layer_ratio:.4f}x"
            )

            rice_data_bytes += len(rice_payload)
            rice_valid_bits += int(rice_bits)
            rice_layers += 1

            # Reference cost: minimum fixed-width index representation for
            # this layer (not Python-list/pickle size).
            index_width = max(
                1,
                int(max(1, tensor.numel() - 1)).bit_length()
            )
            fixed_index_bits += int(flip_indices.numel()) * index_width

            if sync_round:
                m = self._prepare_m_buffer(key, tensor)
                ternary_entry, _, ternary_diag = self._quantize_momentum_ternary(m)
                momentum_ternary_model[key] = ternary_entry
                sync_payload_bytes += int(ternary_diag["packed_bytes"]) + 4
                sync_nonzero += int(ternary_diag["nonzero"])
                sync_rel_rmse_sum += float(ternary_diag["rel_rmse"])
                sync_layers += 1

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
            self._log(
                f"GAP_RICE_UPLINK round={round_number} id={id} "
                f"layers={rice_layers} events={total_votes} "
                f"valid_bits={rice_valid_bits} data_bytes={rice_data_bytes} "
                f"fixed_index_bits={fixed_index_bits} "
                f"codec_ratio={fixed_index_bits/max(1,rice_valid_bits):.4f}x"
            )
            self._log(
                f"BETA_UPLINK round={round_number} id={id} mode={self.beta_comm_mode} "
                f"numel={beta_numel} payload_KiB={beta_payload_bytes/1024.0:.3f} "
                f"representation=absolute adam_state_local=1"
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
            "beta_model": beta_model,
            "stats_model": stats_model,
        }
        if sync_round:
            packet["momentum_ternary_model"] = momentum_ternary_model
            if self.diag:
                self._log(
                    f"M_SYNC_UPLINK round={round_number} id={id} "
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
        """
        Decode Gap+Rice sparse event indices after the packet has already
        crossed the framework communication/profiler boundary.

        The returned structure intentionally restores the legacy
            {"idx": [...]}
        representation so aggregate() does not need any algorithmic change.

        Backward compatibility:
            legacy {"idx": [...]} entries are accepted unchanged.
        """
        packet = packed_model

        tensors = packet.get("tensors", {})
        decoded_tensors = {}

        for key, entry in tensors.items():

            # New compact wire representation.
            if (
                isinstance(entry, (tuple, list))
                and len(entry) == 5
                and entry[0] == "gr1"
            ):
                idx = self._decode_sparse_indices_gap_rice(entry)

                # Defensive lossless/order checks.
                if idx.numel() > 1 and bool((idx[1:] <= idx[:-1]).any()):
                    raise RuntimeError(
                        f"Gap-Rice decode produced non-increasing indices for {key}."
                    )

                decoded_tensors[key] = {
                    "idx": idx.tolist()
                }
                continue

            # Legacy representation for old checkpoints/tests.
            if isinstance(entry, dict) and "idx" in entry:
                decoded_tensors[key] = entry
                continue

            raise ValueError(
                f"Unsupported sparse tensor payload for layer {key}: "
                f"{type(entry).__name__}"
            )

        # Shallow-copy only the outer packet. Other sidecars are unchanged.
        decoded_packet = dict(packet)
        decoded_packet["tensors"] = decoded_tensors

        return decoded_packet

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
    # Server aggregation
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
            raise ValueError("FedBop12 requires 0 < gamma < 1")

        # One server EMA update represents K client-event contributions.
        #
        # IMPORTANT: calibrate the pseudo-gradient scale ONCE using the initial
        # gamma. FedBop12 recomputed this denominator using the current gamma,
        # which exactly cancelled gamma decay in the fresh-evidence term. Here we
        # preserve the original round-1 strict-majority calibration, but allow
        # gamma decay to anneal fresh event evidence while increasing memory.
        beta = 1.0 - gamma
        beta_k = beta ** k_clients
        beta_ref = 1.0 - float(self.initial_gamma)
        beta_ref_k = beta_ref ** k_clients
        decision_boundary = float(majority) - 0.5
        event_scale = (
            self.tau * float(k_clients)
            / ((1.0 - beta_ref_k) * decision_boundary)
        )
        event_gain = (1.0 - beta_k) / (1.0 - beta_ref_k)

        total_binary = 0
        total_union = 0
        total_votes = 0
        total_flips = 0
        total_instant = 0

        # Extended diagnostic accumulators.
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

            w = self._sign_binary(global_tensor.detach().cpu()).view(-1)
            n = w.numel()
            vote_count = torch.zeros(n, dtype=torch.int16, device="cpu")

            for client_label, packet in clients_models:
                entry = packet.get("tensors", {}).get(key)
                if entry is None:
                    continue
                idx_raw = entry.get("idx", [])
                idx = torch.as_tensor(idx_raw, dtype=torch.long, device="cpu").view(-1)

                if self.diag and self.diag_extended:
                    client_id = packet.get("client_id", client_label)
                    state_key = (str(client_id), key)
                    prev = self._diag_prev_client_events.get(state_key)
                    if prev is not None and int(prev[0]) == int(round_number) - 1:
                        prev_idx = prev[1]
                        inter = self._diag_sparse_intersection(idx, prev_idx)
                        same_client_repeat_intersection += inter
                        same_client_repeat_union += int(idx.numel()) + int(prev_idx.numel()) - inter
                        same_client_current_events += int(idx.numel())
                    self._diag_prev_client_events[state_key] = (
                        int(round_number), idx.clone()
                    )

                if idx.numel() == 0:
                    continue
                vote_count[idx] += 1

            server_m = self.server_m_buffers.get(key)
            if server_m is None or server_m.shape != w.shape:
                server_m = torch.zeros_like(w, dtype=torch.float32)
            else:
                server_m = server_m.to(dtype=torch.float32, device="cpu")

            vote_fraction = vote_count.to(torch.float32) / float(k_clients)
            pseudo_grad = w.to(torch.float32) * vote_fraction * float(event_scale)

            # Effective aligned fresh-evidence increment per support count c is:
            #   tau * (c / decision_boundary) * event_gain
            # with event_gain=1 in phase 1 and naturally decreasing after gamma
            # decay. This restores Bop-like annealing at the server.
            server_m.mul_(beta_k).add_(pseudo_grad, alpha=(1.0 - beta_k))
            self.server_m_buffers[key] = server_m

            # Server Bop condition, again w*M > tau.
            flip_mask_cpu = w.to(torch.float32) * server_m > self.tau
            instant_mask = vote_count >= majority

            if self.diag and self.diag_extended:
                # Current-round support distribution.
                support_hist_all += torch.bincount(
                    vote_count.to(torch.long), minlength=k_clients + 1
                )[: k_clients + 1]
                if bool(flip_mask_cpu.any()):
                    support_hist_flips += torch.bincount(
                        vote_count[flip_mask_cpu].to(torch.long),
                        minlength=k_clients + 1,
                    )[: k_clients + 1]

                # Exact micro-Jaccard across all client pairs, without building K
                # dense masks: each coordinate with support c contributes C(c,2)
                # pairwise intersections and C(K,2)-C(K-c,2) pairwise unions.
                c64 = vote_count.to(torch.int64)
                pair_intersections += int(((c64 * (c64 - 1)) // 2).sum().item())
                kmc = int(k_clients) - c64
                total_pairs = int(k_clients * (k_clients - 1) // 2)
                pair_unions += int(
                    (total_pairs - ((kmc * (kmc - 1)) // 2)).sum().item()
                )

                # Consecutive server-flip overlap. Re-flipping the same coordinate
                # on consecutive rounds is literal binary backtracking/churn.
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

            if flip_mask_cpu.any():
                mask = flip_mask_cpu.to(device=global_tensor.device, dtype=torch.bool)
                global_tensor.view(-1)[mask] *= -1
            global_tensor.copy_(self._sign_binary(global_tensor))

            if self.diag:
                total_binary += n
                total_union += int((vote_count > 0).sum().item())
                total_votes += int(vote_count.sum().item())
                total_flips += int(flip_mask_cpu.sum().item())
                total_instant += int(instant_mask.sum().item())

        # Aggregate the real-valued global model state. BN beta uses the
        # quantized absolute-value path; decoding is back to FP32 before averaging.
        # Other non-binary tensors retain the generic compatibility path.
        beta_decoded_values = 0
        for key, global_tensor in global_model.items():
            if not torch.is_tensor(global_tensor):
                continue
            if self._is_binary_param_tensor(key, global_tensor):
                continue

            if self._is_bn_beta_key(key):
                received_beta = []
                for _, packet in clients_models:
                    entry = packet.get("beta_model", {}).get(key)
                    if entry is not None:
                        received_beta.append(
                            self._decode_wire_tensor(
                                entry,
                                device=global_tensor.device,
                                dtype=torch.float32,
                            )
                        )
                if received_beta:
                    acc = torch.zeros_like(
                        global_tensor, dtype=torch.float32, device=global_tensor.device
                    )
                    for beta_value in received_beta:
                        acc.add_(beta_value)
                    updated = acc / float(len(received_beta))
                    global_tensor.copy_(updated.to(dtype=global_tensor.dtype))
                    beta_decoded_values += int(global_tensor.numel()) * len(received_beta)
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

        # Build client-EMA consensus on its own cadence. If this round is also a
        # gamma-decay boundary, pack_server_model() will decay gamma AFTER this
        # round's training/aggregation and broadcast the shared state with the new
        # gamma. On sync-only rounds, gamma is left unchanged.
        if self._is_sync_round(round_number):
            self._prepare_momentum_consensus(clients_models, global_model, round_number)

        if self.diag:
            self._log(
                f"SERVER_BOP_ANNEAL round={round_number} K={k_clients} gamma={self.gamma:.3e} initial_gamma={self.initial_gamma:.3e} sync_every={self.sync_rounds} decay_every={self.decay_rounds} "
                f"tau={self.tau:.3e} event_scale_fixed={event_scale:.3e} event_gain={event_gain:.6e} "
                f"effective_per_vote={(self.tau/decision_boundary)*event_gain:.6e} "
                f"union_rate={total_union/max(1,total_binary):.6e} "
                f"mean_support={total_votes/max(1,total_union):.3f} "
                f"instant_majority_rate={total_instant/max(1,total_binary):.6e} "
                f"global_flip_rate={total_flips/max(1,total_binary):.6e}"
            )

            if self.diag_extended:
                flip_hist_total = int(support_hist_flips.sum().item())
                weighted_flip_support = int(
                    (support_hist_flips * torch.arange(k_clients + 1, dtype=torch.int64)).sum().item()
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
                    f"c{i}={int(support_hist_all[i].item())}" for i in range(k_clients + 1)
                )
                hist_flip_text = ",".join(
                    f"c{i}={int(support_hist_flips[i].item())}" for i in range(k_clients + 1)
                )
                self._log(
                    f"EVENT_ID_DIAG round={round_number} "
                    f"cross_client_micro_jaccard={micro_pair_jaccard:.6e} "
                    f"same_client_repeat_jaccard={same_client_repeat_jaccard:.6e} "
                    f"same_client_repeat_rate={same_client_repeat_rate:.6e}"
                )
                self._log(
                    f"SERVER_FLIP_DIAG round={round_number} "
                    f"mean_current_support_on_flips={weighted_flip_support/max(1,flip_hist_total):.6f} "
                    f"low_support_flip_rate={low_support_flips/max(1,flip_hist_total):.6e} "
                    f"zero_support_flip_rate={zero_support_flips/max(1,flip_hist_total):.6e} "
                    f"consecutive_reflip_rate={server_flip_repeat_rate:.6e} "
                    f"consecutive_reflip_jaccard={server_flip_repeat_jaccard:.6e}"
                )
                self._log(
                    f"EVENT_SUPPORT_HIST round={round_number} {hist_all_text}"
                )
                self._log(
                    f"FLIP_SUPPORT_HIST round={round_number} {hist_flip_text}"
                )

        return global_model

    # -------------------------------------------------------------------------
    # Server -> client model/state update
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def pack_server_model(self, raw_model):
        bin_model = {}
        stats_model = {}
        beta_model = {}
        beta_payload_bytes = 0
        beta_numel = 0

        # Binary downlink diagnostics.  These counters describe only the
        # authoritative global binary state; beta/stat/momentum sidecars keep
        # their existing codecs and accounting unchanged.
        binary_numel = 0
        binary_flip_events = 0
        binary_rice_valid_bits = 0
        binary_rice_data_bytes = 0
        binary_snapshot_data_bytes = 0
        binary_selected_pickle_bytes = 0
        binary_rice_layers = 0
        binary_snapshot_layers = 0
        binary_snapshot_fallback_layers = 0
        binary_layer_diag = []

        for key, tensor in raw_model.items():
            if not torch.is_tensor(tensor):
                continue
            if self._is_binary_param_tensor(key, tensor):
                current_sign = self._sign_binary(tensor.detach().cpu()).contiguous()
                previous_sign = self._server_downlink_prev_binary.get(key)

                entry, cdiag = self._encode_server_binary_wire(
                    current_sign=current_sign,
                    previous_sign=previous_sign,
                )
                bin_model[key] = entry

                # Advance the server's delta base only after the exact packet
                # entry has been constructed.  All clients receive this same
                # authoritative broadcast in the current FedWork protocol.
                self._server_downlink_prev_binary[key] = current_sign.clone()

                binary_numel += int(current_sign.numel())
                binary_flip_events += int(cdiag["events"])
                binary_rice_valid_bits += int(cdiag["rice_valid_bits"])
                binary_rice_data_bytes += int(cdiag["rice_data_bytes"])
                binary_snapshot_data_bytes += int(cdiag["snapshot_data_bytes"])
                binary_selected_pickle_bytes += int(cdiag["selected_pickle_bytes"])
                if cdiag["mode"] == "rice_delta":
                    binary_rice_layers += 1
                elif cdiag["mode"] == "snapshot_fallback":
                    binary_snapshot_fallback_layers += 1
                else:
                    binary_snapshot_layers += 1
                binary_layer_diag.append((key, int(current_sign.numel()), cdiag))
                continue

            if self._is_bn_beta_key(key):
                entry, nbytes = self._encode_wire_tensor(
                    tensor.detach(), self.beta_comm_mode
                )
                beta_model[key] = entry
                beta_payload_bytes += int(nbytes)
                beta_numel += int(tensor.numel())
                continue

            if int(self.omit_legacy_bn_buffers) and self._is_legacy_bn_buffer_key(key):
                continue

            stats_model[key] = tensor.detach().cpu()

        try:
            round_number = int(self.server.round_number)
        except Exception:
            round_number = -1

        if self.diag:
            full_1bit_bytes = (int(binary_numel) + 7) // 8
            self._log(
                f"GAP_RICE_DOWNLINK round={round_number} "
                f"binary={binary_numel} flips={binary_flip_events} "
                f"flip_rate={binary_flip_events/max(1,binary_numel):.6e} "
                f"rice_layers={binary_rice_layers} "
                f"snapshot_layers={binary_snapshot_layers} "
                f"snapshot_fallback_layers={binary_snapshot_fallback_layers} "
                f"rice_valid_bits={binary_rice_valid_bits} "
                f"rice_data_bytes={binary_rice_data_bytes} "
                f"selected_entry_pickle_KiB={binary_selected_pickle_bytes/1024.0:.3f} "
                f"dense_1bit_reference_KiB={full_1bit_bytes/1024.0:.3f}"
            )
            if self._layer_log_enabled(round_number):
                for key, n, d in binary_layer_diag:
                    self._log(
                        f"GAP_RICE_DOWNLINK_LAYER round={round_number} "
                        f"layer={key} numel={n} mode={d['mode']} "
                        f"events={d['events']} k={d['rice_k']} "
                        f"valid_bits={d['rice_valid_bits']} "
                        f"rice_data_bytes={d['rice_data_bytes']} "
                        f"snapshot_data_bytes={d['snapshot_data_bytes']} "
                        f"selected_pickle_bytes={d['selected_pickle_bytes']}"
                    )

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
            "beta_model": beta_model,
            "stats_model": stats_model,
            "gamma": self.gamma,
            "next_round": next_round,
            "momentum_sync": bool(momentum_sync),
            "momentum_sync_round": int(round_number) if momentum_sync else -1,
        }
        if momentum_sync:
            payload["shared_m_ternary"] = self.pending_shared_m_ternary

        if self.diag:
            self._log(
                f"BETA_DOWNLINK round={round_number} mode={self.beta_comm_mode} "
                f"numel={beta_numel} logical_payload_KiB={beta_payload_bytes/1024.0:.3f} "
                f"representation=absolute decoded_fp32=1"
            )
        return payload

    @torch.no_grad()
    def unpack_server_model(self, packed_model, current_model):
        bin_model = packed_model["bin_model"]
        beta_model = packed_model.get("beta_model", {})
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

        for key, server_entry in bin_model.items():
            if key not in current_model:
                continue

            client_tensor = current_model[key]

            # -------------------------------------------------------------
            # Reconstruct the authoritative global binary tensor.
            # -------------------------------------------------------------
            # New packets are either:
            #   bp1 : absolute one-bit snapshot
            #   gr1 : Gap+Rice indices of flips relative to the PREVIOUS
            #         authoritative global state stored in round_global_weights.
            # A legacy dense tensor path is intentionally retained so old saved
            # packets/tests do not break.
            if isinstance(server_entry, (tuple, list)) and len(server_entry) > 0:
                codec = server_entry[0]
            else:
                codec = None

            if codec == "bp1":
                flat = self._decode_binary_snapshot_1bit(
                    server_entry,
                    device=client_tensor.device,
                    dtype=client_tensor.dtype,
                )
                if int(flat.numel()) != int(client_tensor.numel()):
                    raise ValueError(
                        f"Downlink binary snapshot size mismatch for {key}: "
                        f"{flat.numel()} != {client_tensor.numel()}"
                    )
                w_new = flat.view_as(client_tensor)

            elif codec == "gr1":
                base = self.round_global_weights.get(key)
                if base is None:
                    raise RuntimeError(
                        f"Cannot apply Rice downlink delta for {key}: client has no "
                        f"previous authoritative global binary state. A 1-bit "
                        f"snapshot is required before deltas."
                    )
                if int(base.numel()) != int(client_tensor.numel()):
                    raise RuntimeError(
                        f"Rice downlink base-size mismatch for {key}: "
                        f"{base.numel()} != {client_tensor.numel()}"
                    )

                flip_indices = self._decode_sparse_indices_gap_rice(server_entry)
                if flip_indices.numel() > 0:
                    if int(flip_indices[-1].item()) >= int(client_tensor.numel()):
                        raise ValueError(
                            f"Rice downlink index out of range for {key}: "
                            f"max={int(flip_indices[-1].item())}, "
                            f"numel={client_tensor.numel()}"
                        )

                w_new = self._sign_binary(
                    base.to(device=client_tensor.device, dtype=client_tensor.dtype)
                ).view_as(client_tensor).clone()
                if flip_indices.numel() > 0:
                    idx_dev = flip_indices.to(device=client_tensor.device)
                    w_new.view(-1)[idx_dev] *= -1

            elif torch.is_tensor(server_entry):
                # Backward compatibility with the old uncompressed downlink.
                w_new = self._sign_binary(
                    server_entry.to(
                        device=client_tensor.device, dtype=client_tensor.dtype
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported FedBop binary downlink entry for {key}: "
                    f"{type(server_entry).__name__}, codec={codec!r}"
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
                        f"Missing shared momentum for layer {key} on momentum sync round "
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

        # Decode the aggregated global beta back to the client's FP32 model.
        # Adam m/v remain private and persistent; only the model parameter beta
        # is synchronized.
        for key, entry in beta_model.items():
            if key not in current_model:
                continue
            target = current_model[key]
            decoded = self._decode_wire_tensor(
                entry, device=target.device, dtype=target.dtype
            ).view_as(target)
            target.copy_(decoded)

        # Generic compatibility path for any remaining small non-binary state.
        for key, server_tensor in stats_model.items():
            if key not in current_model:
                continue

            lname = key.lower()
            if (
                "running_mean" in lname
                or "running_var" in lname
                or "num_batches_tracked" in lname
            ):
                continue

            target = current_model[key]
            target.copy_(
                server_tensor.to(device=target.device, dtype=target.dtype)
            )

        if self.diag and momentum_sync:
            self._log(
                f"M_SYNC_DOWNLINK round={momentum_sync_round} gamma_received={self.gamma:.3e} "
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
