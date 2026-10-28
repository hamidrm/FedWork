import torch
from core.FederatedLearningClass import *
from utils.logger import *


class FedBop10(FederatedLearningClass):
    """
    Clean FedBop variant with one fixed policy for the whole training run.

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

    There is deliberately NO reflection, NO canonical +/-tau reset, NO momentum
    anchor/synchronization, and NO phase-dependent rule switching.

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
        self.tau = self.get_arg(float, "tau", 1e-8)
        self.decay_factor = self.get_arg(float, "decay_factor", 1e-1)
        self.decay_rounds = self.get_arg(int, "decay_rounds", 200)

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

        # Server-private Bop state.
        self.server_m_buffers = {}

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

    def get_name(self):
        return "FedBop10"

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
        text = f"[FedBop10CleanStateMajMinProj] {message}"
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
            raise ValueError("FedBop10 requires 0 < gamma < 1")

        # One server EMA update represents K client-event contributions. The scale
        # is inherited from the strongest FedBop7 server-Bop formulation so that,
        # from zero server momentum, the Bop threshold corresponds to the strict
        # client-majority decision boundary.
        beta = 1.0 - gamma
        beta_k = beta ** k_clients
        decision_boundary = float(majority) - 0.5
        event_scale = (
            self.tau * float(k_clients)
            / ((1.0 - beta_k) * decision_boundary)
        )

        total_binary = 0
        total_union = 0
        total_votes = 0
        total_flips = 0
        total_instant = 0

        for key, global_tensor in global_model.items():
            if not torch.is_tensor(global_tensor):
                continue
            if not self._is_binary_param_tensor(key, global_tensor):
                continue

            w = self._sign_binary(global_tensor.detach().cpu()).view(-1)
            n = w.numel()
            vote_count = torch.zeros(n, dtype=torch.int16, device="cpu")

            for _, packet in clients_models:
                entry = packet.get("tensors", {}).get(key)
                if entry is None:
                    continue
                idx = entry.get("idx", [])
                if len(idx) == 0:
                    continue
                idx = torch.as_tensor(idx, dtype=torch.long, device="cpu")
                vote_count[idx] += 1

            server_m = self.server_m_buffers.get(key)
            if server_m is None or server_m.shape != w.shape:
                server_m = torch.zeros_like(w, dtype=torch.float32)
            else:
                server_m = server_m.to(dtype=torch.float32, device="cpu")

            vote_fraction = vote_count.to(torch.float32) / float(k_clients)
            pseudo_grad = w.to(torch.float32) * vote_fraction * float(event_scale)

            server_m.mul_(beta_k).add_(pseudo_grad, alpha=(1.0 - beta_k))
            self.server_m_buffers[key] = server_m

            # Server Bop condition, again w*M > tau.
            flip_mask_cpu = w.to(torch.float32) * server_m > self.tau
            instant_mask = vote_count >= majority

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

        # Aggregate all non-binary state by simple client mean for floating tensors;
        # integer counters use max, matching the previous FedBop implementation.
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

        if self.diag:
            self._log(
                f"SERVER_BOP round={round_number} K={k_clients} gamma={self.gamma:.3e} "
                f"tau={self.tau:.3e} event_scale={event_scale:.3e} "
                f"union_rate={total_union/max(1,total_binary):.6e} "
                f"mean_support={total_votes/max(1,total_union):.3f} "
                f"instant_majority_rate={total_instant/max(1,total_binary):.6e} "
                f"global_flip_rate={total_flips/max(1,total_binary):.6e}"
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

        next_round = int(round_number) + 1 if round_number >= 0 else 1
        return {
            "bin_model": bin_model,
            "stats_model": stats_model,
            "gamma": self.gamma,
            "next_round": next_round,
        }

    @torch.no_grad()
    def unpack_server_model(self, packed_model, current_model):
        bin_model = packed_model["bin_model"]
        stats_model = packed_model["stats_model"]
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

        for key, server_tensor in bin_model.items():
            if key not in current_model:
                continue

            client_tensor = current_model[key]
            w_new = self._sign_binary(
                server_tensor.to(device=client_tensor.device, dtype=client_tensor.dtype)
            )
            m = self._prepare_m_buffer(key, client_tensor)

            if self.has_local_update and key in self.round_global_weights:
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

        if self.diag and self.has_local_update:
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
