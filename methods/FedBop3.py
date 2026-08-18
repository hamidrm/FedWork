import math
import torch
from core.FederatedLearningClass import *
from utils.logger import *


class FedBop3(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.contributors_percent = (
            float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        )
        self.gamma = self.get_arg(float, "gamma", 1e-4)
        self.tau = self.get_arg(float, "tau", 1e-8)
        self.decay_factor = self.get_arg(float, "decay_factor", 1e-1)
        self.decay_rounds = self.get_arg(int, "decay_rounds", 600)
        self.rho = self.get_arg(float, "rho", 1.0)

        self.m_buffers = None
        self.e_buffers = None
        self.round_global_weights = None
        self.has_local_update = False

        self.stats_lr = self.get_arg(float, "stats_lr", 1e-2)
        self.sync_stats = bool(self.get_arg(int, "sync_stats", 1))
        self.reset_stats_optimizer_on_sync = bool(
            self.get_arg(int, "reset_stats_optimizer_on_sync", 1)
        )
        self.stats_optimizer = None
        self.stats_param_ids = None

        # Diagnostics. 1 = enabled. Per-layer logs are emitted every N local rounds.
        self.diag = self.get_arg(int, "diag", 1)
        self.diag_layer_every = self.get_arg(int, "diag_layer_every", 10)

        self._diag_client_round = 0
        self._diag_steps = 0
        self._diag_flip_events = {}
        self._diag_grad_abs_sum = {}
        self._diag_m_abs_sum = {}
        self._diag_layer_steps = {}

        self._diag_server_pack_round = None
        self._diag_server_pack_calls = 0

        # Cross-round diagnostics only. They do not affect optimization.
        self._diag_prev_endpoint_masks = {}
        self._diag_prev_server_flip_masks = {}

        # Per-round temporal Bop-request voting state. For each binary coordinate,
        # after each minibatch momentum update we ask whether the current momentum
        # supports the *opposite state relative to the round-start global weight*:
        #     q_s = 1[w_start * m_s > tau].
        # f_local=1 iff q_s is true for a strict majority of local minibatches.
        # Only the resulting positive indices are transmitted to the server.
        self.temporal_request_counts = {}
        self.temporal_request_steps = {}
        self.round_local_vote_masks = {}
        self.diag_pairwise_max_clients = self.get_arg(
            int, "diag_pairwise_max_clients", 12
        )

        self.server_m_buffers = {}

    def get_name(self):
        return "FedBop3"

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

    def _diag_log(self, message):
        if not self.diag:
            return

        text = f"[FedBopTemporalBopRequestMajority] {message}"

        try:
            logger.log_info(text)
            return
        except Exception:
            pass

        try:
            logger.log_debug(text)
            return
        except Exception:
            pass

        print(text, flush=True)

    def _diag_layer_log_enabled(self, round_number):
        if not self.diag:
            return False

        if round_number <= 3:
            return True

        if self.diag_layer_every > 0 and round_number % self.diag_layer_every == 0:
            return True

        if self.decay_rounds > 0:
            d = round_number % self.decay_rounds
            if d in (self.decay_rounds - 1, 0, 1):
                return True

        return False

    def _diag_reset_local_round(self):
        self._diag_steps = 0
        self._diag_flip_events = {}
        self._diag_grad_abs_sum = {}
        self._diag_m_abs_sum = {}
        self._diag_layer_steps = {}

    def _diag_pairwise_jaccard_from_pattern_hist(
        self,
        pattern_hist,
        client_counts,
        k_clients,
    ):
        if (
            pattern_hist is None
            or k_clients < 2
            or len(client_counts) != k_clients
        ):
            return None

        codes = torch.arange(
            pattern_hist.numel(),
            dtype=torch.long,
        )
        values = []

        for i in range(k_clients):
            for j in range(i + 1, k_clients):
                both = (
                    ((codes >> i) & 1).bool()
                    & ((codes >> j) & 1).bool()
                )
                inter = int(pattern_hist[both].sum().item())
                union = (
                    int(client_counts[i])
                    + int(client_counts[j])
                    - inter
                )
                if union > 0:
                    values.append(inter / union)

        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        values = sorted(values)
        n = len(values)
        if n % 2:
            median = values[n // 2]
        else:
            median = 0.5 * (values[n // 2 - 1] + values[n // 2])

        return {
            "mean": sum(values) / n,
            "median": median,
            "min": values[0],
            "max": values[-1],
        }

    @torch.no_grad()
    def _diag_accumulate_local_step(self, name, grad, m, flip_mask):
        if not self.diag:
            return

        device = m.device

        if name not in self._diag_flip_events:
            self._diag_flip_events[name] = torch.zeros((), device=device, dtype=torch.float64)
            self._diag_grad_abs_sum[name] = torch.zeros((), device=device, dtype=torch.float64)
            self._diag_m_abs_sum[name] = torch.zeros((), device=device, dtype=torch.float64)
            self._diag_layer_steps[name] = 0

        self._diag_flip_events[name].add_(flip_mask.sum().to(torch.float64))
        self._diag_grad_abs_sum[name].add_(grad.detach().abs().mean().to(torch.float64))
        self._diag_m_abs_sum[name].add_(m.detach().abs().mean().to(torch.float64))
        self._diag_layer_steps[name] += 1

    @torch.no_grad()
    def _sign_binary(self, tensor):
        return torch.where(
            tensor >= 0,
            torch.ones_like(tensor),
            -torch.ones_like(tensor),
        )

    def _prepare_stats_optimizer(self, model):
        stats_params = [
            param
            for name, param in model.named_parameters()
            if param.requires_grad
            and not self._is_binary_param_tensor(
                name,
                param.data,
            )
        ]

        param_ids = tuple(id(param) for param in stats_params)

        if (
            self.stats_optimizer is None
            or self.stats_param_ids != param_ids
        ):
            self.stats_optimizer = torch.optim.Adam(
                stats_params,
                lr=self.stats_lr,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0.0,
                amsgrad=False,
            )
            self.stats_param_ids = param_ids

            self._diag_log(
                f"STATS_OPT created Adam for {len(stats_params)} non-binary trainable params "
                f"lr={self.stats_lr:.3e} eps=1e-7 "
                f"reset_on_sync={int(self.reset_stats_optimizer_on_sync)}"
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

        skip_tokens = [
            "bn",
            "batchnorm",
            "norm",
            "gn",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ]

        return not any(token in lname for token in skip_tokens)

    @torch.no_grad()
    def _create_client_state(self, model):
        self.m_buffers = {}
        self.e_buffers = {}
        self.round_global_weights = {}
        self.temporal_request_counts = {}
        self.temporal_request_steps = {}
        self.round_local_vote_masks = {}

        for name, param in model.named_parameters():
            if self._is_binary_param_tensor(name, param.data):
                self.m_buffers[name] = torch.zeros_like(param.data)

    @torch.no_grad()
    def _prepare_m_buffer(self, key, reference):
        if self.m_buffers is None:
            self.m_buffers = {}

        if key not in self.m_buffers:
            self.m_buffers[key] = torch.zeros_like(reference)
            return self.m_buffers[key]

        m = self.m_buffers[key]

        if m.shape != reference.shape:
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

            param.data.copy_(
                self._sign_binary(param.data)
            )

            m = self._prepare_m_buffer(
                name,
                param.data,
            )

            grad = param.grad.detach()

            m.mul_(1.0 - self.gamma).add_(
                grad,
                alpha=self.gamma,
            )

            flip_mask = (
                (m.abs() > self.tau)
                & (
                    torch.sign(m)
                    == self._sign_binary(param.data)
                )
            )

            # Temporal Bop-request vote.  IMPORTANT: this is evaluated from the
            # freshly updated momentum and the ROUND-START global sign, before the
            # current minibatch's local weight flip is applied.  For binary w_start:
            #
            #     q_s = 1[w_start * m_s > tau]
            #
            # is exactly the Bop threshold/sign condition for requesting the state
            # opposite to w_start.  We count q_s across local minibatches and later
            # transmit the coordinate iff q_s holds for a strict majority.
            round_ref = (
                self.round_global_weights.get(name)
                if self.round_global_weights is not None
                else None
            )
            if round_ref is not None:
                ref = self._sign_binary(
                    round_ref.to(device=param.device, dtype=param.dtype)
                )
                count = self.temporal_request_counts.get(name)
                if (
                    count is None
                    or count.shape != param.shape
                    or count.device != param.device
                ):
                    count = torch.zeros_like(param.data, dtype=torch.int16)

                request_opposite = (ref * m) > self.tau
                count.add_(request_opposite.to(torch.int16))
                self.temporal_request_counts[name] = count
                self.temporal_request_steps[name] = (
                    self.temporal_request_steps.get(name, 0) + 1
                )

            self._diag_accumulate_local_step(
                name,
                grad,
                m,
                flip_mask,
            )

            param.data[flip_mask] = -param.data[flip_mask]

            param.data.copy_(
                self._sign_binary(param.data)
            )

            param.grad = None

        if self.stats_optimizer is not None:
            self.stats_optimizer.step()
            self.stats_optimizer.zero_grad(set_to_none=True)

        return True

    @torch.no_grad()
    def pack_client_model(self, raw_model, global_model, id):
        tensors = {}
        stats_model = {}

        if self.e_buffers is None:
            self.e_buffers = {}
        if self.round_global_weights is None:
            self.round_global_weights = {}

        self._diag_client_round += 1
        round_number = self._diag_client_round

        total_binary = 0
        total_endpoint = 0
        total_votes = 0
        total_vote_endpoint_inter = 0
        total_vote_endpoint_union = 0
        total_vote_final_trigger = 0
        total_old_trigger = 0
        total_near_tie = 0
        total_raw_flip_events = 0
        total_m_abs = 0.0
        total_grad_abs = 0.0
        total_vote_fraction_sum = 0.0
        total_vote_fraction_n = 0
        total_request_fraction_sum = 0.0
        total_request_fraction_n = 0
        fallback_layers = 0
        layer_diag = {}

        for key, tensor in raw_model.items():
            if not torch.is_tensor(tensor):
                continue

            if not self._is_binary_param_tensor(key, tensor):
                stats_model[key] = tensor.detach().cpu()
                continue

            local_weight = self._sign_binary(tensor.detach().cpu())
            global_weight = self._sign_binary(global_model[key].detach().cpu())
            endpoint_flip = local_weight.view(-1) != global_weight.view(-1)

            if self.m_buffers is not None and key in self.m_buffers:
                m = self.m_buffers[key].detach().cpu()
            else:
                m = torch.zeros_like(global_weight)

            aligned_m = global_weight * m
            final_trigger = aligned_m.view(-1) > self.tau

            steps = int(self.temporal_request_steps.get(key, 0))
            counts = self.temporal_request_counts.get(key)
            fallback = 0
            if counts is None or counts.numel() != tensor.numel() or steps <= 0:
                # Should only occur if a framework hook bypasses after_backward.
                # Fall back to the final round-start-aligned Bop request, not the
                # endpoint parity, and surface the fallback explicitly in logs.
                vote_mask = final_trigger.clone()
                vote_fraction = final_trigger.to(torch.float32)
                fallback = 1
                fallback_layers += 1
            else:
                counts_cpu = counts.detach().cpu().view(-1).to(torch.int32)
                threshold = (steps // 2) + 1
                vote_mask = counts_cpu >= threshold
                vote_fraction = counts_cpu.to(torch.float32) / float(steps)

            self.round_local_vote_masks[key] = vote_mask.clone()
            flip_indices = torch.nonzero(vote_mask, as_tuple=False).view(-1)
            tensors[key] = {"idx": flip_indices.tolist()}

            # Keep the same residual/reflection reconciliation as the preceding
            # temporal-occupancy branch.  The controlled change here is ONLY the
            # definition of f_local: strict majority of q_s = 1[w_start*m_s > tau].
            self.e_buffers[key] = self.tau - aligned_m
            self.round_global_weights[key] = global_weight.clone()

            endpoint_n = int(endpoint_flip.sum().item())
            vote_n = int(vote_mask.sum().item())
            inter_n = int((vote_mask & endpoint_flip).sum().item())
            union_n = int((vote_mask | endpoint_flip).sum().item())
            vote_trigger_n = int((vote_mask & final_trigger).sum().item())
            old_trigger_n = int(final_trigger.sum().item())

            if steps > 0 and not fallback:
                near_tie = int(((vote_fraction >= 0.40) & (vote_fraction <= 0.60)).sum().item())
                mean_vote_fraction = float(vote_fraction[vote_mask].mean().item()) if vote_n > 0 else 0.0
            else:
                near_tie = 0
                mean_vote_fraction = 1.0 if vote_n > 0 else 0.0

            raw_events = int(self._diag_flip_events.get(key, torch.tensor(0.0)).item())
            layer_steps = max(1, self._diag_layer_steps.get(key, 0))
            mean_g = float(self._diag_grad_abs_sum.get(key, torch.tensor(0.0)).item()) / layer_steps
            mean_m = float(m.abs().mean().item())

            total_binary += tensor.numel()
            total_endpoint += endpoint_n
            total_votes += vote_n
            total_vote_endpoint_inter += inter_n
            total_vote_endpoint_union += union_n
            total_vote_final_trigger += vote_trigger_n
            total_old_trigger += old_trigger_n
            total_near_tie += near_tie
            total_raw_flip_events += raw_events
            total_m_abs += mean_m * tensor.numel()
            total_grad_abs += mean_g * tensor.numel()
            if vote_n > 0:
                total_vote_fraction_sum += mean_vote_fraction * vote_n
                total_vote_fraction_n += vote_n
            if steps > 0 and not fallback:
                # Mean fraction of local minibatches that requested the opposite
                # round-start state, averaged over all binary coordinates.
                total_request_fraction_sum += float(vote_fraction.sum().item())
                total_request_fraction_n += int(vote_fraction.numel())

            if self.diag:
                layer_diag[key] = {
                    "n": tensor.numel(),
                    "steps": steps,
                    "endpoint": endpoint_n,
                    "votes": vote_n,
                    "inter": inter_n,
                    "union": union_n,
                    "vote_trigger": vote_trigger_n,
                    "old_trigger": old_trigger_n,
                    "near_tie": near_tie,
                    "mean_vote_fraction": mean_vote_fraction,
                    "raw": raw_events,
                    "mean_m": mean_m,
                    "mean_g": mean_g,
                    "fallback": fallback,
                }

        if self.diag:
            vote_rate = total_votes / max(1, total_binary)
            endpoint_rate = total_endpoint / max(1, total_binary)
            vote_endpoint_jaccard = total_vote_endpoint_inter / max(1, total_vote_endpoint_union)
            vote_endpoint_precision = total_vote_endpoint_inter / max(1, total_votes)
            vote_endpoint_recall = total_vote_endpoint_inter / max(1, total_endpoint)
            vote_final_trigger_fraction = total_vote_final_trigger / max(1, total_votes)
            near_tie_rate = total_near_tie / max(1, total_binary)
            mean_vote_fraction = total_vote_fraction_sum / max(1, total_vote_fraction_n)
            mean_request_fraction_all = (
                total_request_fraction_sum / max(1, total_request_fraction_n)
            )
            mean_abs_m = total_m_abs / max(1, total_binary)
            mean_abs_g = total_grad_abs / max(1, total_binary)

            self._diag_log(
                f"CLIENT_BOP_REQUEST_MAJ round={round_number} id={id} gamma={self.gamma:.3e} rho={self.rho:.3e} "
                f"local_steps={self._diag_steps} rule=strict_bop_request_majority_vs_round_start "
                f"endpoint={total_endpoint} endpoint_rate={endpoint_rate:.6e} "
                f"f_local_votes={total_votes} vote_rate={vote_rate:.6e} "
                f"vote_endpoint_jaccard={vote_endpoint_jaccard:.6f} "
                f"vote_endpoint_precision={vote_endpoint_precision:.6f} vote_endpoint_recall={vote_endpoint_recall:.6f} "
                f"vote_final_trigger={total_vote_final_trigger} vote_final_trigger_fraction={vote_final_trigger_fraction:.6f} "
                f"near_tie_40_60={total_near_tie} near_tie_rate={near_tie_rate:.6e} "
                f"mean_request_fraction_on_votes={mean_vote_fraction:.6f} "
                f"mean_request_fraction_all={mean_request_fraction_all:.6f} "
                f"old_global_trigger={total_old_trigger} raw_flip_events={total_raw_flip_events} "
                f"mean|m|={mean_abs_m:.3e} mean|g|={mean_abs_g:.3e} fallback_layers={fallback_layers}"
            )

            if self._diag_layer_log_enabled(round_number):
                for key, d in layer_diag.items():
                    j = d["inter"] / max(1, d["union"])
                    trig_frac = d["vote_trigger"] / max(1, d["votes"])
                    self._diag_log(
                        f"CLIENT_BOP_REQUEST_MAJ_LAYER round={round_number} id={id} layer={key} n={d['n']} "
                        f"steps={d['steps']} endpoint={d['endpoint']} f_local={d['votes']} "
                        f"endpoint_jaccard={j:.4f} vote_final_trigger_fraction={trig_frac:.4f} "
                        f"near_tie_40_60={d['near_tie']} mean_request_fraction_on_votes={d['mean_vote_fraction']:.4f} "
                        f"raw={d['raw']} mean|m|={d['mean_m']:.3e} mean|g|={d['mean_g']:.3e} "
                        f"fallback={d['fallback']}"
                    )

        self.has_local_update = True
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
                "temporal_bop_request_majority_votes": total_votes,
                "total_binary": total_binary,
            }

        self._diag_reset_local_round()
        return packet

    def unpack_client_model(self, packed_model):
        return packed_model

    @torch.no_grad()
    def aggregate(self, clients_models, global_model):
        if len(clients_models) == 0:
            return global_model

        k_clients = len(clients_models)
        majority = (k_clients // 2) + 1

        gamma = float(self.gamma)
        if not (0.0 < gamma < 1.0):
            raise ValueError("FedBop server momentum requires 0 < gamma < 1")

        beta = 1.0 - gamma
        beta_k = beta ** k_clients
        decision_boundary = float(majority) - 0.5
        event_scale = (
            self.tau
            * float(k_clients)
            / ((1.0 - beta_k) * decision_boundary)
        )

        total_binary = 0
        total_union = 0
        total_vote_requests = 0
        total_global_flips = 0
        total_instant_majority_flips = 0
        total_rescued_by_momentum = 0
        total_blocked_by_momentum = 0
        total_no_vote_flips = 0
        total_support_on_flips = 0
        total_server_m_abs = 0.0
        total_server_m_abs_max = 0.0
        total_aligned_before_sum = 0.0
        total_aligned_after_sum = 0.0

        total_server_prev_flips = 0
        total_server_repeat_intersection = 0
        server_repeat_prev_available = 0

        vote_hist_total = torch.zeros(k_clients + 1, dtype=torch.long)

        pairwise_enabled = (
            self.diag
            and k_clients >= 2
            and k_clients <= self.diag_pairwise_max_clients
        )
        pattern_hist_total = (
            torch.zeros(1 << k_clients, dtype=torch.long)
            if pairwise_enabled
            else None
        )

        client_endpoint_counts = []
        client_steps = []
        for _, client_model in clients_models:
            diag = client_model.get("diag", {})
            if diag:
                endpoint_count = int(
                    diag.get(
                        "transmitted_votes",
                        diag.get("endpoint_flips", 0),
                    )
                )
                step_count = int(diag.get("steps", 0))
            else:
                endpoint_count = sum(
                    len(v.get("idx", []))
                    for v in client_model.get("tensors", {}).values()
                )
                step_count = 0

            client_endpoint_counts.append(endpoint_count)
            client_steps.append(step_count)

        layer_vote_diag = {}

        for key, global_tensor in global_model.items():
            if not torch.is_tensor(global_tensor):
                continue

            if not self._is_binary_param_tensor(key, global_tensor):
                continue

            numel = global_tensor.numel()
            global_weight_cpu = self._sign_binary(
                global_tensor.detach().cpu()
            ).view(-1)

            vote_count = torch.zeros(
                numel,
                dtype=torch.int16,
                device="cpu",
            )

            vote_pattern = (
                torch.zeros(numel, dtype=torch.int16, device="cpu")
                if pairwise_enabled
                else None
            )
            layer_client_counts = [0] * k_clients

            for client_pos, (_, client_model) in enumerate(clients_models):
                client_tensors = client_model.get("tensors", {})
                if key not in client_tensors:
                    continue

                indices = client_tensors[key].get("idx", [])
                if len(indices) == 0:
                    continue

                indices = torch.as_tensor(
                    indices,
                    dtype=torch.long,
                    device="cpu",
                )
                vote_count[indices] += 1
                layer_client_counts[client_pos] = len(indices)

                if vote_pattern is not None:
                    bit_value = 1 << client_pos
                    vote_pattern[indices] = torch.bitwise_or(
                        vote_pattern[indices],
                        torch.tensor(bit_value, dtype=torch.int16),
                    )

            server_m = self.server_m_buffers.get(key)
            if (
                server_m is None
                or server_m.shape != global_weight_cpu.shape
            ):
                server_m = torch.zeros_like(
                    global_weight_cpu,
                    dtype=torch.float32,
                )
            else:
                server_m = server_m.to(dtype=torch.float32, device="cpu")

            aligned_before = global_weight_cpu.to(torch.float32) * server_m

            vote_fraction = (
                vote_count.to(torch.float32) / float(k_clients)
            )
            mean_pseudo_grad = (
                global_weight_cpu.to(torch.float32)
                * vote_fraction
                * float(event_scale)
            )

            server_m.mul_(beta_k).add_(
                mean_pseudo_grad,
                alpha=(1.0 - beta_k),
            )
            self.server_m_buffers[key] = server_m

            aligned_after = global_weight_cpu.to(torch.float32) * server_m
            flip_mask_cpu = aligned_after > self.tau
            instant_flip_mask = vote_count >= majority
            current_union = vote_count > 0

            rescued_by_momentum = flip_mask_cpu & (~instant_flip_mask)
            blocked_by_momentum = instant_flip_mask & (~flip_mask_cpu)
            no_vote_flips = flip_mask_cpu & (~current_union)

            global_flips = int(flip_mask_cpu.sum().item())
            instant_majority_flips = int(instant_flip_mask.sum().item())
            union_n = int(current_union.sum().item())
            vote_requests = int(vote_count.sum().item())
            rescued_n = int(rescued_by_momentum.sum().item())
            blocked_n = int(blocked_by_momentum.sum().item())
            no_vote_n = int(no_vote_flips.sum().item())
            support_on_flips = (
                int(vote_count[flip_mask_cpu].sum().item())
                if global_flips > 0
                else 0
            )

            hist = torch.bincount(
                vote_count.to(torch.long),
                minlength=k_clients + 1,
            )
            vote_hist_total += hist

            layer_pairwise = None
            if vote_pattern is not None:
                pattern_hist = torch.bincount(
                    vote_pattern.to(torch.long),
                    minlength=1 << k_clients,
                )
                pattern_hist_total += pattern_hist
                layer_pairwise = self._diag_pairwise_jaccard_from_pattern_hist(
                    pattern_hist,
                    layer_client_counts,
                    k_clients,
                )

            current_server_flip = flip_mask_cpu.clone()
            prev_server_flip = self._diag_prev_server_flip_masks.get(key)
            server_repeat_intersection = 0
            prev_server_flips = 0
            if (
                prev_server_flip is not None
                and prev_server_flip.shape == current_server_flip.shape
            ):
                server_repeat_prev_available = 1
                prev_server_flips = int(prev_server_flip.sum().item())
                server_repeat_intersection = int(
                    (current_server_flip & prev_server_flip).sum().item()
                )
                total_server_prev_flips += prev_server_flips
                total_server_repeat_intersection += server_repeat_intersection

            self._diag_prev_server_flip_masks[key] = current_server_flip

            total_binary += numel
            total_union += union_n
            total_vote_requests += vote_requests
            total_global_flips += global_flips
            total_instant_majority_flips += instant_majority_flips
            total_rescued_by_momentum += rescued_n
            total_blocked_by_momentum += blocked_n
            total_no_vote_flips += no_vote_n
            total_support_on_flips += support_on_flips
            total_server_m_abs += float(server_m.abs().sum().item())
            total_server_m_abs_max = max(
                total_server_m_abs_max,
                float(server_m.abs().max().item()),
            )
            total_aligned_before_sum += float(aligned_before.sum().item())
            total_aligned_after_sum += float(aligned_after.sum().item())

            layer_vote_diag[key] = {
                "numel": numel,
                "union": union_n,
                "vote_requests": vote_requests,
                "instant_majority_flips": instant_majority_flips,
                "global_flips": global_flips,
                "rescued_by_momentum": rescued_n,
                "blocked_by_momentum": blocked_n,
                "no_vote_flips": no_vote_n,
                "support_on_flips": support_on_flips,
                "hist": hist.tolist(),
                "pairwise": layer_pairwise,
                "prev_server_flips": prev_server_flips,
                "server_repeat_intersection": server_repeat_intersection,
                "mean_abs_server_m": float(server_m.abs().mean().item()),
                "max_abs_server_m": float(server_m.abs().max().item()),
                "mean_aligned_before": float(aligned_before.mean().item()),
                "mean_aligned_after": float(aligned_after.mean().item()),
            }

            if flip_mask_cpu.any():
                flip_mask = flip_mask_cpu.to(
                    device=global_tensor.device,
                    dtype=torch.bool,
                )
                global_tensor.view(-1)[flip_mask] *= -1

            global_tensor.copy_(self._sign_binary(global_tensor))

        stats_dispersion = {
            "bias": [],
            "running_mean": [],
            "running_var": [],
        }

        for key, global_tensor in global_model.items():
            if not torch.is_tensor(global_tensor):
                continue

            if self._is_binary_param_tensor(key, global_tensor):
                continue

            received_tensors = []
            for _, client_model in clients_models:
                stats_model = client_model.get("stats_model", {})
                if key in stats_model:
                    received_tensors.append(stats_model[key])

            if len(received_tensors) == 0:
                continue

            if torch.is_floating_point(global_tensor):
                if self.diag and len(received_tensors) > 1:
                    stack = torch.stack([
                        tensor.to(dtype=torch.float32, device="cpu")
                        for tensor in received_tensors
                    ])
                    mean = stack.mean(dim=0)
                    mad = (stack - mean).abs().mean().item()

                    lname = key.lower()
                    if "running_mean" in lname:
                        stats_dispersion["running_mean"].append(mad)
                    elif "running_var" in lname:
                        stats_dispersion["running_var"].append(mad)
                    elif "bias" in lname:
                        stats_dispersion["bias"].append(mad)

                tensor_sum = torch.zeros_like(
                    global_tensor,
                    dtype=torch.float32,
                    device=global_tensor.device,
                )
                for tensor in received_tensors:
                    tensor_sum.add_(
                        tensor.to(
                            device=global_tensor.device,
                            dtype=torch.float32,
                        )
                    )

                global_tensor.copy_(
                    (tensor_sum / len(received_tensors)).to(
                        dtype=global_tensor.dtype
                    )
                )
            else:
                max_tensor = received_tensors[0].to(
                    device=global_tensor.device,
                    dtype=global_tensor.dtype,
                )
                for tensor in received_tensors[1:]:
                    max_tensor = torch.maximum(
                        max_tensor,
                        tensor.to(
                            device=global_tensor.device,
                            dtype=global_tensor.dtype,
                        ),
                    )
                global_tensor.copy_(max_tensor)

        if self.diag:
            try:
                round_number = int(self.server.round_number)
            except Exception:
                round_number = -1

            union_rate = total_union / max(1, total_binary)
            global_flip_rate = total_global_flips / max(1, total_binary)
            instant_majority_rate = (
                total_instant_majority_flips / max(1, total_binary)
            )
            mean_support = total_vote_requests / max(1, total_union)
            mean_support_flipped = (
                total_support_on_flips / max(1, total_global_flips)
            )
            mean_abs_server_m = total_server_m_abs / max(1, total_binary)
            mean_aligned_before = total_aligned_before_sum / max(1, total_binary)
            mean_aligned_after = total_aligned_after_sum / max(1, total_binary)

            pairwise = (
                self._diag_pairwise_jaccard_from_pattern_hist(
                    pattern_hist_total,
                    client_endpoint_counts,
                    k_clients,
                )
                if pairwise_enabled
                else None
            )

            server_repeat_current_rate = (
                total_server_repeat_intersection / total_global_flips
                if server_repeat_prev_available and total_global_flips > 0
                else 0.0
            )
            server_repeat_union = (
                total_global_flips
                + total_server_prev_flips
                - total_server_repeat_intersection
            )
            server_repeat_jaccard = (
                total_server_repeat_intersection / server_repeat_union
                if server_repeat_prev_available and server_repeat_union > 0
                else 0.0
            )
            server_new_flips = max(
                0, total_global_flips - total_server_repeat_intersection
            )

            vote_parts = [
                f"v{i}={int(vote_hist_total[i].item())}"
                for i in range(1, k_clients + 1)
            ]

            endpoint_avg = (
                sum(client_endpoint_counts) / len(client_endpoint_counts)
                if client_endpoint_counts
                else 0.0
            )
            endpoint_min = min(client_endpoint_counts) if client_endpoint_counts else 0
            endpoint_max = max(client_endpoint_counts) if client_endpoint_counts else 0
            steps_avg = (
                sum(client_steps) / len(client_steps)
                if client_steps
                else 0.0
            )

            self._diag_log(
                f"SERVER_BOP round={round_number} K={k_clients} majority={majority} "
                f"evidence=local_temporal_bop_request_majority_positive_only boundary={decision_boundary:.1f} "
                f"gamma={self.gamma:.3e} tau={self.tau:.3e} betaK={beta_k:.6f} "
                f"event_scale={event_scale:.3e} "
                f"client_steps_avg={steps_avg:.1f} client_vote_avg={endpoint_avg:.1f} "
                f"client_vote_min={endpoint_min} client_vote_max={endpoint_max} "
                f"union={total_union} union_rate={union_rate:.6e} "
                f"mean_support={mean_support:.3f} "
                f"instant_majority_flips={total_instant_majority_flips} "
                f"instant_majority_rate={instant_majority_rate:.6e} "
                f"global_flips={total_global_flips} global_flip_rate={global_flip_rate:.6e} "
                f"rescued_by_momentum={total_rescued_by_momentum} "
                f"blocked_by_momentum={total_blocked_by_momentum} "
                f"no_vote_flips={total_no_vote_flips} "
                f"mean_support_flipped={mean_support_flipped:.3f} "
                f"mean|M_server|={mean_abs_server_m:.3e} "
                f"max|M_server|={total_server_m_abs_max:.3e} "
                f"mean_aligned_before={mean_aligned_before:.3e} "
                f"mean_aligned_after={mean_aligned_after:.3e} "
                f"server_repeat={total_server_repeat_intersection} "
                f"server_repeat_rate={server_repeat_current_rate:.6f} "
                f"server_repeat_jaccard={server_repeat_jaccard:.6f} "
                f"server_new_flips={server_new_flips} "
                + (
                    f"pair_jaccard_mean={pairwise['mean']:.6f} "
                    f"pair_jaccard_median={pairwise['median']:.6f} "
                    f"pair_jaccard_min={pairwise['min']:.6f} "
                    f"pair_jaccard_max={pairwise['max']:.6f} "
                    if pairwise is not None
                    else "pair_jaccard=disabled "
                )
                + " ".join(vote_parts)
            )

            if self._diag_layer_log_enabled(max(1, round_number)):
                for key, d in layer_vote_diag.items():
                    hist = d["hist"]
                    hist_text = ",".join(
                        f"{i}:{hist[i]}"
                        for i in range(1, len(hist))
                    )
                    pair_text = ""
                    if d.get("pairwise") is not None:
                        p = d["pairwise"]
                        pair_text = (
                            f" pair_jaccard_mean={p['mean']:.4f}"
                            f" pair_jaccard_median={p['median']:.4f}"
                            f" pair_jaccard_min={p['min']:.4f}"
                            f" pair_jaccard_max={p['max']:.4f}"
                        )

                    mean_support_layer = (
                        d["vote_requests"] / max(1, d["union"])
                    )
                    mean_support_flipped_layer = (
                        d["support_on_flips"] / max(1, d["global_flips"])
                    )

                    self._diag_log(
                        f"SERVER_BOP_LAYER round={round_number} layer={key} n={d['numel']} "
                        f"union={d['union']} mean_support={mean_support_layer:.3f} "
                        f"instant_majority_flips={d['instant_majority_flips']} "
                        f"global_flips={d['global_flips']} "
                        f"rescued_by_momentum={d['rescued_by_momentum']} "
                        f"blocked_by_momentum={d['blocked_by_momentum']} "
                        f"no_vote_flips={d['no_vote_flips']} "
                        f"mean_support_flipped={mean_support_flipped_layer:.3f} "
                        f"mean|M|={d['mean_abs_server_m']:.3e} "
                        f"max|M|={d['max_abs_server_m']:.3e} "
                        f"aligned_before={d['mean_aligned_before']:.3e} "
                        f"aligned_after={d['mean_aligned_after']:.3e} "
                        f"prev_server_flips={d['prev_server_flips']} "
                        f"server_repeat={d['server_repeat_intersection']} "
                        f"votes=[{hist_text}]" + pair_text
                    )

            bias_mad = (
                sum(stats_dispersion["bias"]) / len(stats_dispersion["bias"])
                if stats_dispersion["bias"]
                else 0.0
            )
            mean_mad = (
                sum(stats_dispersion["running_mean"])
                / len(stats_dispersion["running_mean"])
                if stats_dispersion["running_mean"]
                else 0.0
            )
            var_mad = (
                sum(stats_dispersion["running_var"])
                / len(stats_dispersion["running_var"])
                if stats_dispersion["running_var"]
                else 0.0
            )
            self._diag_log(
                f"STATS round={round_number} client_MAD beta={bias_mad:.3e} "
                f"running_mean={mean_mad:.3e} running_var={var_mad:.3e}"
            )

        return global_model

    @torch.no_grad()
    def pack_server_model(self, raw_model):
        bin_model = {}
        stats_model = {}

        for key, tensor in raw_model.items():
            if not torch.is_tensor(tensor):
                continue

            if self._is_binary_param_tensor(key, tensor):
                bin_model[key] = self._sign_binary(
                    tensor.detach().cpu()
                )

            else:
                stats_model[key] = tensor.detach().cpu()

        try:
            round_number = int(self.server.round_number)
        except Exception:
            round_number = -1

        if self.diag:
            if self._diag_server_pack_round != round_number:
                self._diag_server_pack_round = round_number
                self._diag_server_pack_calls = 0
            self._diag_server_pack_calls += 1

        old_gamma = self.gamma
        old_rho = self.rho
        did_decay = False

        # Intentionally preserves the current behavior so diagnostics can reveal
        # whether this function is called more than once on a decay round.
        if (
            round_number > 0
            and self.decay_rounds > 0
            and round_number % self.decay_rounds == 0
        ):
            self.gamma = self.decay_factor * self.gamma
            self.rho = self.decay_factor * self.rho
            did_decay = True

        if self.diag:
            self._diag_log(
                f"SERVER_PACK round={round_number} call={self._diag_server_pack_calls} "
                f"decay={int(did_decay)} gamma={old_gamma:.3e}->{self.gamma:.3e} "
                f"rho={old_rho:.3e}->{self.rho:.3e}"
            )

        return {
            "bin_model": bin_model,
            "stats_model": stats_model,
            "gamma": self.gamma,
            "rho": self.rho,
        }

    @torch.no_grad()
    def unpack_server_model(
        self,
        packed_model,
        current_model,
    ):
        bin_model = packed_model["bin_model"]
        stats_model = packed_model["stats_model"]

        received_gamma = packed_model["gamma"]
        received_rho = packed_model["rho"]

        self.gamma = received_gamma
        # Current algorithm keeps local rho unchanged. Uncomment to test rho decay:
        # self.rho = received_rho

        if self.m_buffers is None:
            self.m_buffers = {}

        if self.e_buffers is None:
            self.e_buffers = {}

        if self.round_global_weights is None:
            self.round_global_weights = {}

        total_binary = 0
        total_local_flip = 0
        total_server_flip = 0
        total_disagreement = 0
        total_local_rejected = 0
        total_server_imposed = 0
        total_pre_trigger_new = 0
        total_post_trigger_new = 0
        total_post_trigger_on_disagreement = 0
        total_e_abs = 0.0
        total_corr_delta_abs = 0.0

        total_rejected_a_lt_tau = 0
        total_rejected_a_neg = 0
        total_rejected_a_0_tau = 0
        total_rejected_a_tau_2tau = 0
        total_rejected_a_ge_2tau = 0
        total_pre_trigger_rejected = 0
        total_pre_trigger_imposed = 0
        total_post_trigger_rejected = 0
        total_post_trigger_imposed = 0
        total_reflection_identity_mismatch = 0
        total_e_fresh_abs_sum = 0.0
        max_e_fresh_abs = 0.0

        sync_layers = {}

        for key, global_tensor in bin_model.items():
            if key not in current_model:
                continue

            client_tensor = current_model[key]

            new_global_weight = self._sign_binary(
                global_tensor.to(
                    device=client_tensor.device,
                    dtype=client_tensor.dtype,
                )
            )

            m = self._prepare_m_buffer(
                key,
                client_tensor,
            )

            layer_diag = None

            if (
                self.has_local_update
                and key in self.round_global_weights
                and key in self.e_buffers
            ):
                old_global_weight = self._sign_binary(
                    self.round_global_weights[key].to(
                        device=client_tensor.device,
                        dtype=client_tensor.dtype,
                    )
                )

                local_weight = self._sign_binary(
                    client_tensor
                )

                e = self.e_buffers[key].to(
                    device=client_tensor.device,
                    dtype=client_tensor.dtype,
                )

                aligned_m = old_global_weight * m

                if e.shape != aligned_m.shape:
                    e = self.tau - aligned_m

                endpoint_flip = (
                    local_weight
                    != old_global_weight
                )

                stored_vote = self.round_local_vote_masks.get(key)
                if stored_vote is not None and stored_vote.numel() == endpoint_flip.numel():
                    local_flip = stored_vote.to(
                        device=client_tensor.device,
                        dtype=torch.bool,
                    ).view_as(endpoint_flip)
                else:
                    local_flip = endpoint_flip

                server_flip = (
                    new_global_weight
                    != old_global_weight
                )

                disagreement = (
                    local_flip
                    != server_flip
                )

                local_rejected = local_flip & (~server_flip)
                server_imposed = (~local_flip) & server_flip

                pre_trigger_new = (
                    new_global_weight * m > self.tau
                )

                rejected_a_lt_tau = (
                    local_rejected & (aligned_m < self.tau)
                )
                rejected_a_neg = (
                    local_rejected & (aligned_m < 0)
                )
                rejected_a_0_tau = (
                    local_rejected
                    & (aligned_m >= 0)
                    & (aligned_m < self.tau)
                )
                rejected_a_tau_2tau = (
                    local_rejected
                    & (aligned_m >= self.tau)
                    & (aligned_m < (2.0 * self.tau))
                )
                rejected_a_ge_2tau = (
                    local_rejected
                    & (aligned_m >= (2.0 * self.tau))
                )

                pre_trigger_rejected = pre_trigger_new & local_rejected
                pre_trigger_imposed = pre_trigger_new & server_imposed

                e_expected = self.tau - aligned_m
                e_fresh_abs = (e - e_expected).abs()

                corrected_aligned_m = (
                    aligned_m
                    + ((1.0 + self.rho) * e)
                )

                corrected_m = (
                    old_global_weight
                    * corrected_aligned_m
                )

                if self.diag:
                    n = client_tensor.numel()
                    local_flip_n = int(local_flip.sum().item())
                    server_flip_n = int(server_flip.sum().item())
                    disagreement_n = int(disagreement.sum().item())
                    local_rejected_n = int(local_rejected.sum().item())
                    server_imposed_n = int(server_imposed.sum().item())
                    pre_trigger_n = int(pre_trigger_new.sum().item())
                    rejected_a_lt_tau_n = int(rejected_a_lt_tau.sum().item())
                    rejected_a_neg_n = int(rejected_a_neg.sum().item())
                    rejected_a_0_tau_n = int(rejected_a_0_tau.sum().item())
                    rejected_a_tau_2tau_n = int(rejected_a_tau_2tau.sum().item())
                    rejected_a_ge_2tau_n = int(rejected_a_ge_2tau.sum().item())
                    pre_trigger_rejected_n = int(pre_trigger_rejected.sum().item())
                    pre_trigger_imposed_n = int(pre_trigger_imposed.sum().item())
                    e_fresh_abs_sum = float(e_fresh_abs.sum().item())
                    e_fresh_abs_max = float(e_fresh_abs.max().item())

                    if disagreement_n > 0:
                        e_abs_sum = float(e[disagreement].abs().sum().item())
                        corr_delta_abs_sum = float(
                            (corrected_m[disagreement] - m[disagreement]).abs().sum().item()
                        )
                    else:
                        e_abs_sum = 0.0
                        corr_delta_abs_sum = 0.0

                m.copy_(
                    torch.where(
                        disagreement,
                        corrected_m,
                        m,
                    )
                )

                if self.diag:
                    post_trigger_new = (
                        new_global_weight * m > self.tau
                    )
                    post_trigger_n = int(post_trigger_new.sum().item())
                    post_trigger_dis_n = int(
                        (post_trigger_new & disagreement).sum().item()
                    )
                    post_trigger_rejected = (
                        post_trigger_new & local_rejected
                    )
                    post_trigger_imposed = (
                        post_trigger_new & server_imposed
                    )
                    post_trigger_rejected_n = int(
                        post_trigger_rejected.sum().item()
                    )
                    post_trigger_imposed_n = int(
                        post_trigger_imposed.sum().item()
                    )
                    reflection_identity_mismatch_n = int(
                        (
                            post_trigger_rejected
                            != rejected_a_lt_tau
                        ).sum().item()
                    )

                    layer_diag = {
                        "numel": n,
                        "local_flip": local_flip_n,
                        "server_flip": server_flip_n,
                        "disagreement": disagreement_n,
                        "local_rejected": local_rejected_n,
                        "server_imposed": server_imposed_n,
                        "pre_trigger_new": pre_trigger_n,
                        "post_trigger_new": post_trigger_n,
                        "post_trigger_dis": post_trigger_dis_n,
                        "rejected_a_lt_tau": rejected_a_lt_tau_n,
                        "rejected_a_neg": rejected_a_neg_n,
                        "rejected_a_0_tau": rejected_a_0_tau_n,
                        "rejected_a_tau_2tau": rejected_a_tau_2tau_n,
                        "rejected_a_ge_2tau": rejected_a_ge_2tau_n,
                        "pre_trigger_rejected": pre_trigger_rejected_n,
                        "pre_trigger_imposed": pre_trigger_imposed_n,
                        "post_trigger_rejected": post_trigger_rejected_n,
                        "post_trigger_imposed": post_trigger_imposed_n,
                        "reflection_identity_mismatch": reflection_identity_mismatch_n,
                        "e_fresh_abs_sum": e_fresh_abs_sum,
                        "e_fresh_abs_max": e_fresh_abs_max,
                        "e_abs_sum": e_abs_sum,
                        "corr_delta_abs_sum": corr_delta_abs_sum,
                    }

                    total_binary += n
                    total_local_flip += local_flip_n
                    total_server_flip += server_flip_n
                    total_disagreement += disagreement_n
                    total_local_rejected += local_rejected_n
                    total_server_imposed += server_imposed_n
                    total_pre_trigger_new += pre_trigger_n
                    total_post_trigger_new += post_trigger_n
                    total_post_trigger_on_disagreement += post_trigger_dis_n
                    total_e_abs += e_abs_sum
                    total_corr_delta_abs += corr_delta_abs_sum
                    total_rejected_a_lt_tau += rejected_a_lt_tau_n
                    total_rejected_a_neg += rejected_a_neg_n
                    total_rejected_a_0_tau += rejected_a_0_tau_n
                    total_rejected_a_tau_2tau += rejected_a_tau_2tau_n
                    total_rejected_a_ge_2tau += rejected_a_ge_2tau_n
                    total_pre_trigger_rejected += pre_trigger_rejected_n
                    total_pre_trigger_imposed += pre_trigger_imposed_n
                    total_post_trigger_rejected += post_trigger_rejected_n
                    total_post_trigger_imposed += post_trigger_imposed_n
                    total_reflection_identity_mismatch += reflection_identity_mismatch_n
                    total_e_fresh_abs_sum += e_fresh_abs_sum
                    max_e_fresh_abs = max(max_e_fresh_abs, e_fresh_abs_max)

            client_tensor.copy_(
                new_global_weight
            )

            self.round_global_weights[key] = (
                new_global_weight
                .detach()
                .cpu()
            )
            self.temporal_request_counts[key] = torch.zeros_like(
                client_tensor, dtype=torch.int16
            )
            self.temporal_request_steps[key] = 0

            if layer_diag is not None:
                sync_layers[key] = layer_diag

        if self.diag and self.has_local_update:
            round_number = self._diag_client_round
            disagreement_rate = total_disagreement / max(1, total_binary)
            local_rejected_rate = total_local_rejected / max(1, total_binary)
            server_imposed_rate = total_server_imposed / max(1, total_binary)
            post_dis_trigger_rate = (
                total_post_trigger_on_disagreement / max(1, total_disagreement)
            )
            mean_e_abs = total_e_abs / max(1, total_disagreement)
            mean_corr_delta = total_corr_delta_abs / max(1, total_disagreement)

            rejected_a_lt_tau_rate = (
                total_rejected_a_lt_tau / max(1, total_local_rejected)
            )
            pre_trigger_rejected_rate = (
                total_pre_trigger_rejected / max(1, total_local_rejected)
            )
            post_trigger_rejected_rate = (
                total_post_trigger_rejected / max(1, total_local_rejected)
            )
            post_trigger_imposed_rate = (
                total_post_trigger_imposed / max(1, total_server_imposed)
            )
            mean_e_fresh_abs = (
                total_e_fresh_abs_sum / max(1, total_binary)
            )

            self._diag_log(
                f"SYNC round={round_number} gamma={self.gamma:.3e} "
                f"rho_local={self.rho:.3e} rho_received={received_rho:.3e} "
                f"local_flip={total_local_flip} server_flip={total_server_flip} "
                f"disagreement={total_disagreement} disagreement_rate={disagreement_rate:.6e} "
                f"local_rejected={total_local_rejected} rejected_rate={local_rejected_rate:.6e} "
                f"server_imposed={total_server_imposed} imposed_rate={server_imposed_rate:.6e} "
                f"pre_new_trigger={total_pre_trigger_new} post_new_trigger={total_post_trigger_new} "
                f"post_trigger_on_disagreement={total_post_trigger_on_disagreement} "
                f"post_dis_trigger_rate={post_dis_trigger_rate:.6f} "
                f"rejected_a_lt_tau={total_rejected_a_lt_tau} rejected_a_lt_tau_rate={rejected_a_lt_tau_rate:.6f} "
                f"rej_a_neg={total_rejected_a_neg} rej_a_0_tau={total_rejected_a_0_tau} "
                f"rej_a_tau_2tau={total_rejected_a_tau_2tau} rej_a_ge_2tau={total_rejected_a_ge_2tau} "
                f"pre_trigger_rejected={total_pre_trigger_rejected} pre_trigger_rejected_rate={pre_trigger_rejected_rate:.6f} "
                f"post_trigger_rejected={total_post_trigger_rejected} post_trigger_rejected_rate={post_trigger_rejected_rate:.6f} "
                f"post_trigger_imposed={total_post_trigger_imposed} post_trigger_imposed_rate={post_trigger_imposed_rate:.6f} "
                f"reflection_identity_mismatch={total_reflection_identity_mismatch} "
                f"e_fresh_mean_abs={mean_e_fresh_abs:.3e} e_fresh_max_abs={max_e_fresh_abs:.3e} "
                f"mean|e|_dis={mean_e_abs:.3e} mean|delta_m|_dis={mean_corr_delta:.3e}"
            )

            if self._diag_layer_log_enabled(round_number):
                for key, d in sync_layers.items():
                    self._diag_log(
                        f"SYNC_LAYER round={round_number} layer={key} n={d['numel']} "
                        f"local={d['local_flip']} server={d['server_flip']} dis={d['disagreement']} "
                        f"rejected={d['local_rejected']} imposed={d['server_imposed']} "
                        f"rej_a_lt_tau={d['rejected_a_lt_tau']} "
                        f"rej_a_neg={d['rejected_a_neg']} rej_a_0_tau={d['rejected_a_0_tau']} "
                        f"rej_a_tau_2tau={d['rejected_a_tau_2tau']} rej_a_ge_2tau={d['rejected_a_ge_2tau']} "
                        f"pre_trigger={d['pre_trigger_new']} post_trigger={d['post_trigger_new']} "
                        f"post_trigger_dis={d['post_trigger_dis']} "
                        f"post_trigger_rejected={d['post_trigger_rejected']} "
                        f"post_trigger_imposed={d['post_trigger_imposed']} "
                        f"reflection_mismatch={d['reflection_identity_mismatch']} "
                        f"e_fresh_max={d['e_fresh_abs_max']:.3e}"
                    )

        stats_sync_tensors = 0
        stats_sync_float_numel = 0
        stats_sync_abs_sum = 0.0
        stats_sync_abs_max = 0.0

        if self.sync_stats:
            for key, tensor in stats_model.items():
                if key not in current_model:
                    continue

                target = current_model[key]
                received = tensor.to(
                    device=target.device,
                    dtype=target.dtype,
                )

                if torch.is_floating_point(target):
                    delta = (target - received).abs()
                    stats_sync_float_numel += target.numel()
                    stats_sync_abs_sum += float(delta.sum().item())
                    if delta.numel() > 0:
                        stats_sync_abs_max = max(
                            stats_sync_abs_max,
                            float(delta.max().item()),
                        )

                target.copy_(received)
                stats_sync_tensors += 1

        adam_state_params_before = 0
        adam_step_max_before = 0
        adam_reset = 0
        adam_moment_numel = 0
        adam_exp_avg_abs_sum = 0.0
        adam_exp_avg_sq_sum = 0.0

        if self.stats_optimizer is not None:
            adam_state_params_before = len(self.stats_optimizer.state)
            for state in self.stats_optimizer.state.values():
                step = state.get("step", 0)
                if torch.is_tensor(step):
                    step = int(step.item())
                else:
                    step = int(step)
                adam_step_max_before = max(adam_step_max_before, step)

                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if torch.is_tensor(exp_avg):
                    adam_moment_numel += exp_avg.numel()
                    adam_exp_avg_abs_sum += float(
                        exp_avg.detach().abs().sum().item()
                    )
                if torch.is_tensor(exp_avg_sq):
                    adam_exp_avg_sq_sum += float(
                        exp_avg_sq.detach().sum().item()
                    )

            if self.sync_stats and self.reset_stats_optimizer_on_sync:
                self.stats_optimizer.state.clear()
                adam_reset = 1

        if self.diag:
            round_number = self._diag_client_round
            stats_sync_mean_abs = (
                stats_sync_abs_sum / max(1, stats_sync_float_numel)
            )
            adam_exp_avg_mean_abs = (
                adam_exp_avg_abs_sum / max(1, adam_moment_numel)
            )
            adam_exp_avg_sq_mean = (
                adam_exp_avg_sq_sum / max(1, adam_moment_numel)
            )
            self._diag_log(
                f"STATS_SYNC round={round_number} enabled={int(self.sync_stats)} "
                f"tensors={stats_sync_tensors} float_numel={stats_sync_float_numel} "
                f"mean_abs_overwrite={stats_sync_mean_abs:.3e} "
                f"max_abs_overwrite={stats_sync_abs_max:.3e} "
                f"reset_adam={adam_reset} "
                f"adam_state_params_before={adam_state_params_before} "
                f"adam_step_max_before={adam_step_max_before} "
                f"adam_exp_avg_mean_abs={adam_exp_avg_mean_abs:.3e} "
                f"adam_exp_avg_sq_mean={adam_exp_avg_sq_mean:.3e}"
            )

        self.has_local_update = False

        return current_model
