import torch
from core.FederatedLearningClass import *
from utils.logger import *


class FedBop2(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.contributors_percent = (
            float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        )
        self.gamma = self.get_arg(float, "gamma", 1e-4)
        self.tau = self.get_arg(float, "tau", 1e-8)
        self.decay_factor = self.get_arg(float, "decay_factor", 1e-1)
        self.decay_rounds = self.get_arg(int, "decay_rounds", 600)

        self.stats_lr = self.get_arg(float, "stats_lr", 1e-2)
        self.sync_stats = bool(self.get_arg(int, "sync_stats", 1))
        self.reset_stats_optimizer_on_sync = bool(
            self.get_arg(int, "reset_stats_optimizer_on_sync", 1)
        )

        self.diag = self.get_arg(int, "diag", 1)
        self.diag_layer_every = self.get_arg(int, "diag_layer_every", 10)
        self.diag_pairwise_max_clients = self.get_arg(
            int, "diag_pairwise_max_clients", 12
        )

        self.m_buffers = None
        self.round_global_weights = None
        self.has_local_update = False

        self.stats_optimizer = None
        self.stats_param_ids = None

        self.server_m_buffers = {}
        self._last_decay_round = None

        self._diag_client_round = 0
        self._diag_steps = 0
        self._diag_flip_events = {}
        self._diag_grad_abs_sum = {}
        self._diag_m_abs_sum = {}
        self._diag_layer_steps = {}
        self._diag_prev_endpoint_masks = {}
        self._diag_prev_server_flip_masks = {}

    def get_name(self):
        return "FedBop2"

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

        text = f"[FedBopDenseMPrivateNoReflect] {message}"
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

    def _prepare_stats_optimizer(self, model):
        stats_params = [
            param
            for name, param in model.named_parameters()
            if param.requires_grad
            and not self._is_binary_param_tensor(name, param.data)
        ]
        param_ids = tuple(id(param) for param in stats_params)

        if self.stats_optimizer is None or self.stats_param_ids != param_ids:
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
                f"STATS_OPT created Adam params={len(stats_params)} "
                f"lr={self.stats_lr:.3e} reset_on_sync={int(self.reset_stats_optimizer_on_sync)}"
            )

    @torch.no_grad()
    def _create_client_state(self, model):
        self.m_buffers = {}
        self.round_global_weights = {}
        for name, param in model.named_parameters():
            if self._is_binary_param_tensor(name, param.data):
                self.m_buffers[name] = torch.zeros_like(param.data)

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
    def _diag_accumulate_local_step(self, name, grad, m, flip_mask):
        if not self.diag:
            return

        device = m.device
        if name not in self._diag_flip_events:
            self._diag_flip_events[name] = torch.zeros(
                (), device=device, dtype=torch.float64
            )
            self._diag_grad_abs_sum[name] = torch.zeros(
                (), device=device, dtype=torch.float64
            )
            self._diag_m_abs_sum[name] = torch.zeros(
                (), device=device, dtype=torch.float64
            )
            self._diag_layer_steps[name] = 0

        self._diag_flip_events[name].add_(flip_mask.sum().to(torch.float64))
        self._diag_grad_abs_sum[name].add_(
            grad.detach().abs().mean().to(torch.float64)
        )
        self._diag_m_abs_sum[name].add_(
            m.detach().abs().mean().to(torch.float64)
        )
        self._diag_layer_steps[name] += 1

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

            param.data.copy_(self._sign_binary(param.data))

            m = self._prepare_m_buffer(name, param.data)
            grad = param.grad.detach()

            m.mul_(1.0 - self.gamma).add_(grad, alpha=self.gamma)

            flip_mask = (
                (m.abs() > self.tau)
                & (torch.sign(m) == self._sign_binary(param.data))
            )

            self._diag_accumulate_local_step(
                name,
                grad,
                m,
                flip_mask,
            )

            param.data[flip_mask] = -param.data[flip_mask]
            param.data.copy_(self._sign_binary(param.data))
            param.grad = None

        if self.stats_optimizer is not None:
            self.stats_optimizer.step()
            self.stats_optimizer.zero_grad(set_to_none=True)

        return True

    @torch.no_grad()
    def pack_client_model(self, raw_model, global_model, id):
        endpoint_tensors = {}
        momentum_model = {}
        stats_model = {}

        if self.round_global_weights is None:
            self.round_global_weights = {}
        self._diag_client_round += 1
        round_number = self._diag_client_round

        total_binary = 0
        total_endpoint_flips = 0
        total_raw_flip_events = 0
        total_m_abs_sum = 0.0
        total_m_max = 0.0
        total_grad_abs_weighted = 0.0
        total_repeat = 0
        total_prev_endpoint = 0
        repeat_available = 0

        layer_diag = {}

        for key, tensor in raw_model.items():
            if not torch.is_tensor(tensor):
                continue

            if self._is_binary_param_tensor(key, tensor):
                local_weight = self._sign_binary(tensor.detach().cpu())
                global_weight = self._sign_binary(global_model[key].detach().cpu())

                local_flip = local_weight.view(-1) != global_weight.view(-1)
                flip_indices = torch.nonzero(
                    local_flip,
                    as_tuple=False,
                ).view(-1)

                endpoint_tensors[key] = {"idx": flip_indices.tolist()}

                if self.m_buffers is not None and key in self.m_buffers:
                    m = self.m_buffers[key].detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                else:
                    m = torch.zeros_like(global_weight, dtype=torch.float32)

                momentum_model[key] = m.clone()
                self.round_global_weights[key] = global_weight.clone()

                n = tensor.numel()
                endpoint_n = int(local_flip.sum().item())
                raw_n = int(
                    self._diag_flip_events.get(
                        key,
                        torch.tensor(0.0),
                    ).item()
                )
                m_abs = float(m.abs().mean().item())
                m_max = float(m.abs().max().item())

                layer_steps = max(1, self._diag_layer_steps.get(key, 0))
                grad_abs = float(
                    self._diag_grad_abs_sum.get(
                        key,
                        torch.tensor(0.0),
                    ).item()
                ) / layer_steps

                prev = self._diag_prev_endpoint_masks.get(key)
                repeat_n = 0
                prev_n = 0
                if prev is not None and prev.shape == local_flip.shape:
                    repeat_available = 1
                    repeat_n = int((prev & local_flip).sum().item())
                    prev_n = int(prev.sum().item())
                    total_repeat += repeat_n
                    total_prev_endpoint += prev_n
                self._diag_prev_endpoint_masks[key] = local_flip.clone()

                total_binary += n
                total_endpoint_flips += endpoint_n
                total_raw_flip_events += raw_n
                total_m_abs_sum += m_abs * n
                total_m_max = max(total_m_max, m_max)
                total_grad_abs_weighted += grad_abs * n

                layer_diag[key] = {
                    "n": n,
                    "endpoint": endpoint_n,
                    "raw": raw_n,
                    "mean_m": m_abs,
                    "max_m": m_max,
                    "mean_g": grad_abs,
                    "repeat": repeat_n,
                    "prev": prev_n,
                }
            else:
                stats_model[key] = tensor.detach().cpu()

        if self.diag:
            steps = self._diag_steps
            ema_new_mass = (
                1.0 - ((1.0 - self.gamma) ** steps)
                if steps > 0
                else 0.0
            )
            mean_m = total_m_abs_sum / max(1, total_binary)
            mean_g = total_grad_abs_weighted / max(1, total_binary)
            endpoint_rate = total_endpoint_flips / max(1, total_binary)
            repeat_rate = (
                total_repeat / total_endpoint_flips
                if repeat_available and total_endpoint_flips > 0
                else 0.0
            )
            repeat_jaccard = (
                total_repeat
                / max(1, total_endpoint_flips + total_prev_endpoint - total_repeat)
                if repeat_available
                else 0.0
            )
            dense_m_mib = (total_binary * 4.0) / (1024.0 ** 2)

            self._diag_log(
                f"CLIENT round={round_number} id={id} mode=dense_m_private_no_reflection "
                f"gamma={self.gamma:.3e} tau={self.tau:.3e} steps={steps} "
                f"ema_new_mass={ema_new_mass:.6f} binary={total_binary} "
                f"endpoint_flips={total_endpoint_flips} endpoint_rate={endpoint_rate:.6e} "
                f"raw_flip_events={total_raw_flip_events} repeat_rate={repeat_rate:.6f} "
                f"repeat_jaccard={repeat_jaccard:.6f} mean|m_local_end|={mean_m:.3e} "
                f"max|m_local_end|={total_m_max:.3e} mean|g|={mean_g:.3e} "
                f"dense_m_uplink_MiB={dense_m_mib:.2f}"
            )

            if self._diag_layer_log_enabled(round_number):
                for key, d in layer_diag.items():
                    self._diag_log(
                        f"CLIENT_LAYER round={round_number} id={id} layer={key} "
                        f"n={d['n']} endpoint={d['endpoint']} raw={d['raw']} "
                        f"repeat={d['repeat']} prev_endpoint={d['prev']} "
                        f"mean|m|={d['mean_m']:.3e} max|m|={d['max_m']:.3e} "
                        f"mean|g|={d['mean_g']:.3e}"
                    )

        self.has_local_update = True

        packet = {
            "client_id": id,
            "tensors": endpoint_tensors,
            "momentum_model": momentum_model,
            "stats_model": stats_model,
        }

        if self.diag:
            packet["diag"] = {
                "round": round_number,
                "steps": self._diag_steps,
                "endpoint_flips": total_endpoint_flips,
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

        total_binary = 0
        total_global_flips = 0
        total_endpoint_union = 0
        total_endpoint_votes = 0
        total_endpoint_support_on_flips = 0
        total_endpoint_majority = 0
        total_trigger_union = 0
        total_trigger_votes = 0
        total_trigger_support_on_flips = 0
        total_trigger_majority = 0
        total_shared_flip_no_endpoint = 0
        total_shared_flip_no_trigger = 0
        total_shared_m_abs = 0.0
        total_shared_m_max = 0.0
        total_client_m_mad = 0.0
        total_client_m_abs = 0.0

        layer_diag = {}

        for key, global_tensor in global_model.items():
            if not torch.is_tensor(global_tensor):
                continue
            if not self._is_binary_param_tensor(key, global_tensor):
                continue

            global_weight = self._sign_binary(
                global_tensor.detach().cpu()
            ).to(dtype=torch.float32)
            flat_w = global_weight.view(-1)
            numel = flat_w.numel()

            m_sum = torch.zeros_like(global_weight, dtype=torch.float32)
            local_ms = []

            for _, client_model in clients_models:
                momentum_model = client_model.get("momentum_model", {})
                if key not in momentum_model:
                    raise RuntimeError(
                        f"Shared-m oracle requires dense momentum for layer '{key}' "
                        f"from every participating client"
                    )

                m_k = momentum_model[key].to(
                    device="cpu",
                    dtype=torch.float32,
                )
                if m_k.shape != global_weight.shape:
                    raise RuntimeError(
                        f"Momentum shape mismatch for '{key}': "
                        f"got {tuple(m_k.shape)}, expected {tuple(global_weight.shape)}"
                    )
                m_sum.add_(m_k)
                local_ms.append(m_k)

            shared_m = m_sum / float(k_clients)
            self.server_m_buffers[key] = shared_m.clone()

            client_m_mad = 0.0
            client_m_abs = 0.0
            for m_k in local_ms:
                client_m_mad += float((m_k - shared_m).abs().mean().item())
                client_m_abs += float(m_k.abs().mean().item())
            client_m_mad /= float(k_clients)
            client_m_abs /= float(k_clients)

            endpoint_count = torch.zeros(numel, dtype=torch.int16)
            trigger_count = torch.zeros(numel, dtype=torch.int16)

            for client_pos, (_, client_model) in enumerate(clients_models):
                client_tensors = client_model.get("tensors", {})
                indices = []
                if key in client_tensors:
                    indices = client_tensors[key].get("idx", [])
                if len(indices) > 0:
                    indices = torch.as_tensor(indices, dtype=torch.long)
                    endpoint_count[indices] += 1

                m_k = local_ms[client_pos].view(-1)
                trigger_count.add_((flat_w * m_k > self.tau).to(torch.int16))

            aligned_shared = flat_w * shared_m.view(-1)
            flip_mask_cpu = aligned_shared > self.tau

            endpoint_union = endpoint_count > 0
            trigger_union = trigger_count > 0
            endpoint_majority_mask = endpoint_count >= majority
            trigger_majority_mask = trigger_count >= majority

            global_flips = int(flip_mask_cpu.sum().item())
            endpoint_union_n = int(endpoint_union.sum().item())
            trigger_union_n = int(trigger_union.sum().item())
            endpoint_votes_n = int(endpoint_count.sum().item())
            trigger_votes_n = int(trigger_count.sum().item())
            endpoint_majority_n = int(endpoint_majority_mask.sum().item())
            trigger_majority_n = int(trigger_majority_mask.sum().item())

            endpoint_support_flips = (
                int(endpoint_count[flip_mask_cpu].sum().item())
                if global_flips > 0
                else 0
            )
            trigger_support_flips = (
                int(trigger_count[flip_mask_cpu].sum().item())
                if global_flips > 0
                else 0
            )
            shared_flip_no_endpoint = int(
                (flip_mask_cpu & (~endpoint_union)).sum().item()
            )
            shared_flip_no_trigger = int(
                (flip_mask_cpu & (~trigger_union)).sum().item()
            )

            mean_shared_abs = float(shared_m.abs().mean().item())
            max_shared_abs = float(shared_m.abs().max().item())
            mean_aligned_shared = float(aligned_shared.mean().item())

            prev_server_flip = self._diag_prev_server_flip_masks.get(key)
            repeat_server = 0
            if prev_server_flip is not None and prev_server_flip.shape == flip_mask_cpu.shape:
                repeat_server = int((prev_server_flip & flip_mask_cpu).sum().item())
            self._diag_prev_server_flip_masks[key] = flip_mask_cpu.clone()

            if flip_mask_cpu.any():
                flip_mask = flip_mask_cpu.to(
                    device=global_tensor.device,
                    dtype=torch.bool,
                )
                global_tensor.view(-1)[flip_mask] *= -1
            global_tensor.copy_(self._sign_binary(global_tensor))

            total_binary += numel
            total_global_flips += global_flips
            total_endpoint_union += endpoint_union_n
            total_endpoint_votes += endpoint_votes_n
            total_endpoint_support_on_flips += endpoint_support_flips
            total_endpoint_majority += endpoint_majority_n
            total_trigger_union += trigger_union_n
            total_trigger_votes += trigger_votes_n
            total_trigger_support_on_flips += trigger_support_flips
            total_trigger_majority += trigger_majority_n
            total_shared_flip_no_endpoint += shared_flip_no_endpoint
            total_shared_flip_no_trigger += shared_flip_no_trigger
            total_shared_m_abs += mean_shared_abs * numel
            total_shared_m_max = max(total_shared_m_max, max_shared_abs)
            total_client_m_mad += client_m_mad * numel
            total_client_m_abs += client_m_abs * numel

            layer_diag[key] = {
                "n": numel,
                "global_flips": global_flips,
                "endpoint_union": endpoint_union_n,
                "endpoint_votes": endpoint_votes_n,
                "endpoint_majority": endpoint_majority_n,
                "endpoint_support_flips": endpoint_support_flips,
                "trigger_union": trigger_union_n,
                "trigger_votes": trigger_votes_n,
                "trigger_majority": trigger_majority_n,
                "trigger_support_flips": trigger_support_flips,
                "shared_flip_no_endpoint": shared_flip_no_endpoint,
                "shared_flip_no_trigger": shared_flip_no_trigger,
                "mean_shared_abs": mean_shared_abs,
                "max_shared_abs": max_shared_abs,
                "mean_aligned_shared": mean_aligned_shared,
                "client_m_mad": client_m_mad,
                "client_m_abs": client_m_abs,
                "repeat_server": repeat_server,
            }

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

            received = []
            for _, client_model in clients_models:
                stats_model = client_model.get("stats_model", {})
                if key in stats_model:
                    received.append(stats_model[key])

            if len(received) == 0:
                continue

            if torch.is_floating_point(global_tensor):
                if self.diag and len(received) > 1:
                    stack = torch.stack([
                        t.to(dtype=torch.float32, device="cpu")
                        for t in received
                    ])
                    mean = stack.mean(dim=0)
                    mad = float((stack - mean).abs().mean().item())
                    lname = key.lower()
                    if "running_mean" in lname:
                        stats_dispersion["running_mean"].append(mad)
                    elif "running_var" in lname:
                        stats_dispersion["running_var"].append(mad)
                    elif "bias" in lname:
                        stats_dispersion["bias"].append(mad)

                acc = torch.zeros_like(
                    global_tensor,
                    dtype=torch.float32,
                    device=global_tensor.device,
                )
                for t in received:
                    acc.add_(t.to(
                        device=global_tensor.device,
                        dtype=torch.float32,
                    ))
                global_tensor.copy_(
                    (acc / float(len(received))).to(dtype=global_tensor.dtype)
                )
            else:
                max_tensor = received[0].to(
                    device=global_tensor.device,
                    dtype=global_tensor.dtype,
                )
                for t in received[1:]:
                    max_tensor = torch.maximum(
                        max_tensor,
                        t.to(
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

            mean_shared_abs = total_shared_m_abs / max(1, total_binary)
            mean_client_m_mad = total_client_m_mad / max(1, total_binary)
            mean_client_m_abs = total_client_m_abs / max(1, total_binary)
            relative_m_mad = mean_client_m_mad / max(1e-30, mean_client_m_abs)
            endpoint_mean_support = total_endpoint_votes / max(1, total_endpoint_union)
            trigger_mean_support = total_trigger_votes / max(1, total_trigger_union)
            endpoint_support_flipped = (
                total_endpoint_support_on_flips / max(1, total_global_flips)
            )
            trigger_support_flipped = (
                total_trigger_support_on_flips / max(1, total_global_flips)
            )
            dense_m_per_client_mib = (total_binary * 4.0) / (1024.0 ** 2)

            self._diag_log(
                f"SERVER_DENSE_M_PRIVATE round={round_number} K={k_clients} mode=dense_fp32_mean_private_m_no_reflection "
                f"gamma={self.gamma:.3e} tau={self.tau:.3e} binary={total_binary} "
                f"global_flips={total_global_flips} "
                f"endpoint_union={total_endpoint_union} endpoint_mean_support={endpoint_mean_support:.3f} "
                f"endpoint_majority={total_endpoint_majority} endpoint_support_flipped={endpoint_support_flipped:.3f} "
                f"trigger_union={total_trigger_union} trigger_mean_support={trigger_mean_support:.3f} "
                f"trigger_majority={total_trigger_majority} trigger_support_flipped={trigger_support_flipped:.3f} "
                f"flip_no_endpoint={total_shared_flip_no_endpoint} "
                f"flip_no_client_trigger={total_shared_flip_no_trigger} "
                f"mean|m_shared|={mean_shared_abs:.3e} max|m_shared|={total_shared_m_max:.3e} "
                f"client_m_MAD={mean_client_m_mad:.3e} client_m_mean_abs={mean_client_m_abs:.3e} "
                f"client_m_relative_MAD={relative_m_mad:.6f} "
                f"dense_m_per_client_MiB={dense_m_per_client_mib:.2f} "
                f"dense_m_total_uplink_MiB={dense_m_per_client_mib * k_clients:.2f}"
            )

            if self._diag_layer_log_enabled(max(1, round_number)):
                for key, d in layer_diag.items():
                    endpoint_support = (
                        d["endpoint_votes"] / max(1, d["endpoint_union"])
                    )
                    trigger_support = (
                        d["trigger_votes"] / max(1, d["trigger_union"])
                    )
                    endpoint_support_flips_l = (
                        d["endpoint_support_flips"] / max(1, d["global_flips"])
                    )
                    trigger_support_flips_l = (
                        d["trigger_support_flips"] / max(1, d["global_flips"])
                    )
                    rel_mad_l = d["client_m_mad"] / max(1e-30, d["client_m_abs"])
                    self._diag_log(
                        f"SERVER_DENSE_M_PRIVATE_LAYER round={round_number} layer={key} n={d['n']} "
                        f"global_flips={d['global_flips']} repeat_server={d['repeat_server']} "
                        f"endpoint_union={d['endpoint_union']} endpoint_support={endpoint_support:.3f} "
                        f"endpoint_majority={d['endpoint_majority']} endpoint_support_flipped={endpoint_support_flips_l:.3f} "
                        f"trigger_union={d['trigger_union']} trigger_support={trigger_support:.3f} "
                        f"trigger_majority={d['trigger_majority']} trigger_support_flipped={trigger_support_flips_l:.3f} "
                        f"flip_no_endpoint={d['shared_flip_no_endpoint']} "
                        f"mean|m_shared|={d['mean_shared_abs']:.3e} max|m_shared|={d['max_shared_abs']:.3e} "
                        f"aligned_shared={d['mean_aligned_shared']:.3e} "
                        f"client_m_MAD={d['client_m_mad']:.3e} relative_MAD={rel_mad_l:.6f}"
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
            self.gamma = self.decay_factor * self.gamma
            self._last_decay_round = round_number
            did_decay = True

        if self.diag:
            self._diag_log(
                f"SERVER_PACK round={round_number} mode=dense_m_private_no_reflection "
                f"decay={int(did_decay)} gamma={old_gamma:.3e}->{self.gamma:.3e} "
                f"shared_m_downlink_MiB=0.00 reflection=0"
            )

        return {
            "bin_model": bin_model,
            "stats_model": stats_model,
            "gamma": self.gamma,
        }

    @torch.no_grad()
    def unpack_server_model(self, packed_model, current_model):
        bin_model = packed_model["bin_model"]
        stats_model = packed_model["stats_model"]
        self.gamma = packed_model["gamma"]

        if self.m_buffers is None:
            self.m_buffers = {}
        if self.round_global_weights is None:
            self.round_global_weights = {}

        total_binary = 0
        total_local_flip = 0
        total_server_flip = 0
        total_disagreement = 0
        total_local_rejected = 0
        total_server_imposed = 0
        total_trigger_new = 0
        total_trigger_dis = 0
        total_old_aligned_sum = 0.0
        total_new_aligned_sum = 0.0
        total_m_abs_sum = 0.0

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
            m = self._prepare_m_buffer(key, client_tensor)

            layer_diag = None
            if self.has_local_update and key in self.round_global_weights:
                old_global_weight = self._sign_binary(
                    self.round_global_weights[key].to(
                        device=client_tensor.device,
                        dtype=client_tensor.dtype,
                    )
                )
                local_weight = self._sign_binary(client_tensor)

                local_flip = local_weight != old_global_weight
                server_flip = new_global_weight != old_global_weight
                disagreement = local_flip != server_flip
                local_rejected = local_flip & (~server_flip)
                server_imposed = (~local_flip) & server_flip

                # Diagnostic only. The private momentum is intentionally not changed.
                aligned_old = old_global_weight * m
                aligned_new = new_global_weight * m
                trigger_new = aligned_new > self.tau

                n = client_tensor.numel()
                local_n = int(local_flip.sum().item())
                server_n = int(server_flip.sum().item())
                dis_n = int(disagreement.sum().item())
                rejected_n = int(local_rejected.sum().item())
                imposed_n = int(server_imposed.sum().item())
                trigger_new_n = int(trigger_new.sum().item())
                trigger_dis_n = int((trigger_new & disagreement).sum().item())

                total_binary += n
                total_local_flip += local_n
                total_server_flip += server_n
                total_disagreement += dis_n
                total_local_rejected += rejected_n
                total_server_imposed += imposed_n
                total_trigger_new += trigger_new_n
                total_trigger_dis += trigger_dis_n
                total_old_aligned_sum += float(aligned_old.sum().item())
                total_new_aligned_sum += float(aligned_new.sum().item())
                total_m_abs_sum += float(m.abs().sum().item())

                if self.diag:
                    layer_diag = {
                        "n": n,
                        "local": local_n,
                        "server": server_n,
                        "dis": dis_n,
                        "rejected": rejected_n,
                        "imposed": imposed_n,
                        "trigger_new": trigger_new_n,
                        "trigger_dis": trigger_dis_n,
                        "mean_abs_m": float(m.abs().mean().item()),
                        "mean_aligned_old": float(aligned_old.mean().item()),
                        "mean_aligned_new": float(aligned_new.mean().item()),
                    }

            # Only binary weights are synchronized. m remains exactly the client's local m.
            client_tensor.copy_(new_global_weight)
            self.round_global_weights[key] = new_global_weight.detach().cpu()

            if layer_diag is not None:
                sync_layers[key] = layer_diag

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

        if self.stats_optimizer is not None:
            adam_state_params_before = len(self.stats_optimizer.state)
            for state in self.stats_optimizer.state.values():
                step = state.get("step", 0)
                if torch.is_tensor(step):
                    step = int(step.item())
                else:
                    step = int(step)
                adam_step_max_before = max(adam_step_max_before, step)

            if self.sync_stats and self.reset_stats_optimizer_on_sync:
                self.stats_optimizer.state.clear()
                adam_reset = 1

        if self.diag:
            round_number = self._diag_client_round
            disagreement_rate = total_disagreement / max(1, total_binary)
            rejected_rate = total_local_rejected / max(1, total_binary)
            imposed_rate = total_server_imposed / max(1, total_binary)
            trigger_dis_rate = total_trigger_dis / max(1, total_disagreement)
            mean_abs_m = total_m_abs_sum / max(1, total_binary)
            mean_aligned_old = total_old_aligned_sum / max(1, total_binary)
            mean_aligned_new = total_new_aligned_sum / max(1, total_binary)
            stats_mean_overwrite = stats_sync_abs_sum / max(1, stats_sync_float_numel)

            self._diag_log(
                f"SYNC_PRIVATE_M_NO_REFLECT round={round_number} gamma={self.gamma:.3e} "
                f"local_flip={total_local_flip} server_flip={total_server_flip} "
                f"disagreement={total_disagreement} disagreement_rate={disagreement_rate:.6e} "
                f"local_rejected={total_local_rejected} rejected_rate={rejected_rate:.6e} "
                f"server_imposed={total_server_imposed} imposed_rate={imposed_rate:.6e} "
                f"trigger_under_new_global={total_trigger_new} "
                f"trigger_on_disagreement={total_trigger_dis} trigger_dis_rate={trigger_dis_rate:.6f} "
                f"mean|m_private|={mean_abs_m:.3e} "
                f"mean_aligned_old={mean_aligned_old:.3e} mean_aligned_new={mean_aligned_new:.3e} "
                f"m_modified=0"
            )

            if self._diag_layer_log_enabled(max(1, round_number)):
                for key, d in sync_layers.items():
                    self._diag_log(
                        f"SYNC_PRIVATE_M_NO_REFLECT_LAYER round={round_number} layer={key} n={d['n']} "
                        f"local={d['local']} server={d['server']} dis={d['dis']} "
                        f"rejected={d['rejected']} imposed={d['imposed']} "
                        f"trigger_new={d['trigger_new']} trigger_dis={d['trigger_dis']} "
                        f"mean|m|={d['mean_abs_m']:.3e} "
                        f"aligned_old={d['mean_aligned_old']:.3e} aligned_new={d['mean_aligned_new']:.3e} "
                        f"m_modified=0"
                    )

            self._diag_log(
                f"STATS_SYNC round={round_number} enabled={int(self.sync_stats)} "
                f"tensors={stats_sync_tensors} float_numel={stats_sync_float_numel} "
                f"mean_abs_overwrite={stats_mean_overwrite:.3e} "
                f"max_abs_overwrite={stats_sync_abs_max:.3e} "
                f"reset_adam={adam_reset} adam_state_params_before={adam_state_params_before} "
                f"adam_step_max_before={adam_step_max_before}"
            )

        self.has_local_update = False
        return current_model

