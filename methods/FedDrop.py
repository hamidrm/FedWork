# FedDrop (uplink-only; forward+grad masking; BN+1D always aggregated)
# Works with your FederatedLearningClass and training glue as-is.
import torch
import torch.nn as nn
import hashlib
from typing import Dict
from core.FederatedLearningClass import *
from utils.logger import *
from utils.common import Common


def _is_bn_buffer(name: str) -> bool:
    return name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked")

def _is_1d_param(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and t.ndim == 1

def _is_heavy_weight(t: torch.Tensor) -> bool:
    # mask only 2D+ weights (e.g., Conv/Linear) along dim 0 (out-channels / out-features)
    return isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] > 0


class FedDrop(FederatedLearningClass):
    """
    Uplink-only federated dropout (Wen et al.-style):
      - Server broadcasts the full model (downlink unchanged).
      - All clients use the SAME deterministic per-parameter keep indices each round.
      - Forward hooks zero-out dropped outputs; grad hooks zero grads for dropped rows.
      - Clients upload only kept rows of 2D+ weights; 1D params & BN buffers are uploaded fully.
      - Server aggregates masked indices and full tensors appropriately (dataset-weighted).
    """
    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        self.contributors_percent = float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        self.keep_rate = float(self.get_arg(float, "keep_rate", 0.9))
        self.min_keep = int(self.get_arg(int, "min_keep", 1))

        # per-client round state
        self._current_mask: Dict[str, torch.Tensor] = None
        self._grad_hook_handles = []
        self._fwd_hook_handles = []
        self._masks_installed = False

    def get_name(self):
        return "FedDrop-UplinkOnly"

    # -------------------- client selection --------------------
    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)

    # -------------------- deterministic mask building --------------------
    @staticmethod
    def _first_tensor_key(state: Dict[str, torch.Tensor]) -> str:
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                return k
        return sorted(state.keys())[0]

    def _round_seed_from_model(self, model_state: Dict[str, torch.Tensor]) -> int:
        key = self._first_tensor_key(model_state)
        t = model_state[key]
        with torch.no_grad():
            flat = t.detach().reshape(-1)
            sample = flat[:16].cpu().numpy().tobytes()
        h = hashlib.sha256()
        h.update(sample)
        h.update(str(t.shape).encode("utf-8"))
        h.update(str(self.fl_context.get("seed", 0)).encode("utf-8"))
        return int.from_bytes(h.digest()[:8], "little", signed=False)

    def _indices_for_tensor(self, name: str, tensor: torch.Tensor, seed: int) -> torch.Tensor:
        if not _is_heavy_weight(tensor):
            return None
        dim0 = tensor.shape[0]
        k = max(self.min_keep, int(round(self.keep_rate * dim0)))
        k = min(k, dim0)
        if k == dim0:
            return torch.arange(dim0, dtype=torch.long)

        h = hashlib.sha256()
        h.update(str(seed).encode("utf-8"))
        h.update(name.encode("utf-8"))
        g = torch.Generator()
        g.manual_seed(int.from_bytes(h.digest()[:8], "little", signed=False))
        perm = torch.randperm(dim0, generator=g)
        return torch.sort(perm[:k]).values

    def _build_mask(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seed = self._round_seed_from_model(model_state)
        mask = {}
        for name, t in model_state.items():
            if Common.is_trainable(model_state, name):
                idx = self._indices_for_tensor(name, t, seed)
                if idx is not None:
                    mask[name] = idx
        return mask

    # -------------------- server broadcast (downlink = full) --------------------
    def pack_server_model(self, raw_model):
        return raw_model

    def unpack_server_model(self, packed_model):
        # reset per-round client-side state
        self._current_mask = None
        self._remove_masks()
        self._masks_installed = False
        return packed_model

    # -------------------- hook management (client) --------------------
    def _remove_masks(self):
        for h in self._grad_hook_handles:
            try: h.remove()
            except: pass
        self._grad_hook_handles = []
        for h in self._fwd_hook_handles:
            try: h.remove()
            except: pass
        self._fwd_hook_handles = []

    def _module_from_param_name(self, model: nn.Module, param_name: str):
        """
        Map 'layer.sub.weight' -> module object if it's Conv/Linear; else None.
        """
        if param_name.endswith(".weight") or param_name.endswith(".bias"):
            mod_name = param_name.rsplit(".", 1)[0]
            for n, m in model.named_modules():
                if n == mod_name and isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    return m
        return None

    def _install_masks(self, model: nn.Module, mask_by_name: Dict[str, torch.Tensor]):
        self._remove_masks()

        for name, p in model.named_parameters():
            # only mask 2D+ weights along dim 0
            if not p.requires_grad or not _is_heavy_weight(p):
                continue
            kept = mask_by_name.get(name, None)
            if kept is None:
                continue

            kept = kept.to(p.device, non_blocking=True).long()
            n0 = p.shape[0]
            drop = torch.ones(n0, dtype=torch.bool, device=p.device)
            drop[kept] = False
            drop_idx = torch.arange(n0, device=p.device)[drop]

            # ---- grad hook: zero grads for dropped rows/filters
            def _grad_hook(grad, drop_idx=drop_idx):
                if grad is None:
                    return grad
                return grad.index_fill(0, drop_idx, 0)
            self._grad_hook_handles.append(p.register_hook(_grad_hook))

            # ---- forward hook: zero activations for dropped outputs
            mod = self._module_from_param_name(model, name)
            if mod is not None:
                def _fwd_hook(module, inp, out, kept=kept):
                    # Conv: [N, C, H, W], Linear: [N, F]
                    if torch.is_tensor(out) and out.dim() >= 2:
                        C = out.shape[1]
                        if kept.max().item() < C:
                            drop_mask = torch.ones(C, dtype=torch.bool, device=out.device)
                            drop_mask[kept] = False
                            idx = torch.arange(C, device=out.device)[drop_mask]
                            out.index_fill_(1, idx, 0)
                    return out
                self._fwd_hook_handles.append(mod.register_forward_hook(_fwd_hook))

        logger.log_debug(f"[FedDrop] Installed {len(self._grad_hook_handles)} grad masks and {len(self._fwd_hook_handles)} fwd masks")

    # -------------------- training glue (client) --------------------
    def train(self, client_train_dict: dict):
        """
        Called each batch. Returning None => use default training step.
        We lazily install hooks on the first batch of each epoch.
        """
        model = client_train_dict["client_model"]

        if not self._masks_installed:
            gstate = client_train_dict.get("global_model_state") or model.state_dict()
            if self._current_mask is None:
                self._current_mask = self._build_mask(gstate)
            self._install_masks(model, self._current_mask)
            self._masks_installed = True

        return None  # use framework's standard step

    def train_after_optimization(self, client_train_dict: dict, epoch_num):
        # reset between epochs (same round) so we reinstall cleanly on next epoch
        self._remove_masks()
        self._masks_installed = False
        return None

    # -------------------- client upload (uplink slices) --------------------
    def pack_client_model(self, raw_model, global_model, id):
        mask = self._current_mask or self._build_mask(global_model)
        partial = {}
        elem_count = 0

        for k, v in raw_model.items():
            # always include BN buffers fully
            if _is_bn_buffer(k):
                t = v.detach().clone()
                partial[k] = t
                elem_count += t.numel()
                continue

            # 1D params (bias, BN affine) — include fully
            if _is_1d_param(v):
                t = v.detach().clone()
                partial[k] = t
                elem_count += t.numel()
                continue

            if Common.is_trainable(global_model, k) and _is_heavy_weight(v):
                kept = mask.get(k, None)
                if kept is None:
                    t = v.detach().clone()
                else:
                    kept_dev = kept.to(v.device, non_blocking=True).long()
                    t = v.index_select(0, kept_dev).detach().clone()
                partial[k] = t
                elem_count += t.numel()

        # Optional: quick comms estimate (elements * element_size bytes)
        try:
            elem_size = next(iter(raw_model.values())).element_size()
            logger.log_debug(f"[FedDrop] Client {id} uplink ~{elem_count * elem_size / 1e6:.2f} MB")
        except Exception:
            pass

        return partial

    # -------------------- aggregation (server) --------------------
    @torch.no_grad()
    def aggregate(self, clients_models, global_model):
        # rebuild mask from current global so server matches clients
        mask = self._build_mask(global_model)

        def _w(cid):
            if isinstance(self.datasets_weights, dict):
                return float(self.datasets_weights.get(cid, 1.0))
            return 1.0

        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, torch.Tensor] = {}

        # accumulate
        for cid, part in clients_models:
            w = _w(cid)
            for k, t in part.items():
                if k not in sums:
                    sums[k] = torch.zeros_like(global_model[k], dtype=global_model[k].dtype, device=global_model[k].device)
                    if _is_heavy_weight(global_model[k]) and k in mask:
                        counts[k] = torch.zeros(global_model[k].shape[0], dtype=sums[k].dtype, device=sums[k].device)
                    else:
                        counts[k] = torch.tensor(0.0, dtype=sums[k].dtype, device=sums[k].device)

                if _is_bn_buffer(k) or _is_1d_param(global_model[k]) or k not in mask or not _is_heavy_weight(global_model[k]):
                    sums[k].add_(t.to(sums[k].device, dtype=sums[k].dtype), alpha=w)
                    counts[k] = counts[k] + w
                else:
                    kept = mask[k].to(global_model[k].device, non_blocking=True).long()
                    sums[k][kept] += t.to(sums[k].device, dtype=sums[k].dtype) * w
                    counts[k][kept] += w

        # write-back (weighted average)
        for k in sums.keys():
            if _is_bn_buffer(k) or _is_1d_param(global_model[k]) or k not in mask or not _is_heavy_weight(global_model[k]):
                denom = float(counts[k].item()) if counts[k].ndim == 0 else counts[k]
                denom = max(denom, 1e-12) if isinstance(denom, float) else denom.clamp_min(1e-12)
                global_model[k] = sums[k] / denom
            else:
                kept = mask[k].to(global_model[k].device, non_blocking=True).long()
                denom = counts[k][kept].clamp_min(1e-12)
                if sums[k].ndim == 1:
                    global_model[k][kept] = sums[k][kept] / denom
                else:
                    global_model[k][kept] = sums[k][kept] / denom.view(-1, *([1] * (sums[k].ndim - 1)))

    # -------------------- misc --------------------
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"[FedDrop] Received {num_of_received_model}/{self.num_of_contributor_nodes} client updates")
        return super().ready_to_aggregate(num_of_received_model)

    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        progress = "["
        for i in range(20):
            r = (self.server.round_number / self.num_of_rounds) * 20
            if i < r:
                progress += "#"
            else:
                progress += " "
        progress += "]"
        logger.log_normal(f"Round {self.server.round_number} / {self.num_of_rounds} is starting {progress}")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != self.num_of_rounds:
            self.server.start_round(self.clients_epochs)

            return (eval_loss, eval_accuracy)
        else:
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None