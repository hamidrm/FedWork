# STC
import math
import torch
from collections import defaultdict
from core.FederatedLearningClass import *
from utils.logger import *
from utils.common import Common

class STC(FederatedLearningClass):
    """
    Sparse Ternary Compression with optional Error-Feedback.
    Client sends, per-trainable tensor: indices, signs (±1), and a scale (mean |selected|).
    Server reconstructs deltas and aggregates them into the global model.
    """

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        # Client selection
        self.contributors_percent = float(self.get_arg(int, "contributors_percent", 100)) / 100.0

        # --- STC knobs ---
        # You may specify either:
        #   - sparsity_percent in [0..100]  (e.g., 1 -> keep top 1%)
        #   - sparsity in (0..1]            (e.g., 0.01 -> keep top 1%)
        sp_percent = Common.get_param_in_args(self.extra_args, "sparsity_percent", None)
        sp_frac    = Common.get_param_in_args(self.extra_args, "sparsity", None)
        if sp_percent is not None:
            try:
                self.sparsity = float(sp_percent) / 100.0
            except:
                self.sparsity = 0.01
        elif sp_frac is not None:
            try:
                sf = float(sp_frac)
                self.sparsity = sf / 100.0 if sf > 1.0 else sf
            except:
                self.sparsity = 0.01
        else:
            self.sparsity = 0.01  # default: keep 1%

        # per-tensor top-k (True) vs global top-k across all trainables (False)
        self.layerwise = bool(int(self.get_arg(int, "layerwise", 1)))

        # error feedback on clients (residual memory)
        self.error_feedback = bool(int(self.get_arg(int, "error_feedback", 1)))

        # deterministic tie-breaking (optional)
        self.random_seed = self.get_arg(int, "random_seed", 0)

        # ---- client-side state ----
        # When running on clients, we’ll create/maintain residual buffers per parameter.
        # (On server this dict remains empty / unused.)
        self._residuals = {}  # {param_name: tensor}

    # ---- metadata ----
    def get_name(self):
        return "STC"

    # ---- server hook ----
    def init_method(self, server=None):
        if server is not None:
            self.server = server
            logger.log_normal(
                f"[STC] sparsity={self.sparsity*100:.2f}%  "
                f"layerwise={self.layerwise}  "
                f"error_feedback={self.error_feedback}  "
                f"contributors={int(self.contributors_percent*100)}%"
            )

    # ---- client selection (server) ----
    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)

    # =========================
    # Client-side: PACK (compress)
    # =========================
    def _topk_mask_flat(self, flat_tensor, k):
        if k <= 0:
            return torch.zeros_like(flat_tensor, dtype=torch.bool)
        if k >= flat_tensor.numel():
            return torch.ones_like(flat_tensor, dtype=torch.bool)
        # top-k by absolute value
        vals = flat_tensor.abs()
        thresh = torch.topk(vals, k, largest=True, sorted=False).values.min()
        return (vals >= thresh)

    def _ensure_residual(self, name, like_tensor):
        if (not self.error_feedback):
            return None
        r = self._residuals.get(name, None)
        if r is None or (r.shape != like_tensor.shape) or (r.device != like_tensor.device) or (r.dtype != like_tensor.dtype):
            r = torch.zeros_like(like_tensor)
            self._residuals[name] = r
        return r

    @torch.no_grad()
    def _compress_param(self, name, delta, k):
        """
        Returns a Python-serializable dict:
            { "shape": [...], "idx": [ints], "signs": [±1 ints], "scale": float }
        and updates residuals if enabled.
        """
        device = delta.device
        flat = delta.view(-1)

        if k <= 0 or flat.numel() == 0:
            # no selection
            if self.error_feedback:
                # residual = delta - 0
                self._residuals[name] = flat.view_as(delta).clone()
            return {"shape": list(delta.shape), "idx": [], "signs": [], "scale": 0.0}

        # top-k mask
        mask = self._topk_mask_flat(flat, k)
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        sel = flat[idx]
        if sel.numel() == 0:
            if self.error_feedback:
                self._residuals[name] = flat.view_as(delta).clone()
            return {"shape": list(delta.shape), "idx": [], "signs": [], "scale": 0.0}

        # ternary: ±mu with mu = mean |sel|
        mu = sel.abs().mean().item()
        signs = torch.sign(sel)  # -1 or +1

        # build ternary flat for residual update
        ternary_flat = torch.zeros_like(flat)
        ternary_flat[idx] = signs * mu

        if self.error_feedback:
            # residual = (delta + residual_old) - ternary
            # Note: delta passed in already included residual_old (see pack_client_model).
            res = (flat - ternary_flat).view_as(delta)
            self._residuals[name] = res.detach().clone()

        # Python-serializable
        signs_list = [1 if s.item() > 0 else -1 for s in signs]
        return {
            "shape": list(delta.shape),
            "idx": idx.tolist(),
            "signs": signs_list,
            "scale": float(mu)
        }

    @torch.no_grad()
    def pack_client_model(self, raw_model, global_model, id):
        """
        Called on the client after local training.
        We send sparse ternary *deltas* for trainable params and include full values
        for non-trainables (BN stats etc.) so the server can keep buffers consistent.
        """
        if self.random_seed:
            torch.manual_seed(self.random_seed + int(id) + int(getattr(self, "server", None).round_number if hasattr(self, "server") else 0))

        # Build deltas and apply error feedback (if enabled).
        # If layerwise=False, we compute a single global k and split back per tensor.
        trainable_names = [k for k in raw_model.keys() if Common.is_trainable(global_model, k)]
        nontrainable_names = [k for k in raw_model.keys() if not Common.is_trainable(global_model, k)]

        # Compute flattened deltas (+ residuals) for global top-k if needed
        if not self.layerwise:
            flats = []
            shapes = []
            names = []
            for k in trainable_names:
                d = raw_model[k] - global_model[k]
                r = self._ensure_residual(k, d)
                d_eff = d if (not self.error_feedback) else (d + r)
                flats.append(d_eff.view(-1))
                shapes.append(d.shape)
                names.append(k)
            if len(flats) == 0:
                total_k = 0
            else:
                total_n = sum([f.numel() for f in flats])
                total_k = max(1, int(self.sparsity * total_n)) if self.sparsity > 0 else 0

            # one big vector
            if total_k > 0:
                big = torch.cat(flats, dim=0)
                big_mask = self._topk_mask_flat(big, total_k)
            else:
                big_mask = torch.zeros(sum([f.numel() for f in flats]), dtype=torch.bool, device=flats[0].device if flats else "cpu")

            # split mask back and compress per tensor
            payload = {}
            offset = 0
            for k, f, shape in zip(names, flats, shapes):
                m = big_mask[offset : offset + f.numel()]
                offset += f.numel()
                if m.sum().item() == 0:
                    # update residual (delta_eff - 0) if EF
                    if self.error_feedback:
                        self._residuals[k] = f.view(*shape).detach().clone()
                    payload[k] = {"shape": list(shape), "idx": [], "signs": [], "scale": 0.0}
                else:
                    sel = f[m]
                    mu = sel.abs().mean().item()
                    idx = torch.nonzero(m, as_tuple=False).view(-1).tolist()
                    signs_list = [1 if s.item() > 0 else -1 for s in torch.sign(sel)]
                    # residual update
                    if self.error_feedback:
                        # reconstruct ternary flat to update residual
                        flat = f.detach().clone()
                        flat_tern = torch.zeros_like(flat)
                        flat_tern[m] = torch.tensor(signs_list, device=flat.device, dtype=flat.dtype) * float(mu)
                        self._residuals[k] = (flat - flat_tern).view(*shape).detach().clone()
                    payload[k] = {"shape": list(shape), "idx": idx, "signs": signs_list, "scale": float(mu)}
        else:
            # layer-wise: independent top-k per tensor
            payload = {}
            for k in raw_model.keys():
                if Common.is_trainable(global_model, k):
                    d = raw_model[k] - global_model[k]
                    r = self._ensure_residual(k, d)
                    d_eff = d if (not self.error_feedback) else (d + r)
                    n = d_eff.numel()
                    keep = max(1, int(self.sparsity * n)) if self.sparsity > 0 else 0
                    payload[k] = self._compress_param(k, d_eff, keep)

        # Include non-trainable params as full tensors (copied by server from first client).
        nontrain = {k: raw_model[k] for k in nontrainable_names}

        return {
            "__type__": "stc",
            "sparsity": self.sparsity,
            "layerwise": self.layerwise,
            "error_feedback": self.error_feedback,
            "tensors": payload,
            "non_trainable": nontrain,
        }

    # =========================
    # Server-side: UNPACK (decompress to dense deltas)
    # =========================
    @torch.no_grad()
    def unpack_client_model(self, packed_model):
        """
        Server decodes the client's payload into a dict:
            trainables -> dense delta tensors
            non-trainables -> untouched (copied over as-is)
        """
        if not isinstance(packed_model, dict) or packed_model.get("__type__") != "stc":
            # Not an STC payload; return as-is.
            return packed_model

        tensors = {}
        tensors_payload = packed_model["tensors"]

        for name, enc in tensors_payload.items():
            shape = tuple(enc["shape"])
            idx = enc["idx"]
            signs = enc["signs"]
            scale = float(enc["scale"])
            numel = 1
            for s in shape:
                numel *= s

            if len(idx) == 0 or scale == 0.0:
                tensors[name] = torch.zeros(numel, dtype=torch.float32).view(*shape)
            else:
                flat = torch.zeros(numel, dtype=torch.float32)
                # write ±scale at idx
                signs_tensor = torch.tensor(signs, dtype=torch.float32)
                flat[torch.tensor(idx, dtype=torch.long)] = signs_tensor * scale
                tensors[name] = flat.view(*shape)

        # carry non-trainables along
        for name, full_val in packed_model.get("non_trainable", {}).items():
            tensors[name] = full_val

        return tensors

    # =========================
    # Server-side: AGGREGATE
    # =========================
    @torch.no_grad()
    def aggregate(self, clients_models, global_model):
        """
        clients_models: list of (client_id, model_dict) where model_dict
                        has dense deltas for trainables and full tensors for non-trainables.
        We apply dataset-weighted averaging to deltas and add to the global model.
        For non-trainables we copy the first client’s value (like FedYogi).
        """
        if len(clients_models) == 0:
            return

        # Prepare accumulators
        delta_sum = {k: torch.zeros_like(v) for k, v in global_model.items() if Common.is_trainable(global_model, k)}
        total_weight = 0.0

        # Sum deltas with dataset weights
        first_client_nt = None
        for cid, client_dict in clients_models:
            w = float(self.datasets_weights[cid]) if cid in self.datasets_weights else 1.0
            total_weight += w
            for k in global_model.keys():
                if Common.is_trainable(global_model, k):
                    if k in client_dict:
                        d = client_dict[k]
                        # match dtype/device
                        d = d.to(device=global_model[k].device, dtype=global_model[k].dtype)
                        delta_sum[k] += d * w
                else:
                    if first_client_nt is None and k in client_dict:
                        # Remember non-trainables from the first client seen
                        if first_client_nt is None:
                            first_client_nt = {}
                        first_client_nt[k] = client_dict[k]

        if total_weight == 0.0:
            total_weight = 1.0

        # Apply averaged delta
        for k in delta_sum.keys():
            global_model[k] = global_model[k] + (delta_sum[k] / total_weight)

        # Update non-trainables from the first client (if present)
        if first_client_nt is not None:
            for k, v in first_client_nt.items():
                global_model[k] = v