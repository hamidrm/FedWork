import math
import pickle
import types
import torch
import numpy as np
from core.FederatedLearningClass import *
from utils.logger import *


# ---------------------------------------------------------------------------
# Inline wire codecs (formerly methods/_biprune_wire.py)
# Encodings, metadata, and decoding semantics are unchanged.
# ---------------------------------------------------------------------------

def _array_index(indices):
    if torch.is_tensor(indices):
        arr = indices.detach().reshape(-1).cpu().numpy().astype(np.int64, copy=False)
    else:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.size:
        if arr[0] < 0 or np.any(arr[1:] <= arr[:-1]):
            raise ValueError("Indices must be sorted, unique and nonnegative")
    return arr


def rice_encode(sorted_indices):
    """Return (k, packed_byte_string, count, meaningful_bits)."""
    ix = _array_index(sorted_indices)
    n = int(ix.size)
    if not n:
        return 0, b"", 0, 0
    gaps = ix.copy()
    gaps[1:] = ix[1:] - ix[:-1] - 1
    if gaps.min() < 0:
        raise ValueError("Negative gap")
    # k is small for sparse events, so searching 0..24 is sufficient for 13M weights.
    # Up to 31 supports larger models without a hard-coded distribution.
    costs = [int((gaps >> k).sum(dtype=np.int64)) + n * (k + 1) for k in range(32)]
    k = min(range(32), key=lambda i: costs[i])
    q = gaps >> k
    lengths = q + (k + 1)
    ends = np.cumsum(lengths, dtype=np.int64)
    total_bits = int(ends[-1])
    begin = ends - lengths
    zero_pos = begin + q
    # Every bit up to terminators is one. Terminations are zero; remainders
    # are written separately. The numpy builder avoids per-index Python loops.
    bits = np.ones(total_bits, dtype=np.uint8)
    bits[zero_pos] = 0
    if k:
        shifts = np.arange(k, dtype=np.int64)
        pos = zero_pos[:, None] + 1 + shifts[None, :]
        bits[pos] = ((gaps[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return int(k), np.packbits(bits, bitorder="little").tobytes(), n, total_bits


def rice_decode(k, blob, count, nbits=None, numel=None):
    """Reconstruct exact positions from a packed Golomb--Rice bitstream."""
    count = int(count)
    k = int(k)
    if not (0 <= k <= 31 and count >= 0):
        raise ValueError("Invalid Rice header")
    if count == 0:
        if blob:
            raise ValueError("Nonempty Rice bytes for zero indices")
        return np.empty(0, dtype=np.int64)
    raw = memoryview(blob)
    nbits = int(nbits) if nbits is not None else len(raw) * 8
    pos = 0
    previous = -1
    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        quotient = 0
        while True:
            if pos >= nbits:
                raise ValueError("Truncated Rice stream (unary)")
            bit = (raw[pos >> 3] >> (pos & 7)) & 1
            pos += 1
            if not bit:
                break
            quotient += 1
        remainder = 0
        if pos + k > nbits:
            raise ValueError("Truncated Rice stream (remainder)")
        for j in range(k):
            remainder |= ((raw[pos >> 3] >> (pos & 7)) & 1) << j
            pos += 1
        gap = (quotient << k) | remainder
        previous += gap + 1
        if numel is not None and previous >= int(numel):
            raise ValueError("Rice index outside tensor")
        out[i] = previous
    if pos != nbits:
        raise ValueError("Rice payload bit count does not match decoded count")
    return out


def pack_signs(signs):
    """1 bit per ternary nonzero sign (+1=1)."""
    x = signs.detach().reshape(-1).cpu().numpy() if torch.is_tensor(signs) else np.asarray(signs)
    return np.packbits(np.asarray(x > 0, dtype=np.uint8), bitorder="little").tobytes()


def unpack_signs(blob, count):
    count = int(count)
    if len(blob) != (count + 7) // 8:
        raise ValueError("Bad sign stream length")
    a = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[:count]
    return a.astype(np.int8) * 2 - 1


def pack_binary(b):
    """Losslessly encode a ±1 tensor as one bit per entry, no shape overhead."""
    x = b.detach().reshape(-1).cpu().numpy() if torch.is_tensor(b) else np.asarray(b).reshape(-1)
    return np.packbits(np.asarray(x > 0, dtype=np.uint8), bitorder="little").tobytes()


def unpack_binary(blob, count, shape=None):
    count = int(count)
    if len(blob) != (count + 7) // 8:
        raise ValueError("Bad binary base stream length")
    a = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[:count]
    w = torch.from_numpy((a.astype(np.int8) * 2 - 1).copy())
    return w.reshape(shape) if shape is not None else w



class _BiPruneBiRealSign(torch.autograd.Function):
    """BiPruneFL-only BiRealAct surrogate for the shared VGG7 quantizers.

    Forward intentionally preserves VGG7Bin's convention sign(0)=+1; it does
    not alter the model's inference function.  Backward matches the released
    BiPruneFL F_BinAct.backward:
         grad_input = grad_output * 2*(1 - abs(clamp(input, -1, 1))).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        slope = 2.0 * (1.0 - x.clamp(-1.0, 1.0).abs())
        return grad_output * slope


def _biprune_binary_input_forward(module, x):
    """Per-instance replacement, never a global class monkeypatch."""
    if not module.binary:  # first RGB convolution remains unquantized
        return x
    return _BiPruneBiRealSign.apply(x)


class BiPruneFL(FederatedLearningClass):
    """BiPruneFL adapted to FedWork's existing BinaryNet.

    Lee & Jang, IEEE Access 2025.

    Preserved components:
      * frozen random real weights W and binary base B=sign(W)
      * trainable pruning scores S on the server
      * layer-wise top-(1-p) mask M from |S|
      * layer gain alpha = ||M*W||_1 / ||M||_1
      * STE score gradient on clients
      * Sparse Ternary Compression (STC) + persistent error feedback
      * Binary Mask Compression (BMC) downstream
      * REAL Golomb-Rice bitstreams for sparse index lists, plus packed signs
      * lossless score-sign synchronization, needed for official STE gradient
      * shared-seed regeneration of frozen real weights for exact STE gradient
      * server SGD with cosine LR decay
      * one weight-recycling event (default T/2, rate 0.2)

    FedWork/BinaryNet adaptation:
      * same BinaryNet topology is kept for controlled comparison with FedBop
      * default client_grad_steps=1 matches the paper's one local backprop step
      * client_grad_steps=0 uses all local minibatches as an equal-local-work ablation
      * BinaryNet's non-binary state is sent only for server evaluation and is
        included in strict communication accounting
      * full participation gives exact BMC with one common previous-mask cache;
        partial participation safely falls back to absolute masks
      * paper-main dense FP32-gradient condition is selectable with compression=fp32
      * BiRealAct backward for binary activations is installed on this method's
        own client quantizer instances, without editing the shared VGG7 source
      * internal k-bit activation-gradient quantization from the paper is not
        implemented because it requires modifying BinaryNet/autograd internals;
        this affects computation, not the STC/BMC communication protocol
    """

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        self._method_name = method_name
        self._fl_context = fl_context

        self.contributors_percent = float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        self.pruning_ratio = self.get_arg(float, "pruning_ratio", 0.70)
        self.score_lr = self.get_arg(float, "score_lr", 0.10)
        self.score_lr_min = self.get_arg(float, "score_lr_min", 0.0)
        self.stc_keep_ratio = self.get_arg(float, "stc_keep_ratio", 0.01)
        self.compression = self.get_arg(str, "compression", "stc").lower()
        self.fast_local_step = self.get_arg(int, "fast_local_step", 1)
        self.client_grad_steps = self.get_arg(int, "client_grad_steps", 1)
        # BiReal surrogate is installed on this method's client-model instance
        # only.  "clipped" preserves the original shared VGG7's STE.
        self.activation_ste = self.get_arg(str, "activation_ste", "bireal").lower()
        self._activation_patch_logged = False
        self.recycle_rate = self.get_arg(float, "recycle_rate", 0.20)
        self.recycle_round = self.get_arg(int, "recycle_round", 0)
        self.weight_seed = self.get_arg(int, "weight_seed", 42)
        self.score_seed = self.get_arg(int, "score_seed", 4242)
        self.diag = self.get_arg(int, "diag", 1)
        self.diag_layer_every = self.get_arg(int, "diag_layer_every", 20)

        if not (0.0 <= self.pruning_ratio < 1.0):
            raise ValueError("pruning_ratio must be in [0,1)")
        if not (0.0 < self.stc_keep_ratio <= 1.0):
            raise ValueError("stc_keep_ratio must be in (0,1]")
        if self.compression not in ("stc", "fp32"):
            raise ValueError("compression must be stc or fp32")
        if self.client_grad_steps < 0:
            raise ValueError("client_grad_steps must be >=0")
        if self.activation_ste not in ("bireal", "clipped"):
            raise ValueError("activation_ste must be bireal or clipped")
        if not (0.0 <= self.recycle_rate < 0.5):
            raise ValueError("recycle_rate must be in [0,0.5)")

        # Server state.
        self.server_initialized = False
        self.W_buffers = {}
        self.B_buffers = {}
        self.S_buffers = {}
        self.M_buffers = {}
        self.alpha_buffers = {}
        self._last_broadcast_masks = {}
        # The official STE has dL/dS proportional to sign(S). Since the
        # client does not receive full FP32 S, transmit its signs losslessly.
        self._last_broadcast_score_signs = {}
        self._last_broadcast_score_zeros = {}
        self._pending_recycle_broadcast = False
        self._last_recycle_round = -1
        self._server_pack_cache_round = None
        self._server_pack_cache_payload = None
        self._last_k_clients = self._known_client_count(1)

        # Communication totals. Downlink is logical broadcast unless explicitly
        # multiplied by K in the log.
        self.comm_uplink_bits_total = 0
        self.comm_downlink_paper_bits_total = 0
        self.comm_downlink_strict_bits_total = 0

        # Client state.
        self.client_B = {}
        self.client_M = {}
        self.client_alpha = {}
        self.client_W = {}  # frozen FP32 W regenerated from shared seed
        self.client_score_sign = {}  # sign of S, transferred as packed bits/deltas
        self._w_device_cache = {}
        self._score_sign_device_cache = {}
        self.error_feedback = {}
        self.score_grad_sum = {}
        # The released BiPruneFL also optimizes nonbinary normalization/bias
        # parameters on the server. The first FedWork port dropped these grads.
        self.aux_grad_sum = {}
        self.score_grad_steps = 0
        self._scheduled_local_round = 1
        self._diag_client_round = 0

    def get_name(self):
        return "BiPruneFL"

    def init_method(self, server=None):
        if server is not None:
            self.server = server

    def select_clients_to_train(self, all_clients):
        if self.contributors_percent != 1.0:
            return self.select_random_clients(all_clients, self.contributors_percent)
        return super().select_clients_to_train(all_clients)

    def _log(self, msg):
        if not self.diag:
            return
        text = f"[BiPruneFL] {msg}"
        try:
            logger.log_info(text)
        except Exception:
            try:
                logger.log_debug(text)
            except Exception:
                print(text, flush=True)

    def _known_client_count(self, default=1):
        try:
            info = self._fl_context.get("methods_list", {}).get(self._method_name, {})
            return max(1, int(info.get("localclients_num", default)))
        except Exception:
            return max(1, int(default))

    def _infer_round_number(self, clients_models=None):
        if clients_models:
            for _, packet in clients_models:
                try:
                    return int(packet.get("diag", {})["round"])
                except Exception:
                    pass
        try:
            return int(self.server.round_number)
        except Exception:
            return -1

    def _layer_log_enabled(self, r):
        return bool(self.diag and (r <= 3 or (self.diag_layer_every > 0 and r % self.diag_layer_every == 0)))

    def _is_binary_param_tensor(self, name, tensor):
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor) or tensor.dim() < 2:
            return False
        lname = name.lower()
        if "weight" not in lname:
            return False
        skip = ("bn", "batchnorm", "norm", "gn", "running_mean", "running_var", "num_batches_tracked")
        return not any(t in lname for t in skip)

    @staticmethod
    def _is_aux_trainable(name, tensor):
        # VGG7 CenterOnlyBN has fixed gamma and a learnable beta. Do not
        # mistake running buffers for trainable parameters.
        lname = name.lower()
        return (torch.is_tensor(tensor) and tensor.ndim == 1 and
                lname.endswith('.bias') and
                ('bn' in lname or 'norm' in lname))

    def _fan_in(self, t):
        return max(1, int(t.shape[1]) * math.prod(t.shape[2:])) if t.dim() >= 2 else max(1, t.numel())

    # ------------------------------------------------------------------
    # Server-side Biprop state
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _init_server_state(self, state):
        if self.server_initialized:
            return

        wg = torch.Generator(device="cpu"); wg.manual_seed(int(self.weight_seed))
        sg = torch.Generator(device="cpu"); sg.manual_seed(int(self.score_seed))
        total = 0

        for key, t in state.items():
            if not torch.is_tensor(t) or not self._is_binary_param_tensor(key, t):
                continue
            shape = tuple(t.shape)
            fan_in = self._fan_in(t)
            W = torch.randn(shape, generator=wg, dtype=torch.float32) * math.sqrt(2.0 / fan_in)
            B = torch.where(W >= 0, torch.ones_like(W, dtype=torch.int8), -torch.ones_like(W, dtype=torch.int8))
            bound = 1.0 / math.sqrt(float(fan_in))
            S = torch.empty(shape, dtype=torch.float32); S.uniform_(-bound, bound, generator=sg)
            self.W_buffers[key] = W
            self.B_buffers[key] = B
            self.S_buffers[key] = S
            total += W.numel()

        self._renew_masks_and_gains()
        self._apply_effective_model(state)
        self.server_initialized = True
        self._log(
            f"INIT layers={len(self.W_buffers)} binary={total} pruning_ratio={self.pruning_ratio:.3f} "
            f"score_lr={self.score_lr:.3e} stc_keep={self.stc_keep_ratio:.4f} client_grad_steps={self.client_grad_steps}"
        )

    @torch.no_grad()
    def _mask_from_score(self, S):
        x = S.abs().view(-1)
        n = x.numel()
        k = max(1, min(n, int(math.ceil((1.0 - self.pruning_ratio) * n))))
        idx = torch.topk(x, k=k, largest=True, sorted=False).indices
        m = torch.zeros(n, dtype=torch.bool); m[idx] = True
        return m.view_as(S)

    @torch.no_grad()
    def _renew_masks_and_gains(self):
        for key, S in self.S_buffers.items():
            M = self._mask_from_score(S)
            W = self.W_buffers[key]
            self.M_buffers[key] = M
            self.alpha_buffers[key] = float(W[M].abs().mean().item()) if M.any() else 1.0

    @torch.no_grad()
    def _apply_effective_model(self, state):
        for key, B in self.B_buffers.items():
            if key not in state:
                continue
            target = state[key]
            eff = B.float() * self.M_buffers[key].float() * float(self.alpha_buffers[key])
            target.copy_(eff.to(device=target.device, dtype=target.dtype))

    # ------------------------------------------------------------------
    # Lossless wire codecs
    # ------------------------------------------------------------------
    @staticmethod
    def _rice_entry(sorted_indices, numel):
        k, blob, count, nbits = rice_encode(sorted_indices)
        return {"n": int(numel), "k": int(k), "count": int(count), "nbits": int(nbits), "bytes": blob}

    @staticmethod
    def _rice_indices(entry):
        return rice_decode(
            entry["k"], entry["bytes"], entry["count"],
            nbits=entry["nbits"], numel=entry["n"]
        )

    # ------------------------------------------------------------------
    # STC uplink
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _stc_compress(self, key, grad):
        g = grad.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
        if self.compression == "fp32":
            # Paper's main accuracy results use uncompressed score gradients.
            # Preserve all 32-bit values and do NOT apply ternarization or STC.
            self.error_feedback.pop(key, None)
            return {"codec": "fp32", "n": int(g.numel()), "values": g.clone()}

        residual = self.error_feedback.get(key)
        if residual is None or residual.numel() != g.numel():
            residual = torch.zeros_like(g)
        else:
            residual = residual.to(dtype=torch.float32, device="cpu").view(-1)
        corrected = g + residual
        n = corrected.numel()
        keep = max(1, min(n, int(math.ceil(self.stc_keep_ratio * n))))
        if not torch.isfinite(corrected).all().item():
            raise FloatingPointError("Nonfinite score gradient or residual")
        if corrected.abs().max().item() == 0.0:
            idx = torch.empty(0, dtype=torch.long)
            signs = torch.empty(0, dtype=torch.int8)
            mu = 0.0
            recon = torch.zeros_like(corrected)
        else:
            top = torch.topk(corrected.abs(), k=keep, largest=True, sorted=False).indices
            vals = corrected[top]
            mu = float(vals.abs().mean().item())
            signs0 = torch.where(vals >= 0, 1, -1).to(torch.int8)
            order = torch.argsort(top)
            idx = top[order].long()
            signs = signs0[order]
            recon = torch.zeros_like(corrected)
            recon[idx] = signs.float() * mu
        self.error_feedback[key] = corrected - recon
        positions = self._rice_entry(idx, n)
        return {
            "codec": "stc-rice1", "loc": positions,
            "signs": pack_signs(signs), "mu": mu,
        }

    @torch.no_grad()
    def _stc_decode(self, entry, n):
        if int(entry.get("n", entry.get("loc", {}).get("n", -1))) != int(n):
            raise ValueError("STC/FP32 tensor size mismatch")
        if entry["codec"] == "fp32":
            vals = entry["values"]
            if int(vals.numel()) != int(n):
                raise ValueError("Dense score-gradient tensor length mismatch")
            return vals.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if entry["codec"] != "stc-rice1":
            raise ValueError("Unknown uplink codec")
        ix = self._rice_indices(entry["loc"])
        signs = unpack_signs(entry["signs"], ix.size)
        result = torch.zeros(int(n), dtype=torch.float32)
        if ix.size:
            result[torch.from_numpy(ix)] = torch.from_numpy(signs.astype(np.float32)) * float(entry["mu"])
        return result

    # ------------------------------------------------------------------
    # BMC downlink
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _make_bmc_entry(self, key, full_participation):
        cur = self.M_buffers[key].view(-1)
        prev = self._last_broadcast_masks.get(key)
        if full_participation and prev is not None and prev.numel() == cur.numel():
            mode = "xor"
            idx = torch.nonzero(cur ^ prev.view(-1), as_tuple=False).view(-1)
        else:
            mode = "absolute"
            idx = torch.nonzero(cur, as_tuple=False).view(-1)
        return {"mode": mode, "loc": self._rice_entry(idx, cur.numel())}

    @torch.no_grad()
    def _apply_bmc_client(self, key, entry):
        loc = entry["loc"]
        n = int(loc["n"])
        idx = torch.from_numpy(self._rice_indices(loc))
        if entry["mode"] == "absolute":
            m = torch.zeros(n, dtype=torch.bool)
            if idx.numel():
                m[idx] = True
            self.client_M[key] = m
        elif entry["mode"] == "xor":
            old = self.client_M.get(key)
            if old is None or old.numel() != n:
                raise RuntimeError("Missing cached pruning mask for delta update: " + key)
            m = old.clone().view(-1)
            if idx.numel():
                m[idx] = ~m[idx]
            self.client_M[key] = m
        else:
            raise ValueError("Invalid BMC transmission mode")

    def _ensure_activation_ste(self, model):
        """Install the official activation backward on THIS client's modules.

        FedWork calls method.train() before model(inputs), so hooks are active
        from the first minibatch.  Avoid modifying Vgg7Bin.py, the module class,
        server evaluation, or any other method's client network.  The instance
        forward remains sign(x) with VGG7's +1 convention at exactly zero.
        """
        if self.activation_ste != "bireal" or model is None:
            return
        patched = 0
        already = 0
        for module in model.modules():
            if (module.__class__.__name__ != "BinaryInputActivation" or
                    not hasattr(module, "binary")):
                continue
            if getattr(module, "_biprune_bireal_installed", False):
                already += 1
                continue
            module.forward = types.MethodType(_biprune_binary_input_forward, module)
            module._biprune_bireal_installed = True
            patched += 1
        if patched and not self._activation_patch_logged:
            self._log("ACTIVATION_STE mode=bireal patched={} already={} forward=unchanged".format(
                patched, already))
            self._activation_patch_logged = True

    def train(self, client_train_dict):
        # Set the method-specific STE BEFORE the first forward/backward.
        self._ensure_activation_ste(client_train_dict.get("client_model"))
        # FedWork's ClientExp iterates the entire local DataLoader. BiPruneFL's
        # released protocol computes ONE minibatch gradient per participation.
        # Skip extra minibatches (rather than doing 49 unnecessary backwards).
        # IMPORTANT: EpochDone training accuracy/loss in this fast mode is NOT
        # representative; only global EvaluationAccuracy is meaningful.
        if (self.fast_local_step and self.client_grad_steps > 0 and
                self.score_grad_steps >= self.client_grad_steps):
            return 0.0, torch.tensor(0, device=client_train_dict["inputs"].device)
        return None

    # ------------------------------------------------------------------
    # Lossless sign(S) synchronization: one-bit bootstrap and sparse XOR
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _make_score_sign_entry(self, key):
        scores = self.S_buffers[key].view(-1)
        cur = scores >= 0
        zero = scores == 0
        old = self._last_broadcast_score_signs.get(key)
        old_zero = self._last_broadcast_score_zeros.get(key)
        if old is None or self.contributors_percent < 1.0:
            # Even PyTorch's float32 uniform_ can produce exactly-zero scores
            # across 13M positions. sign(0)=0, so synchronize zeros exactly.
            zix = torch.nonzero(zero, as_tuple=False).view(-1)
            return {"mode": "absolute", "n": int(cur.numel()),
                    "packed": pack_binary(torch.where(cur, 1, -1)),
                    "zero_loc": self._rice_entry(zix, cur.numel())}
        idx = torch.nonzero(cur ^ old.view(-1), as_tuple=False).view(-1)
        zix = torch.nonzero(zero ^ old_zero.view(-1), as_tuple=False).view(-1)
        return {"mode": "xor", "loc": self._rice_entry(idx, cur.numel()),
                "zero_xor_loc": self._rice_entry(zix, cur.numel())}

    @torch.no_grad()
    def _apply_score_sign_client(self, key, entry):
        if entry["mode"] == "absolute":
            n = int(entry["n"])
            nxt = unpack_binary(entry["packed"], n).reshape(-1)
            zix = torch.from_numpy(self._rice_indices(entry["zero_loc"]))
            if zix.numel():
                nxt[zix] = 0
            self.client_score_sign[key] = nxt
        elif entry["mode"] == "xor":
            old = self.client_score_sign.get(key)
            if old is None:
                raise RuntimeError("Missing previous pruning-score signs for " + key)
            old = old.view(-1)
            newpositive = old >= 0
            ix = torch.from_numpy(self._rice_indices(entry["loc"]))
            if ix.numel():
                newpositive[ix] = ~newpositive[ix]
            newzero = old == 0
            zix = torch.from_numpy(self._rice_indices(entry["zero_xor_loc"]))
            if zix.numel():
                newzero[zix] = ~newzero[zix]
            nxt = torch.where(newpositive, 1, -1).to(torch.int8)
            nxt[newzero] = 0
            self.client_score_sign[key] = nxt
        else:
            raise ValueError("Invalid score-sign payload mode")
        self._score_sign_device_cache.pop(key, None)

    @torch.no_grad()
    def _initialize_client_real_weights(self, base_model, current_model):
        # Same fixed W and same generator/iteration ordering as _init_server_state.
        # W is therefore reproducible from the public shared weight_seed;
        # no uncounted FP32 bootstrap tensor crosses the network.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.weight_seed))
        for key, blob in base_model.items():
            shape = tuple(current_model[key].shape)
            W = torch.randn(shape, generator=generator, dtype=torch.float32) * \
                math.sqrt(2.0 / self._fan_in(current_model[key]))
            expected_B = unpack_binary(blob, W.numel(), shape)
            assert torch.equal(W >= 0, expected_B >= 0), (key, "shared-seed weight mismatch")
            self.client_W[key] = W
            self._w_device_cache.pop(key, None)

    def _client_real_weight_on_device(self, key, device):
        cached = self._w_device_cache.get(key)
        if cached is None or cached.device != device:
            cached = self.client_W[key].to(device=device, dtype=torch.float32)
            self._w_device_cache[key] = cached
        return cached

    def _client_score_sign_on_device(self, key, device):
        cached = self._score_sign_device_cache.get(key)
        if cached is None or cached.device != device:
            cached = self.client_score_sign[key].to(device=device, dtype=torch.float32)
            self._score_sign_device_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Client score-gradient computation
    # ------------------------------------------------------------------
    def _reset_client_grad(self):
        self.score_grad_sum = {}
        self.aux_grad_sum = {}
        self.score_grad_steps = 0

    @torch.no_grad()
    def after_backward(self, client_train_dict):
        model = client_train_dict["client_model"]
        use_step = self.client_grad_steps == 0 or self.score_grad_steps < self.client_grad_steps

        if use_step:
            used = False
            for name, p in model.named_parameters():
                if p.grad is None or not self._is_binary_param_tensor(name, p.data):
                    continue
                B = self.client_B.get(name)
                if B is None:
                    raise RuntimeError(f"Missing binary base weight {name}")
                if name not in self.client_W or name not in self.client_score_sign:
                    raise RuntimeError("Missing fixed W or sign(S) for " + name)
                w = self._client_real_weight_on_device(name, p.grad.device).view_as(p.grad)
                ssign = self._client_score_sign_on_device(name, p.grad.device).view_as(p.grad)
                # Faithful to src/utils/conv_type.py GetQuantnet_binary.backward:
                #   gS = gW_eff * sign(W) * |W| * sign(S)
                #      = gW_eff * W * sign(S).
                # DO NOT substitute alpha*sign(W): that omits |W| and sign(S).
                gs = p.grad.detach() * w * ssign
                acc = self.score_grad_sum.get(name)
                if acc is None or acc.shape != gs.shape:
                    acc = torch.zeros_like(gs, dtype=torch.float32)
                acc.add_(gs.float())
                self.score_grad_sum[name] = acc
                used = True
            # The paper's public code forwards gradients of BN/bias parameters
            # as d_etc and updates them on the server alongside score tensors.
            for name, p in model.named_parameters():
                if (p.grad is None or
                        not self._is_aux_trainable(name, p.data)):
                    continue
                aux = self.aux_grad_sum.get(name)
                if aux is None or aux.shape != p.grad.shape:
                    aux = torch.zeros_like(p.grad, dtype=torch.float32)
                aux.add_(p.grad.detach().float())
                self.aux_grad_sum[name] = aux
            if used or self.aux_grad_sum:
                self.score_grad_steps += 1

        # BiPruneFL performs no client-side model/score update.
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        return False

    @torch.no_grad()
    def pack_client_model(self, raw_model, global_model, id):
        r = int(self._scheduled_local_round or (self._diag_client_round + 1))
        self._diag_client_round = r
        steps = max(1, self.score_grad_steps)
        comp = {}
        nselected = nscore = 0
        for key, B in self.client_B.items():
            n = B.numel(); nscore += n
            gs = self.score_grad_sum.get(key)
            grad = torch.zeros_like(B, dtype=torch.float32) if gs is None else gs.detach().cpu().float() / float(steps)
            e = self._stc_compress(key, grad)
            comp[key] = e
            nselected += int(e.get("loc", {}).get("count", 0))
        # Raw running-stat buffers are still needed for VGG7's evaluation.
        # Trainable BN beta is NOT copied as an absolute client value; the server
        # instead applies its gradient and sends the new global beta downlink.
        stats = {}
        for key, value in raw_model.items():
            if (torch.is_tensor(value) and
                    not self._is_binary_param_tensor(key, value) and
                    not self._is_aux_trainable(key, value)):
                stats[key] = value.detach().cpu()
        aux_grad = {key: g.detach().cpu().float().clone() / float(steps)
                    for key, g in self.aux_grad_sum.items()}
        packet = {
            "codec": "biprunefl-up-packed-v1",
            "compressed_score_grad": comp,
            "aux_grad_model": aux_grad,
            "stats_model": stats,
            "diag": {"round": r, "grad_steps": self.score_grad_steps, "stc_nnz": nselected, "score_numel": nscore},
        }
        actual_bytes = len(pickle.dumps(packet))
        self._log(
            "CLIENT_UP round={} id={} stored_grad_steps={} selected={} "
            "wire_bytes={} compression={} aux_grad_tensors={}".format(
                r, id, self.score_grad_steps, nselected, actual_bytes,
                self.compression, len(aux_grad))
        )
        self._reset_client_grad()
        return packet

    def unpack_client_model(self, packed_model):
        return packed_model

    # ------------------------------------------------------------------
    # Server score SGD and recycling
    # ------------------------------------------------------------------
    def _cosine_lr(self, r):
        T = max(1, int(self.num_of_rounds))
        if T <= 1: return float(self.score_lr)
        q = min(1.0, max(0.0, float(max(1, r) - 1) / float(T - 1)))
        return float(self.score_lr_min) + 0.5 * (float(self.score_lr) - float(self.score_lr_min)) * (1.0 + math.cos(math.pi * q))

    def _resolved_recycle_round(self):
        return int(self.recycle_round) if int(self.recycle_round) > 0 else max(1, int(self.num_of_rounds) // 2)

    @torch.no_grad()
    def _weight_recycle(self):
        total = 0
        if self.recycle_rate <= 0: return total
        for key, S in self.S_buffers.items():
            s = S.view(-1); W = self.W_buffers[key].view(-1); n = s.numel()
            k = max(0, min(int(math.floor(self.recycle_rate * n)), n // 2))
            if k == 0: continue
            # Released BiPruneFL recycles by |score| ranking, not signed S.
            high = torch.topk(s.abs(), k=k, largest=True, sorted=True).indices
            low = torch.topk(s.abs(), k=k, largest=False, sorted=True).indices
            W[low] = W[high].clone()
            self.W_buffers[key] = W.view_as(self.W_buffers[key])
            self.B_buffers[key] = torch.where(
                self.W_buffers[key] >= 0,
                torch.ones_like(self.W_buffers[key], dtype=torch.int8),
                -torch.ones_like(self.W_buffers[key], dtype=torch.int8),
            )
            total += k
        self._renew_masks_and_gains()
        return total

    @torch.no_grad()
    def aggregate(self, clients_models, global_model):
        if not clients_models: return global_model
        self._init_server_state(global_model)
        r = self._infer_round_number(clients_models)
        K = len(clients_models); self._last_k_clients = K
        lr = self._cosine_lr(r)
        stats_bits = nnz = 0
        score_grad_abs_sum = 0.0
        score_grad_numel = 0

        # Current controlled experiment has equal client sizes, so paper's |D_i|/|D|
        # weighting is exactly the arithmetic mean used below.
        for key, S in self.S_buffers.items():
            n = S.numel(); gsum = torch.zeros(n, dtype=torch.float32); got = 0
            for _, packet in clients_models:
                e = packet.get("compressed_score_grad", {}).get(key)
                if e is None: continue
                gsum.add_(self._stc_decode(e, n)); got += 1
                nnz += int(e.get("loc", {}).get("count", 0))
            if got:
                avg_grad = (gsum / float(got)).view_as(S)
                score_grad_abs_sum += float(avg_grad.abs().sum().item())
                score_grad_numel += int(avg_grad.numel())
                self.S_buffers[key].sub_(avg_grad, alpha=float(lr))

        # Server SGD for normalization beta; aggregate running-stat buffers.
        aux_grad_abs_sum = 0.0
        aux_grad_numel = 0
        for key, target in global_model.items():
            if not torch.is_tensor(target) or self._is_binary_param_tensor(key, target):
                continue
            if self._is_aux_trainable(key, target):
                grads = [p.get('aux_grad_model', {})[key]
                         for _, p in clients_models
                         if key in p.get('aux_grad_model', {})]
                if grads:
                    mean = torch.stack([x.detach().to(device=target.device,
                              dtype=torch.float32) for x in grads]).mean(0)
                    target.sub_(mean.to(target.dtype), alpha=float(lr))
                    aux_grad_abs_sum += float(mean.abs().sum().item())
                    aux_grad_numel += int(mean.numel())
                continue
            vals = [p.get("stats_model", {})[key] for _, p in clients_models if key in p.get("stats_model", {})]
            if not vals: continue
            if torch.is_floating_point(target):
                acc = torch.zeros_like(target, dtype=torch.float32, device=target.device)
                for x in vals: acc.add_(x.to(device=target.device, dtype=torch.float32))
                target.copy_((acc / len(vals)).to(target.dtype))
            else:
                v = vals[0].to(device=target.device, dtype=target.dtype)
                for x in vals[1:]: v = torch.maximum(v, x.to(device=target.device, dtype=target.dtype))
                target.copy_(v)

        for _, p in clients_models:
            stats_bits += sum(int(v.numel() * v.element_size() * 8)
                              for v in p.get("stats_model", {}).values())

        old = {k: m.clone() for k, m in self.M_buffers.items()}
        self._renew_masks_and_gains()
        changes = sum(int((self.M_buffers[k] ^ old[k]).sum().item()) for k in self.M_buffers)
        ntotal = sum(m.numel() for m in self.M_buffers.values())

        recycle_now = r == self._resolved_recycle_round() and r != self._last_recycle_round
        recycled = 0
        if recycle_now:
            recycled = self._weight_recycle()
            self._pending_recycle_broadcast = True
            self._last_recycle_round = r

        self._apply_effective_model(global_model)
        # Actual serialized wire bytes are profiled natively by FedWork.
        strict_up = sum(8 * len(pickle.dumps(p)) for _, p in clients_models)
        self.comm_uplink_bits_total += strict_up
        self._server_pack_cache_round = None; self._server_pack_cache_payload = None

        active = sum(int(m.sum().item()) for m in self.M_buffers.values())
        self._log(
            f"SERVER_AGG round={r} K={K} score_lr={lr:.6e} active_rate={active/max(1,ntotal):.6e} "
            f"mask_change_rate={changes/max(1,ntotal):.6e} stc_nnz_total={nnz} "
            f"score_grad_abs_mean={score_grad_abs_sum/max(1,score_grad_numel):.6e} "
            f"stats_Mbit_total={stats_bits/1e6:.6f} "
            f"bn_beta_grad_abs_mean={aux_grad_abs_sum/max(1,aux_grad_numel):.6e} "
            f"strict_uplink_Mbit_round={strict_up/1e6:.6f} strict_uplink_Gbit_cum={self.comm_uplink_bits_total/1e9:.6f} "
            f"recycle_now={int(recycle_now)} recycled={recycled}"
        )
        return global_model

    # ------------------------------------------------------------------
    # Server downlink
    # ------------------------------------------------------------------
    @torch.no_grad()
    def pack_server_model(self, raw_model):
        self._init_server_state(raw_model)
        try:
            r = int(self.server.round_number)
        except Exception:
            r = 0
        if self._server_pack_cache_round == r and self._server_pack_cache_payload is not None:
            return self._server_pack_cache_payload

        full = abs(self.contributors_percent - 1.0) < 1e-12
        first = len(self._last_broadcast_masks) == 0
        bmc = {key: self._make_bmc_entry(key, full) for key in self.M_buffers}
        score_sign_model = {key: self._make_score_sign_entry(key)
                            for key in self.S_buffers}
        alpha = {key: float(self.alpha_buffers[key]) for key in self.M_buffers}
        # Broadcast the latest BN beta and running statistics; these have been
        # updated on server and must not silently remain stale on clients.
        nonbinary = {key: value.detach().cpu().clone()
                     for key, value in raw_model.items()
                     if torch.is_tensor(value) and
                     not self._is_binary_param_tensor(key, value)}
        payload = {
            "codec": "biprunefl-down-packed-v1",
            "bmc_model": bmc, "score_sign_model": score_sign_model,
            "alpha_model": alpha,
            "nonbinary_model": nonbinary,
            "next_round": max(1, r + 1), "first_broadcast": first,
        }
        if first:
            # 1 bit per signed base weight, genuinely packed.
            payload["binary_base_model"] = {
                key: pack_binary(B) for key, B in self.B_buffers.items()
            }
        if self._pending_recycle_broadcast:
            # Include actual 32-bit recycled W, as required for strict protocol
            # accounting (and as sent in the supplied candidate implementation).
            payload["recycle_binary_base_model"] = {
                key: pack_binary(B) for key, B in self.B_buffers.items()
            }
            payload["recycle_real_weight_model"] = {
                key: W.detach().clone() for key, W in self.W_buffers.items()
            }
            self._pending_recycle_broadcast = False
        real_bytes = len(pickle.dumps(payload))
        sign_changes = sum(int(e["loc"]["count"])
                           for e in score_sign_model.values()
                           if e["mode"] == "xor")
        sign_bytes = sum((len(e["packed"]) if e["mode"] == "absolute"
                          else len(e["loc"]["bytes"]))
                         for e in score_sign_model.values())
        self._last_broadcast_masks = {k: m.clone() for k, m in self.M_buffers.items()}
        self._last_broadcast_score_signs = {
            k: (S >= 0).clone() for k, S in self.S_buffers.items()}
        self._last_broadcast_score_zeros = {
            k: (S == 0).clone() for k, S in self.S_buffers.items()}
        self.comm_downlink_strict_bits_total += real_bytes * 8
        self._log(
            "SERVER_DOWN round={} first={} masks={} serialized_bytes={} "
            "unicast_x{}_bytes={} score_sign_flips={} score_sign_payload_bytes={}".format(
                r, int(first), len(bmc), real_bytes,
                self._known_client_count(1), real_bytes * self._known_client_count(1),
                sign_changes, sign_bytes)
        )
        self._server_pack_cache_round = r
        self._server_pack_cache_payload = payload
        return payload

    # ------------------------------------------------------------------
    # Client downlink application
    # ------------------------------------------------------------------
    @torch.no_grad()
    def unpack_server_model(self, packed_model, current_model):
        if packed_model.get("codec") != "biprunefl-down-packed-v1":
            raise ValueError("Invalid BiPruneFL downlink packet")
        if packed_model.get("binary_base_model"):
            self._initialize_client_real_weights(
                packed_model["binary_base_model"], current_model)
        for key, blob in packed_model.get("binary_base_model", {}).items():
            if key not in current_model:
                raise ValueError("Unknown binary layer: " + key)
            self.client_B[key] = unpack_binary(blob, current_model[key].numel(), current_model[key].shape)
        for key, W in packed_model.get("recycle_real_weight_model", {}).items():
            self.client_W[key] = W.detach().to("cpu", dtype=torch.float32).clone()
            self._w_device_cache.pop(key, None)
        for key, blob in packed_model.get("recycle_binary_base_model", {}).items():
            if key not in current_model:
                raise ValueError("Unknown recycled binary layer: " + key)
            self.client_B[key] = unpack_binary(blob, current_model[key].numel(), current_model[key].shape)
        # Recycling changes W magnitudes: update them in the client too.
        for key, entry in packed_model.get("bmc_model", {}).items():
            self._apply_bmc_client(key, entry)
        for key, entry in packed_model.get("score_sign_model", {}).items():
            self._apply_score_sign_client(key, entry)
        for key, a in packed_model.get("alpha_model", {}).items():
            self.client_alpha[key] = float(a)
        for key, value in packed_model.get("nonbinary_model", {}).items():
            if key not in current_model:
                raise ValueError('Unknown nonbinary layer/buffer: ' + key)
            dst = current_model[key]
            dst.copy_(value.to(device=dst.device, dtype=dst.dtype))

        for key, B in self.client_B.items():
            if key not in current_model:
                continue
            M = self.client_M.get(key)
            if M is None or M.numel() != B.numel():
                raise RuntimeError("BiPruneFL binary mask/base mismatch: " + key)
            target = current_model[key]
            eff = (B.float().view(-1) * M.float().view(-1) * float(self.client_alpha[key])).view_as(target)
            target.copy_(eff.to(device=target.device, dtype=target.dtype))
        self._scheduled_local_round = int(packed_model.get("next_round", self._diag_client_round + 1))
        self._reset_client_grad()
        return current_model
