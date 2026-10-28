import math
import pickle

import numpy as np
import torch

from core.FederatedLearningClass import FederatedLearningClass
from utils.logger import logger


class BiFLBiML(FederatedLearningClass):
    """
    Repository-faithful port of the public FL-BNN ``binfullnew`` path.

    Reference implementation:
      https://github.com/yuzhiyang123/FL-BNN

    This class reproduces the executable public code path rather than replacing
    it with a corrected or reinterpreted version of the paper equations. The
    VGG7 architecture, dataset, and client count remain the common FedWork
    controlled benchmark.

    Public-code behavior reproduced here:
      * auxiliary real-valued BNN weights W in [-1,1], Sign(W) forward;
      * precom(): T = round(1.49 * W / max(abs(W))) in {-1,0,+1};
      * server arithmetic average of those ternary tensors;
      * for M=10 (proportion=0.1), the exact released fitted update:
            z = 1.1717 * log(mean_T * sign(W) + 0.9652) + 0.0998
            W = clamp(alpha * W * z, -1, 1)
      * no Eq.(17) conversion and no numerical ML solver;
      * fresh Adam state each communication round, matching the effective
        behavior of the released adjust_optimizer helper.

    For accurate profiling, the executed ternary uplink is losslessly packed at
    2 bits/value. The aggregated ternary sum has 2M+1 possible states and is
    losslessly packed at ceil(log2(2M+1)) bits/value (5 bits for M=10).
    Non-binary floating-point state is transmitted as FP32.
    """

    UPLINK_CODEC = "BiFLBiML-Repo-U2"
    INIT_CODEC = "BiFLBiML-Repo-I1"
    DOWNLINK_CODEC = "BiFLBiML-Repo-D5"

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.contributors_percent = (
            float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        )
        self.alpha = self.get_arg(float, "alpha", 1.75)
        self.diagnostics = bool(self.get_arg(int, "diagnostics", 1))
        self.diagnostic_every = max(1, self.get_arg(int, "diagnostic_every", 10))

        # Public repository VGG schedule hooks. Base LR is taken from <localclients>.
        self.lr1_round = self.get_arg(int, "lr1_round", 30)
        self.lr1 = self.get_arg(float, "lr1", 0.02)
        self.lr2_round = self.get_arg(int, "lr2_round", 60)
        self.lr2 = self.get_arg(float, "lr2", 0.01)

        if self.contributors_percent != 1.0:
            raise ValueError(
                "BiFL-BiML repository-faithful implementation currently requires "
                "contributors_percent=100. Partial participation changes the "
                "ML observation model and must be evaluated separately."
            )
        if self.alpha <= 0.0:
            raise ValueError("BiFL-BiML requires alpha > 0")

        self._validate_equal_dataset_weights()

        # Server-side aggregation cache used to construct the lossless downlink.
        self._have_server_aggregate = False
        self._last_ternary_sums = None        # flat np.ndarray in [-M, M]
        self._last_num_participants = None

        # Local epoch/round index used for the public VGG learning-rate schedule.
        self._local_epoch_index = 0

        # Diagnostics / logical communication accounting.
        self._server_round = 0
        self._logical_uplink_bits_total = 0
        self._logical_downlink_broadcast_bits_total = 0
        self._logical_downlink_egress_bits_total = 0
        self._downlink_accounted_token = None

        self._log(
            "initialized: "
            f"alpha={self.alpha}, contributors=100%, "
            f"base_lr={self.method_dict.get('learning_rate', 'pending')}"
        )

    # ------------------------------------------------------------------
    # Framework API / validation
    # ------------------------------------------------------------------
    def get_name(self):
        return self.method_name

    def _log(self, msg, level="info"):
        text = f"[{self.get_name()}][BiML] {msg}"
        try:
            if level == "warning":
                logger.log_warning(text)
            elif level == "error":
                logger.log_error(text)
            elif level == "debug":
                logger.log_debug(text)
            else:
                logger.log_info(text)
        except Exception:
            print(text, flush=True)

    def _validate_equal_dataset_weights(self):
        ws = [float(w) for w in self.datasets_weights]
        if len(ws) <= 1:
            return
        ref = ws[0]
        tol = max(1e-12, abs(ref) * 1e-7)
        if any(abs(w - ref) > tol for w in ws[1:]):
            raise ValueError(
                "BiFL-BiML compact ternary-sum downlink requires equal client dataset "
                "sizes. Set equal_client_sizes=True (or implement the unequal-"
                "size virtual-client extension from the paper before using this "
                "method)."
            )

    def select_clients_to_train(self, all_clients):
        # Full participation is required for this exact controlled port so the
        # proportion-dependent released coefficients remain well-defined.
        self.num_of_contributor_nodes = len(all_clients)
        return all_clients

    def select_clients_to_update(self, all_clients):
        return all_clients

    # ------------------------------------------------------------------
    # Model schema
    # ------------------------------------------------------------------
    @staticmethod
    def _is_binary_param_tensor(key: str, tensor: torch.Tensor) -> bool:
        return (
            torch.is_tensor(tensor)
            and torch.is_floating_point(tensor)
            and key.endswith(".weight")
            and tensor.ndim in (2, 4)
        )

    @classmethod
    def _schema(cls, state):
        binary = []
        fp = []
        for key, tensor in state.items():
            if not torch.is_tensor(tensor):
                continue
            n = int(tensor.numel())
            shape = tuple(tensor.shape)
            if cls._is_binary_param_tensor(key, tensor):
                binary.append((key, shape, n))
            elif torch.is_floating_point(tensor):
                fp.append((key, shape, n))
        return binary, fp

    @staticmethod
    def _total_numel(schema):
        return sum(x[2] for x in schema)

    # ------------------------------------------------------------------
    # Raw byte codecs
    # ------------------------------------------------------------------
    @staticmethod
    def _repo_ternarize_tensor(x: torch.Tensor) -> torch.Tensor:
        """Exact released precom quantizer used by binfullnew."""
        x = x.detach()
        maxtheta = torch.max(x.abs())
        if float(maxtheta.item()) == 0.0:
            return torch.zeros_like(x, dtype=torch.int8)
        return torch.round(x * (1.49 / maxtheta)).to(torch.int8)

    @classmethod
    def _pack_ternary_stream(cls, state, binary_schema) -> bytes:
        if not binary_schema:
            return b""
        codes = []
        for key, _, _ in binary_schema:
            ternary = cls._repo_ternarize_tensor(state[key]).reshape(-1).cpu()
            codes.append((ternary.to(torch.int16) + 1).to(torch.uint8))
        values = torch.cat(codes, dim=0).numpy().astype(np.uint8, copy=False)
        bits = np.empty((values.size, 2), dtype=np.uint8)
        bits[:, 0] = values & 1
        bits[:, 1] = (values >> 1) & 1
        return np.packbits(bits.reshape(-1), bitorder="little").tobytes()

    @staticmethod
    def _unpack_ternary_stream(data: bytes, total_numel: int) -> np.ndarray:
        total_numel = int(total_numel)
        expected = (2 * total_numel + 7) // 8
        if len(data) != expected:
            raise ValueError(f"Invalid BiFL ternary stream length: {len(data)} != {expected}")
        if total_numel == 0:
            return np.empty(0, dtype=np.int8)
        raw = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[:2*total_numel]
        bits = raw.reshape(total_numel, 2)
        codes = bits[:,0].astype(np.uint8) | (bits[:,1].astype(np.uint8) << 1)
        if np.any(codes > 2):
            raise ValueError("BiFL ternary stream contains reserved code 3")
        return (codes.astype(np.int8) - 1).astype(np.int8, copy=False)

    @staticmethod
    def _pack_fp32_stream(state, fp_schema) -> bytes:
        if not fp_schema:
            return b""
        chunks = []
        for key, _, _ in fp_schema:
            chunks.append(state[key].detach().reshape(-1).to(device="cpu",dtype=torch.float32).contiguous().numpy())
        return np.concatenate(chunks).astype("<f4", copy=False).tobytes(order="C")

    @staticmethod
    def _unpack_fp32_stream(data: bytes, total_numel: int) -> np.ndarray:
        expected = int(total_numel) * 4
        if len(data) != expected:
            raise ValueError(f"Invalid BiFL FP32 stream length: {len(data)} != {expected}")
        if total_numel == 0:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(data, dtype="<f4", count=total_numel).copy()

    @staticmethod
    def _sum_bit_width(num_clients: int) -> int:
        return max(1, int(math.ceil(math.log2(2 * int(num_clients) + 1))))

    @classmethod
    def _pack_sum_stream(cls, sums: np.ndarray, num_clients: int) -> bytes:
        sums = np.asarray(sums).reshape(-1)
        if sums.size == 0:
            return b""
        if sums.min(initial=0) < -num_clients or sums.max(initial=0) > num_clients:
            raise ValueError("BiFL ternary sum outside [-M,M]")
        b = cls._sum_bit_width(num_clients)
        values = (sums.astype(np.int32, copy=False) + int(num_clients)).astype(np.uint32,copy=False)
        chunk_values = 1_000_000 - (1_000_000 % 8)
        shifts = np.arange(b,dtype=np.uint32)
        out = bytearray()
        for start in range(0,values.size,chunk_values):
            chunk=values[start:start+chunk_values]
            bit_matrix=((chunk[:,None] >> shifts[None,:]) & 1).astype(np.uint8,copy=False)
            out.extend(np.packbits(bit_matrix.reshape(-1),bitorder="little").tobytes())
        return bytes(out)

    @classmethod
    def _unpack_sum_stream(cls, data: bytes, total_numel: int, num_clients: int) -> np.ndarray:
        total_numel=int(total_numel)
        b=cls._sum_bit_width(num_clients)
        expected=(total_numel*b+7)//8
        if len(data)!=expected:
            raise ValueError(f"Invalid BiFL ternary-sum stream length: {len(data)} != {expected}")
        if total_numel==0:
            return np.empty(0,dtype=np.int16)
        raw=np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder="little")[:total_numel*b]
        matrix=raw.reshape(total_numel,b).astype(np.uint32,copy=False)
        weights=(1 << np.arange(b,dtype=np.uint32))
        values=(matrix*weights[None,:]).sum(axis=1,dtype=np.uint32)
        if values.max(initial=0)>2*num_clients:
            raise ValueError("Decoded BiFL ternary sum out of range")
        signed=values.astype(np.int32)-int(num_clients)
        return signed.astype(np.int16 if num_clients<=32767 else np.int32)

    # ------------------------------------------------------------------
    # Local BNN training: use FedWork's fresh per-round optimizer
    # ------------------------------------------------------------------
    def _current_lr(self):
        base_lr=float(self.method_dict.get("learning_rate",0.05))
        e=int(self._local_epoch_index)
        if e>=self.lr2_round:
            return float(self.lr2)
        if e>=self.lr1_round:
            return float(self.lr1)
        return base_lr

    @torch.no_grad()
    def _clip_auxiliary_binary_weights(self, model):
        for name,p in model.named_parameters():
            if self._is_binary_param_tensor(name,p.data):
                p.clamp_(-1.0,1.0)

    def train(self, client_train_dict: dict):
        model=client_train_dict["client_model"]
        inputs=client_train_dict["inputs"]
        labels=client_train_dict["labels"]
        criterion=client_train_dict["criterion"]
        opt=client_train_dict["client_optimizer"]
        lr=self._current_lr()
        for group in opt.param_groups:
            group["lr"]=lr
        opt.zero_grad()
        outputs=model(inputs)
        loss=self.client_training_criterion(criterion,outputs,labels)
        loss.backward()
        opt.step()
        self._clip_auxiliary_binary_weights(model)
        correct=self.client_training_correctness(outputs,labels)
        return loss.item()*inputs.size(0),correct

    def train_after_optimization(self, client_train_dict: dict, epoch_num):
        self._local_epoch_index += 1
        return None

    # ------------------------------------------------------------------
    # Client -> server: released precom() ternary stream
    # ------------------------------------------------------------------
    @torch.no_grad()
    def pack_client_model(self, raw_model, global_model, id):
        binary_schema, fp_schema = self._schema(raw_model)
        binary_bytes = self._pack_ternary_stream(raw_model, binary_schema)
        fp_bytes = self._pack_fp32_stream(raw_model, fp_schema)

        # Compact top-level tuple keeps pickle overhead small and stable.
        return (self.UPLINK_CODEC, binary_bytes, fp_bytes)

    def unpack_client_model(self, packed_model):
        if (
            not isinstance(packed_model, tuple)
            or len(packed_model) != 3
            or packed_model[0] != self.UPLINK_CODEC
            or not isinstance(packed_model[1], (bytes, bytearray))
            or not isinstance(packed_model[2], (bytes, bytearray))
        ):
            raise ValueError("Invalid BiFL-BiML client payload")
        # Keep the wire representation compressed until aggregate().
        return packed_model

    # ------------------------------------------------------------------
    # Server aggregation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def aggregate(self, clients_models, global_model):
        if not clients_models:
            return
        self._server_round += 1
        M=len(clients_models)
        binary_schema,fp_schema=self._schema(global_model)
        n_binary=self._total_numel(binary_schema)
        n_fp=self._total_numel(fp_schema)
        sum_dtype=np.int16 if M<=32767 else np.int32
        ternary_sums=np.zeros(n_binary,dtype=sum_dtype)
        fp_sum=np.zeros(n_fp,dtype=np.float64)
        total_weight=0.0
        zeros=0
        for client_id,packet in clients_models:
            _,ternary_bytes,fp_bytes=packet
            ternary=self._unpack_ternary_stream(ternary_bytes,n_binary)
            ternary_sums += ternary.astype(sum_dtype,copy=False)
            zeros += int(np.count_nonzero(ternary==0))
            w=float(self.datasets_weights[client_id]); total_weight += w
            if n_fp:
                vals=self._unpack_fp32_stream(fp_bytes,n_fp)
                fp_sum += vals.astype(np.float64,copy=False)*w
        if total_weight<=0:
            raise RuntimeError("BiFL-BiML received zero total dataset weight")
        mean_flat=ternary_sums.astype(np.float32)/float(M)
        off=0
        for key,shape,n in binary_schema:
            target=global_model[key]; arr=mean_flat[off:off+n]
            global_model[key]=torch.from_numpy(arr.copy()).reshape(shape).to(device=target.device,dtype=target.dtype)
            off += n
        if n_fp:
            fp_mean=(fp_sum/total_weight).astype(np.float32); off=0
            for key,shape,n in fp_schema:
                target=global_model[key]; arr=fp_mean[off:off+n]
                global_model[key]=torch.from_numpy(arr.copy()).reshape(shape).to(device=target.device,dtype=target.dtype)
                off += n
        self._last_ternary_sums=ternary_sums
        self._last_num_participants=int(M)
        self._have_server_aggregate=True
        uplink_bits_this_round=M*(2*n_binary+32*n_fp)
        self._logical_uplink_bits_total += int(uplink_bits_this_round)
        if hasattr(self,"server"):
            self.server.save_var("BiFLBiMLLogicalUplinkBits",self._logical_uplink_bits_total)
        if self.diagnostics and (self._server_round<=3 or self._server_round%self.diagnostic_every==0):
            wire_bytes=sum(len(pickle.dumps(p[1])) for p in clients_models)
            zero_frac=float(zeros)/float(M*n_binary) if n_binary else 0.0
            mean_abs=float(np.abs(mean_flat).mean()) if n_binary else 0.0
            self._log(f"round={self._server_round} clients={M} binary_params={n_binary} fp_params={n_fp} "
                      f"uplink_logical_MiB={uplink_bits_this_round/8/(1024**2):.3f} "
                      f"uplink_pickle_MiB={wire_bytes/(1024**2):.3f} "
                      f"ternary_zero_frac={zero_frac:.4f} mean|ternary_mean|={mean_abs:.4f}")

    # ------------------------------------------------------------------
    # Server -> client
    # ------------------------------------------------------------------
    def _account_downlink(self, payload, n_binary, n_fp, init=False):
        """Store logical compact-code downlink counters once per broadcast."""
        if not hasattr(self, "server"):
            return

        round_no = int(getattr(self.server, "round_number", 0))
        token = (round_no, bool(init), self._have_server_aggregate)
        if token == self._downlink_accounted_token:
            return
        self._downlink_accounted_token = token

        num_clients = max(1, len(getattr(self.server, "clients", {})) or len(self.datasets_weights))
        if init:
            # Initial synchronization transmits the actual FP32 auxiliary model.
            bits_broadcast = 32 * (n_binary + n_fp)
        else:
            b = self._sum_bit_width(self._last_num_participants)
            bits_broadcast = b * n_binary + 32 * n_fp

        self._logical_downlink_broadcast_bits_total += int(bits_broadcast)
        self._logical_downlink_egress_bits_total += int(bits_broadcast * num_clients)

        self.server.save_var(
            "BiFLBiMLLogicalDownlinkBroadcastBits",
            self._logical_downlink_broadcast_bits_total,
        )
        self.server.save_var(
            "BiFLBiMLLogicalDownlinkEgressBits",
            self._logical_downlink_egress_bits_total,
        )

    @torch.no_grad()
    def pack_server_model(self, raw_model):
        binary_schema, fp_schema = self._schema(raw_model)
        n_binary = self._total_numel(binary_schema)
        n_fp = self._total_numel(fp_schema)
        fp_bytes = self._pack_fp32_stream(raw_model, fp_schema)

        if not self._have_server_aggregate:
            # First synchronization only: clients must receive the same auxiliary
            # real-valued initialization W, not merely Sign(W).
            binary_fp32 = self._pack_fp32_stream(raw_model, binary_schema)
            payload = (self.INIT_CODEC, binary_fp32, fp_bytes)
            self._account_downlink(payload, n_binary, n_fp, init=True)
            return payload

        if self._last_ternary_sums is None or self._last_num_participants is None:
            raise RuntimeError("BiFL-BiML server ternary-sum cache is missing")
        if len(self._last_ternary_sums) != n_binary:
            raise RuntimeError("BiFL-BiML server sum cache/model schema mismatch")

        count_bytes = self._pack_sum_stream(
            self._last_ternary_sums, self._last_num_participants
        )
        payload = (
            self.DOWNLINK_CODEC,
            int(self._last_num_participants),
            count_bytes,
            fp_bytes,
        )
        self._account_downlink(payload, n_binary, n_fp, init=False)
        return payload

    # ------------------------------------------------------------------
    # Exact public-repository binfullnew client update
    # ------------------------------------------------------------------
    @staticmethod
    def _repo_coefficients_for_equal_clients(M: int):
        table={
            100:(0.9899,1.3758,-0.0261),
            50:(0.9820,1.3301,-0.0086),
            200:(0.9955,1.4156,-0.0039),
            44:(0.9797,1.3212,-0.0352),
            178:(0.9943,1.4074,-0.0352),
            89:(0.9887,1.3683,-0.0231),
            10:(0.9652,1.1717,0.0998),
        }
        if int(M) not in table:
            raise ValueError(f"Released FL-BNN binfullnew has no fitted constants for M={M}; available: {sorted(table)}")
        return table[int(M)]

    @torch.no_grad()
    def _apply_repo_binfullnew(self,current_model,sums_flat,M,binary_schema):
        add_c,mul_c,bias_c=self._repo_coefficients_for_equal_clients(M)
        off=0; updated={}
        for key,shape,n in binary_schema:
            local=current_model[key]
            sums_np=sums_flat[off:off+n]
            mean_t=torch.from_numpy(sums_np.astype(np.float32,copy=False)).to(local.device).reshape(shape)/float(M)
            local_f=local.detach().float()
            local_sign = torch.sign(local_f)
            arg=mean_t*local_sign+float(add_c)
            if torch.any(arg<=0):
                raise RuntimeError(f"BiFL-BiML repo log-domain failure in {key}: min={float(arg.min().item()):.6g}")
            z=torch.log(arg)*float(mul_c)+float(bias_c)
            updated[key]=torch.clamp(local_f*z*float(self.alpha),-1.0,1.0).to(dtype=local.dtype)
            off += n
        return updated

    @torch.no_grad()
    def unpack_server_model(self, packed_model, current_model=None):
        if current_model is None:
            current_model={}
        model={key:value.detach().clone() for key,value in current_model.items()}
        binary_schema,fp_schema=self._schema(model)
        n_binary=self._total_numel(binary_schema); n_fp=self._total_numel(fp_schema)
        if not isinstance(packed_model,tuple) or not packed_model:
            return packed_model
        codec=packed_model[0]
        if codec==self.INIT_CODEC:
            if len(packed_model)!=3:
                raise ValueError("Invalid BiFL-BiML initialization packet")
            binary_vals=self._unpack_fp32_stream(packed_model[1],n_binary)
            fp_vals=self._unpack_fp32_stream(packed_model[2],n_fp)
            off=0
            for key,shape,n in binary_schema:
                target=model[key]; arr=binary_vals[off:off+n]
                model[key]=torch.from_numpy(arr.copy()).reshape(shape).to(device=target.device,dtype=target.dtype); off += n
            off=0
            for key,shape,n in fp_schema:
                target=model[key]; arr=fp_vals[off:off+n]
                model[key]=torch.from_numpy(arr.copy()).reshape(shape).to(device=target.device,dtype=target.dtype); off += n
            return model
        if codec!=self.DOWNLINK_CODEC or len(packed_model)!=4:
            raise ValueError("Invalid BiFL-BiML server payload")
        M=int(packed_model[1])
        sums=self._unpack_sum_stream(packed_model[2],n_binary,M)
        fp_vals=self._unpack_fp32_stream(packed_model[3],n_fp)
        updates=self._apply_repo_binfullnew(model,sums,M,binary_schema)
        for key,value in updates.items():
            model[key]=value
        off=0
        for key,shape,n in fp_schema:
            target=model[key]; arr=fp_vals[off:off+n]
            model[key]=torch.from_numpy(arr.copy()).reshape(shape).to(device=target.device,dtype=target.dtype); off += n
        return model
