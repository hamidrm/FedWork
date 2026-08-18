# SparseTernaryCompression
import math
import struct
import torch
from collections import defaultdict

from core.FederatedLearningClass import *
from utils.logger import *
from utils.common import Common


class SparseTernaryCompression(FederatedLearningClass):
    """
    Sparse Ternary Compression (STC)-style FL baseline.

    This implementation follows the STC idea more closely by compressing the
    full flattened trainable model update, not each layer independently.

    Client sends:
        - one STC payload for the full trainable update
        - optional non-trainable/statistical tensors

    Server decodes:
        - trainable parameters as deltas
        - aggregation adds averaged deltas to the global model
    """

    # ---------------------------------------------------------------------
    # Rice / bit packing helpers
    # ---------------------------------------------------------------------

    def choose_rice_b(self, p: float) -> int:
        """
        Choose Rice parameter b using the STC-style golden-ratio rule.
        Rice uses M = 2^b.
        """
        p = min(max(float(p), 1e-12), 1.0 - 1e-12)

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        value = math.log(phi - 1.0) / math.log(1.0 - p)
        b = 1 + math.floor(math.log(value) / math.log(2.0))

        return max(0, int(b))

    def pack_bits(self, bits):
        """
        Pack a list of 0/1 bits into bytes.
        MSB-first packing.
        """
        out = bytearray()
        cur = 0
        count = 0

        for bit in bits:
            cur = (cur << 1) | int(bit)
            count += 1

            if count == 8:
                out.append(cur)
                cur = 0
                count = 0

        if count > 0:
            cur <<= (8 - count)
            out.append(cur)

        return bytes(out)

    def get_bit(self, data: bytes, bit_pos: int) -> int:
        """
        Read one bit from bytes, MSB-first.
        """
        byte = data[bit_pos // 8]
        shift = 7 - (bit_pos % 8)
        return (byte >> shift) & 1

    def rice_encode_integer(self, x: int, b: int):
        """
        Rice encode a non-negative integer x using M = 2^b.
        Code = unary quotient + b-bit remainder.
        """
        if x < 0:
            raise ValueError("Rice coding only supports non-negative integers.")

        M = 1 << b
        q = x // M
        r = x % M

        bits = []

        # Unary quotient: q ones followed by zero
        bits.extend([1] * q)
        bits.append(0)

        # b-bit remainder
        for shift in range(b - 1, -1, -1):
            bits.append((r >> shift) & 1)

        return bits

    def rice_decode_integer(self, data: bytes, bit_pos: int, b: int):
        """
        Decode one Rice-coded non-negative integer from data.

        Returns:
            value, new_bit_pos
        """
        q = 0

        # Unary quotient: read ones until first zero
        while self.get_bit(data, bit_pos) == 1:
            q += 1
            bit_pos += 1

        # Skip terminating zero
        bit_pos += 1

        # Read b-bit remainder
        r = 0
        for _ in range(b):
            r = (r << 1) | self.get_bit(data, bit_pos)
            bit_pos += 1

        value = (q << b) + r
        return value, bit_pos

    def pack_stc_indices_and_signs(self, topk_indices, selected_signs, num_elements):
        """
        Pack sorted index gaps using Rice coding and signs using 1 bit each.

        Important:
        Signs are reordered according to sorted indices.
        """
        k = int(topk_indices.numel())
        p = k / int(num_elements)
        b = self.choose_rice_b(p)

        sorted_indices, order = torch.sort(topk_indices.detach().cpu().long())
        sorted_signs = selected_signs.detach().cpu()[order]

        # Encode sorted index gaps
        gap_bits = []
        prev = -1

        for idx in sorted_indices.tolist():
            gap = int(idx) - prev - 1
            gap_bits.extend(self.rice_encode_integer(gap, b))
            prev = int(idx)

        index_payload = self.pack_bits(gap_bits)

        # Encode signs: positive -> 1, negative -> 0
        sign_bits = (sorted_signs > 0).to(torch.uint8).tolist()
        sign_payload = self.pack_bits(sign_bits)

        return index_payload, sign_payload, b

    # ---------------------------------------------------------------------
    # Flatten / unflatten helpers
    # ---------------------------------------------------------------------

    def flatten_trainable_model_delta(self, raw_model, global_model):
        """
        Flatten and concatenate all trainable parameter deltas:
            delta = raw_model - global_model

        Returns:
            flat_delta: one 1D tensor containing all trainable deltas
            metadata: information needed to reconstruct each tensor
        """
        flat_parts = []
        metadata = []
        offset = 0

        for key in raw_model.keys():
            if not Common.is_trainable(raw_model, key):
                continue

            delta = (raw_model[key].detach() - global_model[key].detach()).reshape(-1)

            numel = delta.numel()
            metadata.append({
                "key": key,
                "shape": tuple(raw_model[key].shape),
                "numel": int(numel),
                "start": int(offset),
                "end": int(offset + numel),
                "dtype": str(raw_model[key].dtype),
            })

            flat_parts.append(delta)
            offset += numel

        if len(flat_parts) == 0:
            return torch.empty(0), metadata

        return torch.cat(flat_parts), metadata

    def unflatten_trainable_delta(self, flat_delta, metadata, device=None):
        """
        Reconstruct trainable tensor deltas from one flat decoded vector.

        Returns:
            decoded_delta: dict[key] = decoded tensor delta
        """
        if device is None:
            device = flat_delta.device

        decoded_delta = {}

        for item in metadata:
            key = item["key"]
            start = item["start"]
            end = item["end"]
            shape = item["shape"]

            decoded_delta[key] = flat_delta[start:end].reshape(shape).to(device)

        return decoded_delta

    # ---------------------------------------------------------------------
    # STC flat compression/decompression
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def stc_compress_flat(self, flat_delta, error_buffer, keep_ratio=0.01):
        """
        STC compression over the full flattened trainable model update.

        Args:
            flat_delta: 1D tensor of all trainable parameter updates
            error_buffer: 1D residual buffer for this client
            keep_ratio: fraction of coordinates to keep, e.g., 0.01 means 1%

        Returns:
            compressed_flat: sparse ternary flat tensor, values in {-mu, 0, +mu}
            payload: packed bytes sent to the server
            new_error_buffer: residual buffer for next round
            info: communication statistics
        """
        if flat_delta.numel() == 0:
            return flat_delta, b"", error_buffer, {
                "num_elements": 0,
                "k": 0,
                "keep_ratio": keep_ratio,
                "rice_b": 0,
                "header_bytes": 0,
                "index_bytes": 0,
                "sign_bytes": 0,
                "payload_bytes": 0,
            }

        input_tensor = flat_delta + error_buffer

        num_elements = input_tensor.numel()

        # keep_ratio is a density, not a compression ratio.
        keep_ratio = min(max(float(keep_ratio), 1.0 / num_elements), 1.0)
        k = max(1, min(num_elements, int(num_elements * keep_ratio)))

        abs_tensor = input_tensor.abs()
        topk_values, topk_indices = torch.topk(abs_tensor, k, sorted=False)

        # STC ternary scale value, often written as mu
        mu = topk_values.mean()

        selected_elements = input_tensor[topk_indices]
        selected_signs = selected_elements.sign()

        # Just in case exact zeros are selected
        selected_signs[selected_signs == 0] = 1

        compressed_flat = torch.zeros_like(input_tensor)
        compressed_flat[topk_indices] = mu * selected_signs

        new_error_buffer = input_tensor - compressed_flat

        index_payload, sign_payload, b = self.pack_stc_indices_and_signs(
            topk_indices=topk_indices,
            selected_signs=selected_signs,
            num_elements=num_elements,
        )

        # Header:
        # k  : uint32, 4 bytes
        # mu : float32, 4 bytes
        # b  : uint8, 1 byte
        header = struct.pack(
            "<IfB",
            int(k),
            float(mu.detach().cpu().item()),
            int(b),
        )

        payload = header + index_payload + sign_payload

        info = {
            "num_elements": int(num_elements),
            "k": int(k),
            "keep_ratio": float(keep_ratio),
            "rice_b": int(b),
            "header_bytes": len(header),
            "index_bytes": len(index_payload),
            "sign_bytes": len(sign_payload),
            "payload_bytes": len(payload),
        }

        return (
            compressed_flat.detach(),
            payload,
            new_error_buffer.detach(),
            info,
        )

    @torch.no_grad()
    def stc_decompress_flat(self, payload: bytes, num_elements: int, device=None, dtype=torch.float32):
        """
        Server-side STC decompression.

        Args:
            payload: bytes produced by stc_compress_flat
            num_elements: length of the original flattened trainable update
            device: target device
            dtype: target dtype

        Returns:
            flat_delta: decoded sparse ternary flat tensor
        """
        if device is None:
            device = torch.device("cpu")

        if num_elements == 0:
            return torch.empty(0, device=device, dtype=dtype)

        header_size = struct.calcsize("<IfB")
        k, mu, b = struct.unpack("<IfB", payload[:header_size])

        body = payload[header_size:]

        # Decode k Rice-coded index gaps
        bit_pos = 0
        indices = []
        prev = -1

        for _ in range(k):
            gap, bit_pos = self.rice_decode_integer(body, bit_pos, b)
            idx = prev + gap + 1
            indices.append(idx)
            prev = idx

        # index_payload was byte-aligned by pack_bits, so signs start at next byte
        index_bytes_used = (bit_pos + 7) // 8
        sign_payload = body[index_bytes_used:]

        # Decode sign bits
        signs = []
        for i in range(k):
            bit = self.get_bit(sign_payload, i)
            signs.append(1.0 if bit == 1 else -1.0)

        flat_delta = torch.zeros(num_elements, device=device, dtype=dtype)

        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        signs_tensor = torch.tensor(signs, device=device, dtype=dtype)

        flat_delta[indices_tensor] = float(mu) * signs_tensor

        return flat_delta

    # ---------------------------------------------------------------------
    # Framework methods
    # ---------------------------------------------------------------------

    # Will be called by Server and Clients
    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.client_round_num = 0
        self.lr = 0.01

        # Each client object owns its own error buffer.
        # Server object will not use it.
        self.error_buffer = None

        # p is the keep ratio / density.
        # STC paper often uses aggressive values; 0.01 is a reasonable default.
        self.p = self.get_arg(float, "p", 0.01)

        self.contributors_percent = float(
            self.get_arg(float, "contributors_percent", 100.0)
        ) / 100.0

    # Will be called by Server
    def get_name(self):
        return "SparseTernaryCompression"

    # Will be called by Server
    def init_method(self, server):
        logger.log_normal(f"SparseTernaryCompression, Keep ratio: {self.p}")
        super().init_method(server)

    # Will be called by Server
    def aggregate(self, clients_models, global_model):
        """
        Aggregate decoded STC deltas.

        Important:
        unpack_client_model returns trainable parameters as deltas, not full
        client weights. Therefore, aggregation must add the averaged delta to
        the current global model.
        """
        counts = defaultdict(int)
        sums = {}

        for _, state in clients_models:
            for key, value in state.items():
                if not Common.is_trainable(global_model, key):
                    continue

                if key not in sums:
                    sums[key] = torch.zeros_like(global_model[key])

                sums[key] += value.to(global_model[key].device)
                counts[key] += 1

        for key, summed_delta in sums.items():
            mean_delta = summed_delta / counts[key]
            global_model[key] = global_model[key] + mean_delta

    # Will be called by Server
    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")

        if self.server.round_number != self.num_of_rounds:
            self.server.start_round(self.clients_epochs)
            return eval_loss, eval_accuracy

        logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
        return None

    # Will be called by Server
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        return super().ready_to_aggregate(num_of_received_model)

    def select_clients_to_train(self, all_clients):
        if self.contributors_percent != 1.0:
            return self.select_random_clients(all_clients, self.contributors_percent)

        return super().select_clients_to_train(all_clients)

    # ---------------------------------------------------------------------
    # Main client/server packing functions
    # ---------------------------------------------------------------------

    def pack_client_model(self, raw_model, global_model, id):
        """
        Client-side function.

        This follows STC more closely:
        one global compression over the full trainable model update,
        not layer-by-layer compression.

        Args:
            raw_model: client model state_dict after local training
            global_model: current global model state_dict before local training
            client_name: optional, unused because each client has its own object

        Returns:
            packet_to_send: one packed STC packet for the trainable model update
        """
        flat_delta, metadata = self.flatten_trainable_model_delta(raw_model, global_model)

        # Each client object owns its own error buffer.
        if (
            not hasattr(self, "error_buffer")
            or self.error_buffer is None
            or self.error_buffer.numel() != flat_delta.numel()
        ):
            self.error_buffer = torch.zeros_like(flat_delta.detach().cpu())

        error_buffer = self.error_buffer.to(flat_delta.device)

        compressed_flat, payload, new_error_buffer, info = self.stc_compress_flat(
            flat_delta=flat_delta,
            error_buffer=error_buffer,
            keep_ratio=self.p,
        )

        # Keep residual inside the client object for the next round
        self.error_buffer = new_error_buffer.detach().cpu()

        # Optional: non-trainable/statistical parameters.
        # Remove this block if your framework does not need these values.
        non_trainable = {}
        non_trainable_bytes = 0

        for key in raw_model.keys():
            if not Common.is_trainable(raw_model, key):
                tensor = raw_model[key].detach().cpu()
                non_trainable[key] = tensor
                non_trainable_bytes += tensor.numel() * tensor.element_size()

        packet_to_send = {
            "type": "stc_full_model",

            # Actual STC payload: header + Rice-coded index gaps + packed signs
            "payload": payload,

            # Metadata is needed by the simulation/server to reshape tensors.
            # It is not counted in payload_bytes because model structure is
            # assumed to be known by both sides.
            "metadata": metadata,
            "numel": int(flat_delta.numel()),

            # Communication accounting
            "payload_bytes": int(info["payload_bytes"]),
            "non_trainable_bytes": int(non_trainable_bytes),
            "total_packet_bytes": int(info["payload_bytes"] + non_trainable_bytes),

            # Optional raw non-trainable tensors
            "non_trainable": non_trainable,

            # Useful for logging/debugging
            "k": int(info["k"]),
            "rice_b": int(info["rice_b"]),
            "index_bytes": int(info["index_bytes"]),
            "sign_bytes": int(info["sign_bytes"]),
            "header_bytes": int(info["header_bytes"]),
        }

        return packet_to_send

    def unpack_client_model(self, packed_model, device=None):
        """
        Server-side function.

        Args:
            packed_model: packet produced by pack_client_model
            device: server device

        Returns:
            unpacked_model: dict of decoded tensors.
                - For trainable parameters: decoded STC delta tensors
                - For non-trainable parameters: raw tensors, if included
        """
        if device is None:
            device = torch.device("cpu")

        if packed_model["type"] != "stc_full_model":
            raise ValueError(f"Unknown packet type: {packed_model['type']}")

        flat_delta = self.stc_decompress_flat(
            payload=packed_model["payload"],
            num_elements=packed_model["numel"],
            device=device,
            dtype=torch.float32,
        )

        unpacked_model = self.unflatten_trainable_delta(
            flat_delta=flat_delta,
            metadata=packed_model["metadata"],
            device=device,
        )

        # Optional: add non-trainable tensors if your aggregation code expects them
        for key, tensor in packed_model.get("non_trainable", {}).items():
            unpacked_model[key] = tensor.to(device)

        return unpacked_model