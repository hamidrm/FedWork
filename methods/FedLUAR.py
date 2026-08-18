import math
import torch

from core.FederatedLearningClass import *
from utils.logger import *
from utils.common import Common


class FedLUAR(FederatedLearningClass):
    """
    FedLUAR implementation.

    - Server sends global model and R_t.
    - Clients skip recyclable tensor keys in R_t.
    - Clients send deltas for non-recycled tensors.
    - Server aggregates fresh deltas.
    - Server reuses previous global delta for recycled tensor keys.
    - Server selects R_{t+1} using weighted random sampling.
    """

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.lcr = self.get_arg(float, "lcr", 0.3)
        self.eps = self.get_arg(float, "eps", 1e-12)

        self.contributors_percent = float(
            self.get_arg(float, "contributors_percent", 100.0)
        ) / 100.0

        # R_t: list of recycled tensor keys, not layer names
        self.r_list = []

        # Previous global aggregated delta per tensor key
        self.last_global_delta = {}

        # Optional, useful for later inspection
        self.layer_scores = {}

    def get_name(self):
        return "FedLUAR"

    def init_method(self, server=None):
        logger.log_normal(f"FedLUAR, lcr={self.lcr}")
        super().init_method(server)

    def select_clients_to_train(self, all_clients):
        if self.contributors_percent != 1.0:
            return self.select_random_clients(all_clients, self.contributors_percent)
        return super().select_clients_to_train(all_clients)

    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        return super().ready_to_aggregate(num_of_received_model)

    def start_training(self):
        logger.log_normal("===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()

        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(
            f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}"
        )

        if self.server.round_number != self.num_of_rounds:
            self.server.start_round(self.clients_epochs)
            return eval_loss, eval_accuracy

        logger.log_normal(
            f"Training done! last global model accuracy is: {eval_accuracy}"
        )
        return None

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def is_recyclable_key(self, model, key):
        """
        Follow the official FedLUAR-style behavior:
        only trainable tensors with dimension > 1 are recyclable.

        Therefore, bias, BN/GN/LN scale/bias, and other 1D tensors
        are always freshly uploaded and aggregated.
        """
        return Common.is_trainable(model, key) and model[key].dim() > 1

    def get_layer_name(self, key: str) -> str:
        if "." not in key:
            return key
        return ".".join(key.rsplit(".", 1)[:-1])

    def get_trainable_keys(self, model):
        return [
            key for key in model.keys()
            if Common.is_trainable(model, key)
        ]

    def get_recyclable_keys(self, model):
        return [
            key for key in model.keys()
            if self.is_recyclable_key(model, key)
        ]

    # ---------------------------------------------------------
    # Server -> Client
    # ---------------------------------------------------------

    def pack_server_model(self, raw_model):
        return {
            "global_model": raw_model,
            "recycle_layers": self.r_list,
        }

    def unpack_server_model(self, packed_model):
        self.r_list = packed_model["recycle_layers"]
        return packed_model["global_model"]

    # ---------------------------------------------------------
    # Client -> Server
    # ---------------------------------------------------------

    def pack_client_model(self, raw_model, global_model, id):
        packet_to_send = {}

        sent_trainable = 0
        skipped_trainable = 0

        for key in raw_model.keys():

            # Keep framework convention:
            # running_mean, running_var, and torch.long tensors are sent directly.
            if not Common.is_trainable(raw_model, key):
                packet_to_send[key] = raw_model[key]
                continue

            # Only recyclable tensor keys can be skipped.
            # 1D tensors are never skipped.
            if key in self.r_list and self.is_recyclable_key(raw_model, key):
                skipped_trainable += 1
                continue

            # Send delta
            packet_to_send[key] = raw_model[key] - global_model[key]
            sent_trainable += 1

        logger.log_normal(
            f"FedLUAR pack_client_model: sent_trainable={sent_trainable}, "
            f"skipped_trainable={skipped_trainable}, "
            f"recycled_keys={len(self.r_list)}"
        )

        return packet_to_send

    def unpack_client_model(self, packed_model):
        return packed_model

    # ---------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------

    @torch.no_grad()
    def aggregate(self, clients_models, global_model):

        trainable_keys = self.get_trainable_keys(global_model)
        recyclable_keys = self.get_recyclable_keys(global_model)

        # First round: no previous delta exists
        if len(self.last_global_delta) == 0:
            self.r_list = []

        logger.log_normal(
            f"FedLUAR aggregate: clients={len(clients_models)}, "
            f"trainable_tensors={len(trainable_keys)}, "
            f"recyclable_tensors={len(recyclable_keys)}, "
            f"recycled_keys={len(self.r_list)}"
        )

        old_global_model = {
            key: global_model[key].detach().clone()
            for key in trainable_keys
        }

        delta_dict = {}

        # -----------------------------------------------------
        # 1. Compute global delta for each trainable tensor
        # -----------------------------------------------------

        for key in trainable_keys:

            # Only recyclable tensor keys can use previous recycled delta.
            # 1D trainable tensors are always freshly aggregated.
            if key in self.r_list and self.is_recyclable_key(global_model, key):
                if key in self.last_global_delta:
                    global_delta = self.last_global_delta[key].to(
                        global_model[key].device
                    ).float()
                else:
                    global_delta = torch.zeros_like(global_model[key]).float()

            else:
                weighted_sum = torch.zeros_like(global_model[key]).float()
                total_weight = 0.0

                for client_id, client_model in clients_models:
                    if key not in client_model:
                        continue

                    client_weight = self.datasets_weights[client_id]
                    client_delta = client_model[key].to(
                        global_model[key].device
                    ).float()

                    weighted_sum += client_delta * client_weight
                    total_weight += client_weight

                if total_weight > 0:
                    global_delta = weighted_sum / total_weight
                else:
                    if key in self.last_global_delta:
                        global_delta = self.last_global_delta[key].to(
                            global_model[key].device
                        ).float()
                    else:
                        global_delta = torch.zeros_like(global_model[key]).float()

            delta_dict[key] = global_delta.detach().clone()

        # -----------------------------------------------------
        # 2. Apply global update
        # -----------------------------------------------------

        for key in trainable_keys:
            global_model[key] = (
                global_model[key]
                + delta_dict[key].to(global_model[key].device)
            )

        # Store aggregated global delta for future recycling
        self.last_global_delta = {
            key: delta_dict[key].detach().cpu()
            for key in trainable_keys
        }

        # -----------------------------------------------------
        # 3. Compute scores only for recyclable tensor keys
        # -----------------------------------------------------

        scores = {}
        inv_weights = []

        for key in recyclable_keys:
            score = torch.norm(delta_dict[key].float(), p=2) / (
                torch.norm(old_global_model[key].float(), p=2) + self.eps
            )

            score_value = float(score.item())
            scores[key] = score_value

            # FedLUAR: lower score -> higher recycling probability
            inv_weights.append(1.0 / (score_value + self.eps))

        self.layer_scores = scores

        # -----------------------------------------------------
        # 4. Select R_{t+1} using weighted random sampling
        # -----------------------------------------------------

        self.r_list = []

        num_recyclable = len(recyclable_keys)

        if num_recyclable == 0:
            return

        lcr = min(max(float(self.lcr), 0.0), 1.0)

        num_upload = max(1, int(math.ceil(lcr * num_recyclable)))
        num_upload = min(num_upload, num_recyclable)

        num_recycle = num_recyclable - num_upload

        if num_recycle <= 0:
            logger.log_normal(
                f"FedLUAR aggregate done: upload_recyclable={num_recyclable}/{num_recyclable}, "
                f"recycle_keys=0/{num_recyclable}"
            )
            return

        probs_tensor = torch.tensor(inv_weights, dtype=torch.float32)

        if not torch.isfinite(probs_tensor).all() or probs_tensor.sum() <= 0:
            # Only numerical safety fallback. Not a deterministic variant.
            probs_tensor = torch.ones(num_recyclable, dtype=torch.float32)

        probs_tensor = probs_tensor / probs_tensor.sum()

        selected_indices = torch.multinomial(
            probs_tensor,
            num_samples=num_recycle,
            replacement=False
        ).tolist()

        self.r_list = [recyclable_keys[i] for i in selected_indices]

        logger.log_normal(
            f"FedLUAR aggregate done: upload_recyclable={num_upload}/{num_recyclable}, "
            f"recycle_keys={len(self.r_list)}/{num_recyclable}"
        )