import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.quantization import QSGDQuantizer

class FedPAQ(FederatedLearningClass):
    def __init__(self, args=()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        num_levels = int(common.Common.get_param_in_args(extra_args, "num_levels", 16))
        r_percent = int(common.Common.get_param_in_args(extra_args, "r_percent", 80))
        self.quantizer = QSGDQuantizer(num_levels)
        self.num_of_nodes_contributor = 0
        self.contributors_percent = (float(r_percent) / 100.0)

    def get_name(self):
        return "FedPAQ"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):
        #fedavg_fraction = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]
        for key in global_model.keys():
            torch_list_weights = torch.stack([(clients_models[i][key].float() + global_model[key]) for i in range(len(clients_models))], 0)
            global_model[key] = torch_list_weights.mean(0)

    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != self.num_of_rounds:
            self.server.start_round(self.clients_epochs, [100, 200], 0.0001)
            return (eval_loss, eval_accuracy)
        else:
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None

    def select_clients_to_train(self, all_clients):
        self.num_of_nodes_contributor = int(float(self.contributors_percent) * len(all_clients))
        return dict(random.sample(list(all_clients.items()), self.num_of_nodes_contributor))

    def select_clients_to_update(self, all_clients):
        return all_clients

    def pack_client_model(self, raw_model, global_model):
        quantized_model = {}
        packet_to_send = {}
        scale = {}
        for key in raw_model.keys():
            if raw_model[key].dtype == torch.long or ('running_var' in key) or ('running_mean' in key):
                quantized_tensor, scale[key] = raw_model[key], 0
                quantized_model[key] = quantized_tensor.to(torch.long)
            else:
                quantized_tensor, scale[key] = self.quantizer.quantize(raw_model[key] - global_model[key])
                quantized_model[key] = quantized_tensor.to(torch.int8)      

        packet_to_send["tensors"] = quantized_model
        packet_to_send["scales"] = scale
        return packet_to_send

    def unpack_client_model(self, packed_model):
        quantized_model = packed_model["tensors"]
        scale = packed_model["scales"]
        dequantized_model = {}

        for key in quantized_model.keys():
            if scale[key] == 0:
                dequantized_model[key] = quantized_model[key]
            else:
                dequantized_model[key] = self.quantizer.dequantize(quantized_model[key], scale[key])
        
        return dequantized_model

    def pack_server_model(self, raw_model):
        return raw_model

    def unpack_server_model(self, packed_model):
        return packed_model
    
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        if num_of_received_model == self.num_of_nodes_contributor:
            return True
        else:
            return False

        
