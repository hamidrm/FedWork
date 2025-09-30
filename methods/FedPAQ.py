import torch
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.quantization import QSGDQuantizer
from utils.common import Common

class FedPAQ(FederatedLearningClass):
    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        
        self.num_levels = self.get_arg(int, "num_levels", 16)
        self.r_percent = self.get_arg(int, "r_percent", 80)

        self.quantizer = QSGDQuantizer(self.num_levels)
        self.contributors_percent = (float(self.r_percent) / 100.0)

    def get_name(self):
        return "FedPAQ"
    
    def aggregate(self, clients_models, global_model):
        for key in global_model.keys():
            if Common.is_trainable(global_model, key):
                torch_list_weights = torch.stack([(clients_models[i][1][key].float() + global_model[key]) for i in range(len(clients_models))], 0)
                global_model[key] = torch_list_weights.mean(0)
            else:
                global_model[key] = clients_models[0][1][key]

    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)

    def pack_client_model(self, raw_model, global_model, client_name):
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
        
