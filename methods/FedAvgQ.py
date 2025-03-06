import torch
from core.FederatedLearningClass import *
from utils.common import Common
from utils.quantization import *
import random
from utils.logger import *



class FedAvgQ(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.no_levels = self.get_arg(int, "levels", 8)
        self.q_method = self.get_arg(str, "quantization", "None")
        self.c_percent = self.get_arg(int, "contributors_percent", 100)

        self.quantization = QuantizationClass()

        if self.q_method == "None":
            self.quantization = NonQuantizer()
        elif self.q_method == "Randomized":
            self.quantization = RandomizedQuantizer(self.no_levels)
        elif self.q_method == "Uniform":
            self.quantization = UniformQuantizer(self.no_levels)
        elif self.q_method == "QSGD":
            self.quantization = QSGDQuantizer(self.no_levels)

    def get_name(self):
        return "FedAvgQ"

    def aggregate(self, clients_models, global_model):
        fedavg_fraction = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]
        for key in global_model.keys():
            torch_list_weights = torch.stack([(clients_models[i][1][key].float() + global_model[key]) * fedavg_fraction[clients_models[i][0]] for i in range(len(clients_models))],0)
            global_model[key] = torch_list_weights.sum(0)

    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.c_percent)

    def pack_client_model(self, raw_model, global_model):
        quantized_model = {}
        packet_to_send = {}
        scale = {}
        mins = {}
        for key in raw_model.keys():
            #Skip the statstical parameters
            if not Common.is_trainable(raw_model, key):
                quantized_tensor, mins[key], scale[key] = raw_model[key], 0, 0
                quantized_model[key] = quantized_tensor.to(torch.long)
            else:
                quantized_tensor, mins[key], scale[key] = self.quantization.quantize(raw_model[key] - global_model[key])

                if self.q_method != "None":
                    quantized_model[key] = quantized_tensor.to(torch.int8)      

        packet_to_send["tensors"] = quantized_model
        if self.q_method != "None":
            packet_to_send["scales"] = scale
            if self.q_method != "QSGD":
                packet_to_send["mins"] = mins
        return packet_to_send

    def unpack_client_model(self, packed_model):
        quantized_model = packed_model["tensors"]
        scale = {}
        mins = {}
        dequantized_model = {}

        if self.q_method != "None":
            scale = packed_model["scales"]
            
            if self.q_method != "QSGD":
                mins = packed_model["mins"]
                for key in quantized_model.keys():
                    if scale[key] == 0:
                        dequantized_model[key] = quantized_model[key]
                    else:
                        dequantized_model[key] = self.quantization.dequantize(quantized_model[key], mins[key], scale[key])
            else:
                for key in quantized_model.keys():
                    if scale[key] == 0:
                        dequantized_model[key] = quantized_model[key]
                    else:
                        dequantized_model[key] = self.quantization.dequantize(quantized_model[key], 0, scale[key])
        return dequantized_model
