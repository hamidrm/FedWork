#FedALA
import pickle
import torch
from core.FederatedLearningClass import *
import random
import torch.nn as nn
from utils.logger import *
from utils.profiler import *
from utils.common import Common
from utils.quantization import RandomizedQuantizer
from utils.security.DataManipulation import MixUpDefense
from utils.security.FedMIA import *
from utils.security.FedALAQ import *
from utils.security.GradientSparsifier import *
from dataset.dataset import _make_eval_train_dataset_from_base
from torch.utils.data import ConcatDataset, DataLoader, Subset, RandomSampler, BatchSampler, SequentialSampler
import math
import os, pickle, torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, SequentialSampler, BatchSampler
from torchvision import datasets, transforms
from torch.utils.data import Subset as TorchSubset
from torch.utils.data import TensorDataset
from utils.security.PrivacyMiaUtils import PrivacyMiaUtils

class FedALAQ(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.fedmia_attack = None
        self.lr = 0.01
        self.lcr = self.get_arg(float, "lcr", 1.0)
        self.alpha = self.get_arg(float, "alpha", 0.0)
        self.ldp_noise_std = self.get_arg(float, "ldp_noise_std", 0)
        self.mixup_alpha = self.get_arg(float, "mixup_alpha", 0)
        self.gradient_sparsifier_ratio = self.get_arg(float, "gradient_sparsifier", 1.0)
        self.q_levels = self.get_arg(int, "randomized_quantization_levels", 0)
        self.fedmia_stride = self.get_arg(int, "fedmia_stride", 10)
        self.contributors_percent = float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        self.p_k_list = []
        self.fla = FedALAQDefense(self.lcr, self.alpha, self.ldp_noise_std)
        self.mixup = MixUpDefense(self.mixup_alpha)
        self.gradient_sparsifier = GradientSparsifier(self.gradient_sparsifier_ratio) if self.gradient_sparsifier_ratio != 1.0 else None
        self.quantization = RandomizedQuantizer(self.q_levels) if self.q_levels != 0 else None

    def get_name(self):
        return "FedALAQ"
    
    def init_method(self, server):

        separator = "-" * 55
        title = "Federated Adaptive Layer Aggregation"
        info1 = f"Layers Contribution Ratio: {self.lcr * 100:.2f}%, Alpha: {self.alpha}"
        info2 = f"LDP Noise Std.: {self.ldp_noise_std:.2f}, MixUp Alpha: {self.mixup_alpha}"
        info3 = f"Sparsifier Ratio: {self.gradient_sparsifier_ratio:.2f}, Quantization Levels: {self.q_levels}"
        logger.log_normal(separator)
        logger.log_normal(f"|{title.center(53)}|")
        logger.log_normal(separator)
        logger.log_normal(f"|{info1.center(53)}|")
        logger.log_normal(f"|{info2.center(53)}|")
        logger.log_normal(f"|{info3.center(53)}|")
        logger.log_normal(separator)
        
        dataset_loader_validation, dataset_loader_train = PrivacyMiaUtils.get_data_loaders(self.fl_context["dataset_train_list"], self.fl_context["seed"])
        self.fedmia_attack = FedMIA(dataset_loader_train, dataset_loader_validation, torch.optim.SGD, nn.CrossEntropyLoss)
        
        super().init_method(server)


    def aggregate(self, clients_models, global_model):
        
        clients_models_copy = copy.deepcopy(clients_models)
        for i, client_model in enumerate(clients_models_copy):
            for key in client_model[1].keys():
                if Common.is_trainable(global_model, key):
                    clients_models_copy[i][1][key] += global_model[key]

                    
        self.fla.aggregate(clients_models, global_model, self.datasets_weights, self.p_k_list)
        
        
        

        if self.round_num() % self.fedmia_stride == 0:
            target_model_id = 0
            
            res_total = PrivacyMiaUtils.FedMiaExec(self.fedmia_attack, global_model,
                                       clients_models_copy, target_model_id,
                                       self.method_dict["arch"], self.lr, self.platform)
            if res_total != None:
                logger.log_normal(f"FedMIA Attack on round {self.round_num()}: model id: {target_model_id}, {res_total}")
                profiler.save_variable("MIA_TPRS_0_1", res_total["tprs"]["0.1"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_02", res_total["tprs"]["0.02"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_01", res_total["tprs"]["0.01"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_001", res_total["tprs"]["0.001"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_0001", res_total["tprs"]["0.001"], self.round_num() - 1)
                profiler.save_variable("MIA_AUC", res_total["auc"], self.round_num() - 1)


    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != self.num_of_rounds:
            self.lr = 0.1 * (1 + math.cos(math.pi * self.server.round_number / self.num_of_rounds)) / 2 
            #self.server.start_round(self.clients_epochs, self.lr)
            self.server.start_round(self.clients_epochs)

            return (eval_loss, eval_accuracy)
        else:
            res = self.fedmia_attack.get_auc_metrics(self.platform)
            logger.log_normal(f"Final FedMIA Attack on {self.round_num()} epochs: {res}")
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None


    def pack_client_model(self, raw_model, global_model, id):
        raw_model, p_k = self.fla.build_packet(raw_model, global_model, id)
        #logger.log_debug(f"p_k:{p_k}")
        packet_to_send = {}
        packet_to_send["p_k"] = p_k
        if self.gradient_sparsifier is not None:
            raw_model = self.gradient_sparsifier.sparsify(raw_model, global_model)
            
            to_send = {}
            for key in raw_model.keys():
                if Common.is_trainable(raw_model, key):
                    to_send[key] = raw_model[key]

            packet_to_send["tensors"] = to_send
            return packet_to_send
        
        if self.quantization is not None:
            quantized_model = {}
            scale = {}
            mins = {}
            for key in raw_model.keys():
                #Skip the statstical parameters
                if not Common.is_trainable(raw_model, key):
                    quantized_tensor, mins[key], scale[key] = raw_model[key], 0, -1
                    quantized_model[key] = quantized_tensor.to(torch.long)
                else:
                    quantized_tensor, mins[key], scale[key] = self.quantization.quantize(raw_model[key] - global_model[key])
                    #if torch.any(quantized_tensor < 0) or torch.any(quantized_tensor > 255):
                    #    raise ValueError("Quantization values outside uint8 range detected!")
                    quantized_model[key] = quantized_tensor.to(torch.uint8)      

            packet_to_send["tensors"] = quantized_model
            packet_to_send["scales"] = scale
            packet_to_send["mins"] = mins
            
            return packet_to_send
        else:
            delta = {}
            for key in raw_model.keys():
                if Common.is_trainable(raw_model, key):
                    delta[key] = (raw_model[key] - global_model[key])
                else:
                    delta[key] = raw_model[key]
            packet_to_send["tensors"] = delta
            return packet_to_send

    def unpack_client_model(self, packed_model):

        p_k = packed_model["p_k"]
        self.p_k_list.append(p_k)
        
        model = packed_model["tensors"]
        if self.quantization is not None:
            quantized_model = packed_model["tensors"]
            scale = {}
            mins = {}
            dequantized_model = {}
            scale = packed_model["scales"]
            mins = packed_model["mins"]
            
            for key in quantized_model.keys():
                if scale[key] == -1:
                    dequantized_model[key] = quantized_model[key]
                else:
                    dequantized_model[key] = self.quantization.dequantize(quantized_model[key], mins[key], scale[key])# / p_k[key]

            return dequantized_model

        return model
    
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        return super().ready_to_aggregate(num_of_received_model)
    

    def select_clients_to_train(self, all_clients):
        if self.contributors_percent != 1.0:
            return self.select_random_clients(all_clients, self.contributors_percent)
        return super().select_clients_to_train(all_clients)
    
    def client_training_get_data(self, inputs, labels):
        inputs, self.labels_actual, labels, _ = self.mixup.get_data(inputs, labels)
        return inputs, labels

    def client_training_correctness(self, outputs, labels):
        return self.mixup.correctness(outputs, self.labels_actual, labels)
    
    def client_training_criterion(self, criterion_fn, outputs, labels):
        return self.mixup.criterion(criterion_fn, outputs, self.labels_actual, labels)