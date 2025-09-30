#FedAvgDP
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

class FedAvgDP(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.fedmia_attack = None
        self.lr = 0.01
        self.clipping_norm = self.get_arg(float, "clipping_norm", 1.0)
        self.noise_multiplie = self.get_arg(float, "noise_multiplie", 0.0)
        self.fedmia_stride = self.get_arg(int, "fedmia_stride", 10)
        
    def get_name(self):
        return "FedAvgDP"
    
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

    def clip_update(self, model, C) -> torch.Tensor:
        norm = model.norm(2)
        scale = min(1.0, C / (norm + 1e-12))
        return model * scale
    
    def aggregate(self, clients_models, global_model):
        delta_model_list = defaultdict(list)
        client_weight = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]
        
        clients_models_copy = copy.deepcopy(clients_models)

        for i, client_model in enumerate(clients_models):
            for key in client_model[1].keys():
                if Common.is_trainable(global_model, key):
                    delta_model_list[key].append(self.clip_update(client_model[1][key], self.clipping_norm))

        for key in delta_model_list:
            agg = torch.sum(torch.stack(delta_model_list[key]), dim=0)
            noise = torch.rand_like(delta_model_list[key]) * (self.noise_multiplier * self.clipping_norm)
            global_model[key] += (agg + noise) * client_weight[client_model[0]]
        

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
        return (raw_model - global_model)

    
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        return super().ready_to_aggregate(num_of_received_model)