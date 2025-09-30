#FedLP
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

class FedLP(FederatedLearningClass):
    #Will be called by Server and Clients
    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        self.client_round_num = 0
        self.lr = 0.01
        self.p  = self.get_arg(float, "p", 1.0)
        self.contributors_percent = float(self.get_arg(int, "contributors_percent", 100)) / 100.0
    #Will be called by Server
    def get_name(self):
        return "FedLP-Homo"
    #Will be called by Server
    def init_method(self, server):

        logger.log_normal(f"FedLP, Probability: {self.p}")
        super().init_method(server)

    #Will be called by Server
    def aggregate(self, clients_models, global_model):
        counts = defaultdict(int)
        sums = defaultdict(list)

        for _, state in clients_models:
            for k, v in state.items():
                if Common.is_trainable(global_model, k):
                    sums[k].append(v)
                    counts[k] += 1

        for k, vs in sums.items():
            global_model[k] = torch.sum(torch.stack(vs), dim=0) / counts[k]

    #Will be called by Server
    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != self.num_of_rounds:
            self.server.start_round(self.clients_epochs)

            return (eval_loss, eval_accuracy)
        else:
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None


    # build once (server or client) – map param name -> group name
    def make_layer_groups(self, model_state_dict):
        groups = {}
        for name in model_state_dict.keys():
            # heuristic: everything up to the last dot (e.g., 'layer1.0.conv1')
            prefix = ".".join(name.split(".")[:-1])
            groups.setdefault(prefix, []).append(name)
        return groups



    #Will be called by Clients after optimizing model in E epochs (raw_model)
    def pack_client_model(self, raw_model, global_model, id):
        self.client_round_num += 1
        groups = self.make_layer_groups(raw_model)
        torch.manual_seed(self.client_round_num * 10000 + id)
        tensor_layers = {}
        for g, keys in groups.items():
            keep = torch.rand(1).item() < self.p
            if keep:
                for k in keys:
                    if Common.is_trainable(global_model, k):
                        tensor_layers[k] = raw_model[k]
        return tensor_layers
    
    #Will be called by Server
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        return super().ready_to_aggregate(num_of_received_model)
    
    def select_clients_to_train(self, all_clients):
        if self.contributors_percent != 1.0:
            return self.select_random_clients(all_clients, self.contributors_percent)
        return super().select_clients_to_train(all_clients)