#SCAFFOLD

import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.profiler import *
import copy
from utils.common import Common

class SCAFFOLD(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        self.contributors_percent = int(common.Common.get_param_in_args(extra_args, "contributors_percent", 100))
        self.num_of_nodes_contributor = 0
        self.round_num = 0
        self.client_control_variate = None
        self.global_control_variate = None
        self.lr = 0
        self.K = 0
        self.N = 0 #Total number of nodes

    def initialize_control_variates(self, org_dict):
        control_variate = {}
        for key in org_dict.keys():
            control_variate[key] = torch.zeros_like(org_dict[key])
        return control_variate

#########################    
# Server Side
#########################

    def get_name(self):
        return "SCAFFOLD"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):
        #datasets_fraction = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]

        delta_global_model = copy.deepcopy(global_model)
        delta_c = copy.deepcopy(global_model)


        for key in global_model.keys():
            delta_global_model[key] = torch.zeros_like(global_model[key])
            delta_c[key] = torch.zeros_like(global_model[key])

        if self.global_control_variate is None:
            self.global_control_variate = self.initialize_control_variates(global_model)
        
        for client_model in clients_models:
            model = client_model["client_model"]
            client_control_variate = client_model["updated_client_control_variate"]
            for key in global_model.keys():
                if Common.is_trainable(model, key):
                    delta_global_model[key] += model[key] / self.num_of_nodes_contributor
                    delta_c[key] += client_control_variate[key] / self.num_of_nodes_contributor

        for key in global_model.keys():
            if Common.is_trainable(global_model, key):
                global_model[key] = global_model[key] + delta_global_model[key]
                self.global_control_variate[key] = self.global_control_variate[key] + (float(self.num_of_nodes_contributor) / float(self.N)) * delta_c[key]
            else:
                global_model[key] = clients_models[0]["sta"][key]
        self.round_num += 1


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
        self.N = len(all_clients)
        self.num_of_nodes_contributor = int((float(self.contributors_percent) / 100.0) * len(all_clients))
        return dict(random.sample(list(all_clients.items()), self.num_of_nodes_contributor))

    def select_clients_to_update(self, all_clients):
        return all_clients

    def unpack_client_model(self, packed_model):
        return packed_model
    
    def pack_server_model(self, raw_model):
        packed_model = {}
        packed_model["global_model"] = raw_model
        packed_model["global_control_variate"] = self.global_control_variate
        return packed_model
    
#########################    
# Client Side
#########################

    def pack_client_model(self, raw_model, global_model):
        packed_data = {}

        packed_data["client_model"] = {}
        packed_data["sta"] = {}
        packed_data["updated_client_control_variate"] = {}

        for key in global_model.keys():
            packed_data["client_model"][key] = raw_model[key] - global_model[key]

        updated_client_control_variate = {}

        #Option II
        with torch.no_grad():
            for key in self.client_control_variate.keys():
                if Common.is_trainable(global_model, key):
                    param, global_param = raw_model[key], global_model[key]
                    updated_client_control_variate[key] = (global_param - param) / (self.K * self.lr) - self.global_control_variate[key]
                    self.client_control_variate[key] = self.client_control_variate[key] + updated_client_control_variate[key]
                else:
                    packed_data["sta"][key] = raw_model[key]
        packed_data["updated_client_control_variate"] = updated_client_control_variate
        return packed_data

    def unpack_server_model(self, packed_model):
        raw_model = packed_model["global_model"]
        self.global_control_variate = packed_model["global_control_variate"]
        return raw_model
    
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        if num_of_received_model == self.num_of_nodes_contributor:
            return True
        else:
            return False
        
    def train_after_optimization(self, client_train_dict, epoch_num):
        client_model = client_train_dict["client_model"]
        global_model_state = client_train_dict["global_model_state"]
        lr = client_train_dict["lr"]
        self.lr = lr

        if self.client_control_variate is None:
            self.client_control_variate = self.initialize_control_variates(global_model_state)

        if self.global_control_variate is None:
            self.global_control_variate = self.initialize_control_variates(global_model_state)

        

        with torch.no_grad():
            for (name, param) in client_model.named_parameters():
                c = self.global_control_variate[name]
                ci = self.client_control_variate[name]
                param -= lr * (param.grad + (c - ci))

        self.K = epoch_num + 1
        return client_model