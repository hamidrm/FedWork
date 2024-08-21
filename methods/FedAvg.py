#FedAvg

import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.profiler import *

class FedAvg(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        self.contributors_percent = int(common.Common.get_param_in_args(extra_args, "contributors_percent", 100))
        self.num_of_nodes_contributor = 0
        self.round_num = 0

    def get_name(self):
        return "FedAvg"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):
        fedavg_fraction = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]
        for key in global_model.keys():
            torch_list_weights = torch.stack([clients_models[i][key].float() * fedavg_fraction[i] for i in range(len(clients_models))],0)
            global_model[key] = torch_list_weights.sum(0)
        
        StdD = self.calculate_sigma_l2_norm(clients_models, global_model)
        profiler().save_variable("StdD", StdD, self.round_num)
        logger.log_normal(f"===================================================")
        logger.log_normal(f"= Std.Dev. : {StdD}")
        logger.log_normal(f"===================================================")
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
        self.num_of_nodes_contributor = int((float(self.contributors_percent) / 100.0) * len(all_clients))
        return dict(random.sample(list(all_clients.items()), self.num_of_nodes_contributor))

    def select_clients_to_update(self, all_clients):
        return all_clients

    def pack_client_model(self, raw_model, global_model):
        return raw_model

    def unpack_client_model(self, packed_model):
        return packed_model
    
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
        
    def calculate_sigma_l2_norm(self, local_models, global_model):
        N = len(local_models)
        
        sum_squared_deviations = {}
        
        for param_name, global_param in global_model.items():
            sum_squared_deviations[param_name] = torch.zeros_like(global_param)

            for local_model in local_models:
                local_param = local_model[param_name]
                sum_squared_deviations[param_name] += (local_param - global_param) ** 2
            
            sum_squared_deviations[param_name] /= N
        
        std_devs = {param_name: torch.sqrt(squared_deviation) for param_name, squared_deviation in sum_squared_deviations.items()}
        
        l2_norm = torch.sqrt(sum(torch.sum(std_dev ** 2) for std_dev in std_devs.values()))
        
        return l2_norm.item()
