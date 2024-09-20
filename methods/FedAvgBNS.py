#FedAvg

import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.profiler import *

class BNSArch(nn.Module):
    def __init__(self, total_nodes):
        super(BNSArch, self).__init__()
        self.fc1 = nn.Linear(total_nodes, <<NumberOfHiddenNodes:integer>>)
        self.fc2 = nn.Linear(<<NumberOfHiddenNodes>>, <<NumberOfOutputNodes:integer>>)

    def forward(self, x):
        x = x.view(-1, <<NumberOfInputNodes>>)
        x = self.fc1(x)
        x = F.<<ActivationFunction:act_fn>>(x)
        x = self.fc2(x)
        return x

class FedAvgBNS(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        self.contributors_percent = int(common.Common.get_param_in_args(extra_args, "contributors_percent", 100))
        self.num_of_nodes_contributor = 0
        self.round_num = 0

    def get_name(self):
        return "FedAvgBNS"
    
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
        # Number of local models
        N = len(local_models)
        
        # Initialize a dictionary to store the sum of squared deviations for each parameter
        sum_squared_deviations = {}
        
        # Iterate over each parameter in the global model
        for param_name, global_param in global_model.items():
            # Initialize sum of squared deviations for this parameter
            sum_squared_deviations[param_name] = torch.zeros_like(global_param)
            
            # Accumulate squared deviations from the global parameter for this parameter across all local models
            for local_model in local_models:
                local_param = local_model[param_name]
                sum_squared_deviations[param_name] += (local_param - global_param) ** 2
            
            # Calculate the mean of squared deviations for this parameter
            sum_squared_deviations[param_name] /= N
        
        # Calculate the standard deviation for each parameter
        std_devs = {param_name: torch.sqrt(squared_deviation) for param_name, squared_deviation in sum_squared_deviations.items()}
        
        # Calculate the L2 norm of the standard deviations
        l2_norm = torch.sqrt(sum(torch.sum(std_dev ** 2) for std_dev in std_devs.values()))
        
        return l2_norm.item()