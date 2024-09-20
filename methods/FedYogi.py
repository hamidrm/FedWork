import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.profiler import *
from utils.common import Common

class FedYogi(FederatedLearningClass):
    def __init__(self, args=()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        self.contributors_percent = int(common.Common.get_param_in_args(extra_args, "contributors_percent", 100))
        self.beta1 = float(common.Common.get_param_in_args(extra_args, "beta1", 0.9))
        self.beta2 = float(common.Common.get_param_in_args(extra_args, "beta2", 0.99))
        self.epsilon = float(common.Common.get_param_in_args(extra_args, "epsilon", 1e-3))
        self.eta = float(common.Common.get_param_in_args(extra_args, "eta", 1e-2))
        self.m = None  # First moment
        self.v = None  # Second moment
        self.num_of_nodes_contributor = 0
        self.round_num = 0

    def initialize_moments(self, org_dict):
        first_moment = {}
        second_moment = {}
        for key in org_dict.keys():
            first_moment[key] = torch.zeros_like(org_dict[key])
            second_moment[key] = torch.zeros_like(org_dict[key])
        return first_moment, second_moment

    def get_name(self):
        return "FedYogi"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):
        # Initialize moments if not done already
        if self.m is None or self.v is None:
            self.m, self.v = self.initialize_moments(global_model)
            for key in global_model.keys():
                torch_list_weights = torch.stack([(clients_models[i][key].float() + global_model[key]) for i in range(len(clients_models))],0)
                global_model[key] = torch_list_weights.sum(0) / self.num_of_nodes_contributor 
            return
        
        # Initialize delta_global_model with zeros for accumulating gradients
        delta_global_model = {key: torch.zeros_like(global_model[key]) for key in global_model.keys()}

        # Accumulate gradients from all client models
        for model in clients_models:
            for key in global_model.keys():
                if Common.is_trainable(model, key):
                    delta_global_model[key] += model[key] / self.num_of_nodes_contributor

        # Update moments and the global model parameters
        for key in global_model.keys():
            if Common.is_trainable(global_model, key):
                # Update biased first moment estimate
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta_global_model[key]
                
                # Compute the squared gradient
                delta_v = delta_global_model[key] ** 2
                
                self.v[key] = self.v[key] - (1 - self.beta2) * delta_v * torch.sign(self.v[key] - delta_v)

                # Prevent potential division by zero by ensuring v is always positive
                adjusted_v = torch.sqrt(self.v[key].abs()) + self.epsilon
                
                # Update global model parameters using the adaptive learning rate
                global_model[key] += self.eta * self.m[key] / adjusted_v
            else:
                # Directly use the first client's parameters for non-trainable parameters
                global_model[key] = clients_models[0][key]

        # Increment round number for the next aggregation
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
        for key in raw_model.keys():
            if Common.is_trainable(global_model, key):
                raw_model[key] = raw_model[key] - global_model[key]

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
