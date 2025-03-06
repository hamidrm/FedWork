import torch
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.profiler import *
from utils.common import Common

class FedYogi(FederatedLearningClass):
    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        
        self.contributors_percent = self.get_arg(int, "contributors_percent", 80)
        self.beta1 = self.get_arg(float, "beta1", 0.9)
        self.beta2 = self.get_arg(float, "beta2", 0.99)
        self.epsilon = self.get_arg(float, "epsilon", 1e-3)
        self.eta = self.get_arg(float, "eta", 1e-2)

        self.m = None  # First moment
        self.v = None  # Second moment


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
                torch_list_weights = torch.stack([(clients_models[i][1][key].float() + global_model[key]) for i in range(len(clients_models))],0)
                global_model[key] = torch_list_weights.sum(0) / self.num_of_nodes_contributor 
            return
        
        # Initialize delta_global_model with zeros for accumulating gradients
        delta_global_model = {key: torch.zeros_like(global_model[key]) for key in global_model.keys()}

        # Accumulate gradients from all client models
        for model in clients_models:
            for key in global_model.keys():
                if Common.is_trainable(model[1], key):
                    delta_global_model[key] += model[1][key] / self.num_of_nodes_contributor

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
                global_model[key] = clients_models[0][1][key]

    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)

    def pack_client_model(self, raw_model, global_model):
        for key in raw_model.keys():
            if Common.is_trainable(global_model, key):
                raw_model[key] = raw_model[key] - global_model[key]

        return raw_model