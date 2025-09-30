#FedMaxMin

import torch
from core.FederatedLearningClass import *
import random
from utils.logger import *

class FedMaxMin(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        self.contributors_percent = self.get_arg(int, "contributors_percent", 100)

    def get_name(self):
        return "FedMaxMin"

    def aggregate(self, clients_models, global_model):
        for key in global_model.keys():
            torch_list_weights = torch.stack([clients_models[i][1][key].float() for i in range(len(clients_models))],0)
            max_v, _ = torch.max(torch_list_weights, dim=0)
            min_v, _ = torch.min(torch_list_weights, dim=0)
            global_model[key] = (max_v + min_v) / 2

    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)

