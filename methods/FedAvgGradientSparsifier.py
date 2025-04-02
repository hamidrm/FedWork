#FedAvg

import torch
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.profiler import *
from utils.common import Common
from utils.security import GradientSparsifier

class FedAvgGradientSparsifier(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        self.contributors_percent = float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        self.gradient_sparsifier = GradientSparsifier(self.get_arg(float, "gradient_sparsifier", 0))

    def get_name(self):
        return "FedAvgGradientSparsifier"

    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)
    

    def pack_client_model(self, raw_model, global_model):
        return self.gradient_sparsifier(raw_model, global_model)
