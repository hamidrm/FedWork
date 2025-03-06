#FedAvg

import torch
from core.FederatedLearningClass import *
import random
from utils.logger import *
from utils.profiler import *
from utils.common import Common

class FedAvg(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        self.contributors_percent = float(self.get_arg(int, "contributors_percent", 100)) / 100.0

    def get_name(self):
        return "FedAvg"

    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)