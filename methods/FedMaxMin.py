#FedMaxMin

import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *


class FedMaxMin(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        self.contributors_percent = int(common.Common.get_param_in_args(extra_args, "contributors_percent", 100))
        self.num_of_nodes_contributor = 0

    def get_name(self):
        return "FedMaxMin"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):
        for key in global_model.keys():
            torch_list_weights = torch.stack([clients_models[i][key].float() for i in range(len(clients_models))],0)
        
            max_v, _ = torch.max(torch_list_weights, dim=0)
            min_v, _ = torch.min(torch_list_weights, dim=0)
            global_model[key] = (max_v + min_v) / 2


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