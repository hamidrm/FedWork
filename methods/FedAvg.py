#FedAvg

import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *

num_of_nodes_contributor = 20

class FedAvg(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights = args

    def get_name(self):
        return "FedAvg"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):
        global_dict = global_model.state_dict()
        fedavg_fraction = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]
        for key in global_dict.keys():
            torch_list_weights = torch.stack([clients_models[i][key].float() * fedavg_fraction[i] for i in range(len(clients_models))],0)
            global_dict[key] = torch_list_weights.sum(0)
        global_model.load_state_dict(global_dict)

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
        return all_clients

    def select_clients_to_update(self, all_clients):
        return all_clients

    def pack_client_model(self, raw_model):
        return raw_model

    def unpack_client_model(self, packed_model):
        return packed_model
    
    def pack_server_model(self, raw_model):
        return raw_model

    def unpack_server_model(self, packed_model):
        return packed_model
    
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        if num_of_received_model == num_of_nodes_contributor:
            return True
        else:
            return False