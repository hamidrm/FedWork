#FedAvg

import torch
import torch.nn as nn
from FederatedLearningClass import *
import random
from utils.logger import *

num_of_nodes_contributor = 20

class FedAvg(FederatedLearningClass):

    def __init__(self):
        super().__init__()
        super().set_hyperparameters(0.0001, 0.9, 1e-7)

    def get_name(self):
        return "FedAvg"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):
        global_dict = global_model.state_dict()
        clients_weights = self.args
        fedavg_fraction = [clients_weights[i] for i in range(len(clients_weights))]
        for key in global_dict.keys():
            torch_list_weights = torch.stack([clients_models[i][key].float() * fedavg_fraction[i] for i in range(len(clients_models))],0)
            global_dict[key] = torch_list_weights.sum(0)
        global_model.load_state_dict(global_dict)

    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != 100:
            self.server.start_round(5, [100, 200], 0.0001)
        else:
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
        return eval_loss, eval_accuracy

    def select_clients_to_train(self, all_clients):
        #selected_clients = dict(random.sample(list(all_clients.items()), num_of_nodes_contributor))
        #logger.log_normal(f"Selected Clients: {selected_clients.keys()}")
        return all_clients

    def select_clients_to_update(self, all_clients):
        return all_clients

    def pack_client_model(self, raw_model):
        return raw_model

    def unpack_client_model(self, packed_model):
        return packed_model
    
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        if num_of_received_model == num_of_nodes_contributor:
            return True
        else:
            return False