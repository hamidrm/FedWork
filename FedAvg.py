#FedAvg

import torch
import torch.nn as nn
from FederatedLearningClass import *


class FedAvg(FederatedLearningClass):

    def __init__(self):
        super().set_hyperparameters(0.0001, 0.9, 1e-7)

    def get_name(self):
        return "FedAvg"
    
    def aggregate(self, clients_train_ds, clients_model, global_model):
        global_dict = global_model.state_dict()
        total_ds_length = 0
        for dataloader in clients_train_ds:
            total_ds_length += len(dataloader.dataset)
        fedavg_fraction = [((float(len(clients_train_ds[i]))) / total_ds_length) for i in range(len(clients_model))]
        for key in global_dict.keys():
            torch_list_weights = torch.stack([clients_model[i].state_dict()[key].float() * fedavg_fraction[i] for i in range(len(clients_model))],0)
            global_dict[key] = torch_list_weights.sum(0)
        global_model.load_state_dict(global_dict)

    def select_clients(self, all_clients, number):
        return dict(list(all_clients.items())[:number])
    
    def pack_client_model(self, raw_model):
        pass


    def unpack_client_model(self, packed_model):
        pass