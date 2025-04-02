from abc import ABC, abstractmethod
import random
import torch
from typing import Type, TypeVar
from utils.common import Common


T = TypeVar("T")

class FederatedLearningClass(ABC):

    def __init__(self, method_name, fl_context, method_args):
        self.method_name = method_name
        self.method_dict = fl_context["methods_list"][method_name]
        self.clients_epochs = self.method_dict["num_of_epochs"]
        self.num_of_rounds = fl_context["num_of_rounds"]
        self.datasets_weights = fl_context["dataset_weights"]
        self.platform = self.method_dict["platform"]
        self.extra_args = method_args
        self.fl_context = fl_context
        self.num_of_contributor_nodes = 0

    def get_arg(self, value_type: Type[T], name: str, default_value: T) -> T:
        return value_type(Common.get_param_in_args(self.extra_args, name, default_value))

    def select_random_clients(self, all_clients : dict, percent):
        self.num_of_contributor_nodes = int(percent * len(all_clients))
        return dict(random.sample(list(all_clients.items()), self.num_of_contributor_nodes)) 
    
    @abstractmethod
    def get_name(self):
        pass

    def init_method(self, server = None):
        # Is this called by the Server-Side?
        if server != None:
            self.server = server

    def aggregate(self, clients_models, global_model):
        
        for key in global_model.keys():
            torch_list_weights = torch.stack([clients_models[i][1][key].float() * self.datasets_weights[clients_models[i][0]] for i in range(len(clients_models))],0)
            total_weight = sum([self.datasets_weights[clients_models[i][0]] for i in range(len(clients_models))])
            global_model[key] = torch_list_weights.sum(0) / total_weight

    def round_num(self):
        return self.server.round_number

    def select_clients_to_train(self, all_clients):
        self.num_of_contributor_nodes = len(all_clients)
        return all_clients
    
    def select_clients_to_update(self, all_clients):
        return all_clients

    def start_training(self):
        eval_loss, eval_accuracy = self.server.evaluate_model()
        if self.server.round_number != self.num_of_rounds:
            self.server.start_round(self.clients_epochs)
            return (eval_loss, eval_accuracy)
        else:
            return None
    
    def pack_client_model(self, raw_model, global_model):
        return raw_model

    def unpack_client_model(self, packed_model):
        return packed_model
    
    def pack_server_model(self, raw_model):
        return raw_model

    def unpack_server_model(self, packed_model):
        return packed_model

    def train(self, client_train_dict : dict):
        return None
    
    def train_after_optimization(self, client_train_dict : dict, epoch_num):
        return None
    
    def client_training_get_data(self, inputs, labels):
        return inputs, labels

    def client_training_correctness(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        return torch.sum(preds == labels.data)
    
    def client_training_criterion(self, criterion_fn, outputs, labels):
        return criterion_fn(outputs, labels)

    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        if num_of_received_model == self.num_of_contributor_nodes:
            return True
        else:
            return False