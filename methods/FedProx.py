#FedProx

import torch
from core.FederatedLearningClass import *
import random
from utils.common import Common
from utils.logger import *
from utils.profiler import *

class FedProx(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        self.mu = self.get_arg(float, "mu", 0.0)
        self.contributors_percent = self.get_arg(int, "contributors_percent", 80)


    def get_name(self):
        return "FedProx"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):

        for key in global_model.keys():
            if Common.is_trainable(global_model, key):
                torch_list_weights = torch.stack([(clients_models[i][1][key].float() + global_model[key]) for i in range(len(clients_models))], 0)
                global_model[key] = torch_list_weights.mean(0)
            else:
                global_model[key] = clients_models[0][1][key]
        self.round_num += 1


    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)
        
    def train(self, client_train_dict):
        client_optimizer = client_train_dict["client_optimizer"]
        client_model = client_train_dict["client_model"]
        global_model = client_train_dict["global_model_state"]
        criterion = client_train_dict["criterion"]
        inputs = client_train_dict["inputs"]
        labels = client_train_dict["labels"]

        client_optimizer.zero_grad()
        outputs = client_model(inputs)
        _, preds = torch.max(outputs, 1)

        # Calculate the original loss
        loss = criterion(outputs, labels)

        # Add the FedProx proximal term
        prox_term = 0.0
        for (name, param) in client_model.named_parameters():
            global_param = global_model[name]
            prox_term += ((param - global_param) ** 2).sum()
        prox_term *= (self.mu / 2.0)

        # Total loss with FedProx term
        total_loss = loss + prox_term

        # Backpropagation
        total_loss.backward()
        client_optimizer.step()

        running_loss = total_loss.item() * inputs.size(0)
        running_corrects = torch.sum(preds == labels.data)

        return running_loss, running_corrects