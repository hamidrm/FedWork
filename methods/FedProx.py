#FedProx

import torch
from core.FederatedLearningClass import *
import random
from utils.common import Common
from utils.logger import *
from utils.profiler import *

class FedProx(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        self.contributors_percent = int(Common.get_param_in_args(extra_args, "contributors_percent", 100))
        self.mu = float(Common.get_param_in_args(extra_args, "mu", 0.0))
        self.num_of_nodes_contributor = 0
        self.round_num = 0


    def get_name(self):
        return "FedProx"
    
    def init_method(self):
        pass

    def aggregate(self, clients_models, global_model):

        for key in global_model.keys():
            if Common.is_trainable(global_model, key):
                torch_list_weights = torch.stack([(clients_models[i][key].float() + global_model[key]) for i in range(len(clients_models))], 0)
                global_model[key] = torch_list_weights.mean(0)
            else:
                global_model[key] = clients_models[0][key]
        self.round_num += 1


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