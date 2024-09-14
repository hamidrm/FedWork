#FedAvg

import torch
from core.FederatedLearningClass import *
import random
from utils.logger import *
import copy
from utils.common import Common


class FedPollN(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()

        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        
        self.client_side_r_tensors = []
        self.num_of_nodes_contributor = 0
        self.first_aggregation = True
        bits = int(Common.get_param_in_args(extra_args, "bits", 8))
        self.no_r_mat = 2 ** bits
        self.contributors_percent = int(Common.get_param_in_args(extra_args, "contributors_percent", 80))
        self.current_seeds = [0] * self.no_r_mat
        self.clients_first_aggregation = True
        self.current_radius = {}
        self.loss0 = -1
        self.current_loss = 0
        self.epsilon = float(Common.get_param_in_args(extra_args, "epsilon", 1e-1))

        logger.log_normal(f"===================================================")
        logger.log_normal(f"bits: {bits}, contributors_percent: {self.contributors_percent}, epsilon: {self.epsilon}")
        logger.log_normal(f"===================================================")

    def get_name(self):
        return "FedPollN"
    
    def init_method(self):
        pass


# # # # # # # # # # #
#    Server Side    # 
# # # # # # # # # # # 

    def avg_aggregate(self, clients_models, global_model):
        global_dict = global_model
        fedavg_fraction = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]
        for key in global_dict.keys():
            torch_list_weights = torch.stack([clients_models[i][key].float() * fedavg_fraction[i] for i in range(len(clients_models))],0)
            global_dict[key] = torch_list_weights.sum(0)
        
    def aggregate(self, clients_models, global_model):

        r_tensors = []
        if self.first_aggregation:
            self.avg_aggregate(clients_models, global_model)
            self.first_aggregation = False
            return

        for r_tensor_index in range(len(self.current_seeds)):
            r_tensors_copied = copy.deepcopy(global_model)
            torch.manual_seed(self.current_seeds[r_tensor_index])
            for key in r_tensors_copied.keys():
                if key not in self.current_radius.keys():
                    r_tensors_copied[key] = global_model[key]
                else:
                    r_tensors_copied[key] = global_model[key] + (torch.rand_like(r_tensors_copied[key], dtype=torch.float, device=self.platform) - 0.5) * 2.0 * self.current_radius[key]
            
            r_tensors.append(r_tensors_copied)
        
        fedavg_fraction = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]

        for key in global_model.keys():
            clients_models_per_key = [torch.zeros_like(global_model[key], device=self.platform) for _ in range(len(clients_models))]
            #clients_selected_model = torch.zeros_like(global_model[key])
            if clients_models[0][key].dtype != torch.long and ('running_var' not in key) and ('running_mean' not in key):
                for client_index in range(len(clients_models)):
                    clients_selected_model = clients_models[client_index][key]
                    
                    for idx in range(len(r_tensors)):
                        mask = (clients_selected_model == idx)
                        clients_models_per_key[client_index][mask] = r_tensors[idx][key][mask]

                torch_list_weights = torch.stack([clients_models_per_key[i].float() * fedavg_fraction[i] for i in range(len(clients_models))],0)
                global_model_new = torch_list_weights.sum(0)
                diff = torch.mean(global_model_new - global_model[key])

                global_model[key] = global_model_new

                self.current_radius[key] = diff.item() + self.epsilon
            else:
                global_model[key] = clients_models[0][key]






    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.first_aggregation == False and self.loss0 == -1:
            self.loss0 = eval_loss
        self.current_loss = eval_loss
        if self.server.round_number != self.num_of_rounds:
            #self.server.update_clients()
            self.server.start_round(self.clients_epochs, [100, 200], 0.0001)
            return (eval_loss, eval_accuracy)
        else:
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None

    def select_clients_to_train(self, all_clients):
        if self.first_aggregation:
            self.num_of_nodes_contributor = len(all_clients)
            return all_clients
        self.num_of_nodes_contributor = int((float(self.contributors_percent) / 100.0) * len(all_clients))
        return dict(random.sample(list(all_clients.items()), self.num_of_nodes_contributor))

    def select_clients_to_update(self, all_clients):
        return all_clients

    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        if num_of_received_model == self.num_of_nodes_contributor:
            return True
        else:
            return False


    def unpack_client_model(self, packed_model):
        # No additional process is needed in this stage
        # We'll work on packets in aggregation step
        return packed_model
    
    def pack_server_model(self, raw_model):
        packet_to_send = {}
        packet_to_send["global_model"] = raw_model
        packet_to_send["radius"] = self.current_radius
        packet_to_send["seeds"] = []

        for seed_index in range(self.no_r_mat):
            seed = random.randint(1, 1000000)
            self.current_seeds[seed_index] = seed
            packet_to_send["seeds"].append(seed)
        return packet_to_send
    

# # # # # # # # # # #
#    Client Side    # 
# # # # # # # # # # # 

    def unpack_server_model(self, packed_model):
        global_model = packed_model["global_model"]
        radius = packed_model["radius"]
        seeds = packed_model["seeds"]
        
        self.client_side_r_tensors = []
        # Generate R tensors
        for r_tensor_index in range(len(seeds)):
            copied_model = copy.deepcopy(global_model)

            torch.manual_seed(seeds[r_tensor_index])
            for key in copied_model.keys():
                if key not in radius.keys():
                    copied_model[key] = global_model[key]
                else:
                    copied_model[key] = global_model[key] + (torch.rand_like(copied_model[key], dtype=torch.float32, device=self.platform) - 0.5) * 2.0 * radius[key]

            self.client_side_r_tensors.append(copied_model)

        return global_model
    
    def pack_client_model(self, raw_model, global_model):

        client_trained_model = raw_model

        if self.clients_first_aggregation:
            self.clients_first_aggregation = False
            return client_trained_model

        output_model = {}
        for key in client_trained_model.keys():
            if raw_model[key].dtype != torch.long and ('running_var' not in key) and ('running_mean' not in key):
                r_tensors_list = [sub_elem[key].to(self.platform) for sub_elem in self.client_side_r_tensors]
                r_tensors_stacked = torch.stack(r_tensors_list)

                distance = torch.abs(r_tensors_stacked - client_trained_model[key])
                nearest_values_indices = torch.argmin(distance, dim=0)
                output_model[key] = nearest_values_indices.clone().detach().to(torch.uint8)
            else:
                output_model[key] = client_trained_model[key]
        return output_model