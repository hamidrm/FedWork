#FedAvg

import torch
from core.FederatedLearningClass import *
import random
from utils.logger import *
import copy
from utils.common import Common


class FedPollN(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.client_side_r_tensors = []
        
        self.first_aggregation = True

        self.bits = self.get_arg(int, "bits", 8)
        self.contributors_percent = self.get_arg(int, "contributors_percent", 80)
        self.epsilon = self.get_arg(int, "epsilon", 1e-1)

        self.current_seeds = [0] * 2 ** self.bits
        self.clients_first_aggregation = True
        self.current_radius = {}
        self.loss0 = -1
        self.current_loss = 0

        logger.log_normal(f"===================================================")
        logger.log_normal(f"bits: {self.bits}, contributors_percent: {self.contributors_percent}, epsilon: {self.epsilon}")
        logger.log_normal(f"===================================================")

    def get_name(self):
        return "FedPollN"
    

# # # # # # # # # # #
#    Server Side    # 
# # # # # # # # # # # 

    def avg_aggregate(self, clients_models, global_model):
        fedavg_fraction = [self.datasets_weights[i] for i in range(len(self.datasets_weights))]
        for key in global_model.keys():
            torch_list_weights = torch.stack([clients_models[i][1][key].float() * fedavg_fraction[clients_models[i][0]] for i in range(len(clients_models))],0)
            global_model[key] = torch_list_weights.sum(0)
        
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
            if clients_models[0][1][key].dtype != torch.long and ('running_var' not in key) and ('running_mean' not in key):
                for client_index in range(len(clients_models)):
                    clients_selected_model = clients_models[client_index][1][key]
                    
                    for idx in range(len(r_tensors)):
                        mask = (clients_selected_model == idx)
                        clients_models_per_key[client_index][mask] = r_tensors[idx][key][mask]

                torch_list_weights = torch.stack([clients_models_per_key[i].float() * fedavg_fraction[i] for i in range(len(clients_models))],0)
                global_model_new = torch_list_weights.sum(0)
                diff = torch.mean(global_model_new - global_model[key])

                global_model[key] = global_model_new

                self.current_radius[key] = diff.item() + self.epsilon
            else:
                global_model[key] = clients_models[0][1][key]




    def select_clients_to_train(self, all_clients):
        if self.first_aggregation:
            self.num_of_nodes_contributor = len(all_clients)
            return all_clients
        self.num_of_nodes_contributor = int((float(self.contributors_percent) / 100.0) * len(all_clients))
        return dict(random.sample(list(all_clients.items()), self.num_of_nodes_contributor))

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
    
    def pack_client_model(self, raw_model, global_model, client_name):

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