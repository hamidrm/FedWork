#FedAvg

import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *
import copy

num_of_nodes_contributor = 10

class FedPoll(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.no_r_mat = 16
        self.clients_epochs, self.num_of_rounds, self.datasets_weights = args
        self.current_seeds = [0] * self.no_r_mat
        self.client_side_r_tensors = []
        self.current_radius = 1e-3
        self.first_aggregation = True
        self.clients_first_aggregation = True

    def get_name(self):
        return "FedPoll"
    
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
                r_tensors_copied[key] = global_model[key] + (torch.rand_like(r_tensors_copied[key], dtype=torch.float) - 0.5) * 2.0 * self.current_radius
            r_tensors.append(r_tensors_copied)
            
        for key in global_model.keys():
            
            V_mat_max = [torch.zeros_like(global_model[key], dtype=torch.long) for _ in range(self.no_r_mat)]
            V_mat_min = [torch.zeros_like(global_model[key], dtype=torch.long) for _ in range(self.no_r_mat)]
            
            for model in clients_models:
                for R_mat_i in range(self.no_r_mat):
                    V_mat_max[R_mat_i] += torch.where(model[R_mat_i][key] == True, 1, 0)
                    V_mat_min[R_mat_i] += torch.where(model[R_mat_i][key] == True, 0, 1)
            V_mat_max_stacked = torch.stack(V_mat_max, dim = 0)
            V_mat_min_stacked = torch.stack(V_mat_min, dim = 0)

            V_mat_max_stacked[V_mat_max_stacked==len(clients_models)] = 0 #Ignore the absolut Maximums
            _, max_index = torch.max(V_mat_max_stacked,dim=0)

            V_mat_min_stacked[V_mat_min_stacked==len(clients_models)] = 0 #Ignore the absolut Minimums
            _, min_index = torch.max(V_mat_min_stacked,dim=0)

            flattened_r_tensors = torch.cat([tensor[key].view(1, -1) for tensor in r_tensors], dim=0)
            

            new_tensor_flat = torch.gather(flattened_r_tensors, dim=0, index=max_index.view(1, -1))
            max_tensor = new_tensor_flat.view(global_model[key].shape)
            new_tensor_flat = torch.gather(flattened_r_tensors, dim=0, index=min_index.view(1, -1))
            min_tensor = new_tensor_flat.view(global_model[key].shape)

            global_model[key] = (max_tensor + min_tensor) / 2


    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != self.num_of_rounds:
            #self.server.update_clients()
            self.server.start_round(self.clients_epochs, [100, 200], 0.0001)
            return (eval_loss, eval_accuracy)
        else:
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None

    def select_clients_to_train(self, all_clients):
        return all_clients

    def select_clients_to_update(self, all_clients):
        return all_clients

    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        if num_of_received_model == num_of_nodes_contributor:
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
                copied_model[key] = global_model[key] + (torch.rand_like(copied_model[key], dtype=torch.float) - 0.5) * 2.0 * radius

            self.client_side_r_tensors.append(copied_model)

        return global_model
    
    def pack_client_model(self, raw_model):

        client_trained_model = raw_model

        if self.clients_first_aggregation:
            self.clients_first_aggregation = False
            return client_trained_model
        

        output_model = []

        for r_tensor_i in range(len(self.client_side_r_tensors)):
            copied_model = copy.deepcopy(client_trained_model)
            for param_key in copied_model.keys():
                copied_model[param_key].copy_(torch.zeros_like(copied_model[param_key], dtype=torch.bool))
            output_model.append(copied_model)
        
        for key in client_trained_model.keys():
            for r_tensor_i in range(len(self.client_side_r_tensors)):
                output_model[r_tensor_i][key] = self.client_side_r_tensors[r_tensor_i][key] > client_trained_model[key]

        return output_model