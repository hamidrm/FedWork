#FedAvg

import torch
import torch.nn as nn
from core.FederatedLearningClass import *
import random
from utils.logger import *
import copy

num_of_nodes_contributor = 20

class FedPoll(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.no_r_mat = 2
        self.clients_epochs, self.num_of_rounds, self.datasets_weights = args
        self.current_seeds = [0] * self.no_r_mat
        self.client_side_r_tensors = []

    def get_name(self):
        return "FedPoll"
    
    def init_method(self):
        pass

    r_num_mat  = 2
    is_first_aggr = 1
    p_radius = {}
    use_first_max_min = True

    def aggregate(self, clients_models, global_model):

        global is_first_aggr

        if is_first_aggr == 1:
            is_first_aggr = 2
            server_aggregate(global_model, all_clients)
            return

        global_dict = global_model.state_dict() # Get a copy of the global model state_dict

        if is_first_aggr == 2:
            is_first_aggr = 0
            for key in global_dict.keys():
                p_radius[key] = np.abs(torch.max(torch.stack([(global_dict[key] - client_model.state_dict()[key]).float() for client_model in client_models],dim=0).view(1,-1)).item())
                print(f'Key {key}\t: {p_radius[key]}')
                
        percent = 0
        for key in global_dict.keys():
            # Generating R random matrixes around global_dict
            R_mats = [(global_dict[key] + (torch.rand_like(global_dict[key], dtype=torch.float, device=device) - 0.5) * 2.0 * p_radius[key]) for _ in range(r_num_mat)]
            V_mat_max = [torch.zeros_like(global_dict[key], dtype=torch.long, device=device) for _ in range(r_num_mat)]
            V_mat_min = [torch.zeros_like(global_dict[key], dtype=torch.long, device=device) for _ in range(r_num_mat)]
            
            for model in clients_models:
                model_dict = model.state_dict()[key]
                for R_mat_i in range(r_num_mat): 
                    V_mat_max[R_mat_i] += torch.where(R_mats[R_mat_i] > model_dict, 1, 0)
                    V_mat_min[R_mat_i] += torch.where(R_mats[R_mat_i] < model_dict, 1, 0)

            
            #stacked_randoms = torch.stack(R_mats, dim = 0)
            V_mat_max_stacked = torch.stack(V_mat_max, dim = 0)
            V_mat_min_stacked = torch.stack(V_mat_min, dim = 0)

            V_mat_max_stacked[V_mat_max_stacked==len(all_clients)] = 0 #Ignore the absolut Maximums
            max_val,max_index = torch.max(V_mat_max_stacked,dim=0)

            V_mat_min_stacked[V_mat_min_stacked==len(all_clients)] = 0 #Ignore the absolut Minimums
            min_val,min_index = torch.max(V_mat_min_stacked,dim=0)

            flattened_r_tensors = torch.cat([tensor.view(1, -1) for tensor in R_mats], dim=0)
            


            if use_first_max_min == True:
                new_tensor_flat = torch.gather(flattened_r_tensors, dim=0, index=max_index.view(1, -1))
                max_tensor = new_tensor_flat.view(global_dict[key].shape)
                new_tensor_flat = torch.gather(flattened_r_tensors, dim=0, index=min_index.view(1, -1))
                min_tensor = new_tensor_flat.view(global_dict[key].shape)

                if 'running_var' in key or 'running_mean' in key:
                    global_dict[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))],0).mean(0)
                    print(global_dict[key].shape)
                else:
                    global_dict[key] = (max_tensor + min_tensor) / 2
                
            else:

                max_tensor_list = []
                min_tensor_list = []
                print(f'Working on layer: {key}')
                while(True):
                    new_max_val,max_index = torch.max(V_mat_max_stacked,dim=0)
                    new_min_val,min_index = torch.max(V_mat_min_stacked,dim=0)
                    if torch.eq(new_max_val,max_val).any():
                        new_tensor_flat = torch.gather(flattened_r_tensors, dim=0, index=max_index.view(1, -1))
                        max_tensor_list.append(new_tensor_flat.view(global_dict[key].shape))
                        V_mat_max_stacked[max_index] = 0
                    if torch.eq(new_min_val,min_val).any():
                        new_tensor_flat = torch.gather(flattened_r_tensors, dim=0, index=min_index.view(1, -1))
                        min_tensor_list.append(new_tensor_flat.view(global_dict[key].shape))
                        V_mat_min_stacked[min_index] = 0
                    if not (torch.eq(new_max_val,max_val).any() or torch.eq(new_min_val,min_val).any()):
                        break

                global_dict[key] = (max_tensor + min_tensor) / 2

        global_model.load_state_dict(global_dict)
        for model in all_clients:
            model.load_state_dict(global_model.state_dict())



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

    def unpack_client_model(self, packed_model):
        return packed_model
    
    def pack_server_model(self, raw_model):
        packet_to_send = {}
        packet_to_send["global_model"] = raw_model
        packet_to_send["radius"] = self.current_radius
        packet_to_send["seeds"] = []

        for seed_index in range(self.no_r_mat):
            seed = random.randint(1, 1000)
            self.current_seeds[seed_index] = seed
            packet_to_send["seeds"].append(seed)
        return raw_model

    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        if num_of_received_model == num_of_nodes_contributor:
            return True
        else:
            return False

    # # # # # # # # # # # # # # # # # #
    # # # # #   Client Side   # # # # #
    # # # # # # # # # # # # # # # # # #

    def unpack_server_model(self, packed_model):
        global_model = packed_model["global_model"]
        radius = packed_model["radius"]
        seeds = packed_model["seeds"]

        # Generate R tensors
        for r_tensor_index in range(len(seeds)):
            self.client_side_r_tensors[r_tensor_index] = copy.deepcopy(global_model)

            for param in self.client_side_r_tensors.parameters():
                param.data = (torch.rand_like(param.data, dtype=torch.float) - 0.5) * 2.0 * radius

        
        return global_model
    
    def pack_client_model(self, raw_model):

        client_trained_model = raw_model
        for key in global_dict.keys():
            # Generating R random matrixes around global_dict
            R_mats = [(global_dict[key] + (torch.rand_like(global_dict[key], dtype=torch.float, device=device) - 0.5) * 2.0 * p_radius[key]) for _ in range(r_num_mat)]
            V_mat_max = [torch.zeros_like(global_dict[key], dtype=torch.long, device=device) for _ in range(r_num_mat)]
            V_mat_min = [torch.zeros_like(global_dict[key], dtype=torch.long, device=device) for _ in range(r_num_mat)]
            
            for model in clients_models:
                model_dict = model.state_dict()[key]
                for R_mat_i in range(r_num_mat): 
                    V_mat_max[R_mat_i] += torch.where(R_mats[R_mat_i] > model_dict, 1, 0)
                    V_mat_min[R_mat_i] += torch.where(R_mats[R_mat_i] < model_dict, 1, 0)

            
            #stacked_randoms = torch.stack(R_mats, dim = 0)
            V_mat_max_stacked = torch.stack(V_mat_max, dim = 0)
            V_mat_min_stacked = torch.stack(V_mat_min, dim = 0)

            V_mat_max_stacked[V_mat_max_stacked==len(all_clients)] = 0 #Ignore the absolut Maximums
            max_val,max_index = torch.max(V_mat_max_stacked,dim=0)

            V_mat_min_stacked[V_mat_min_stacked==len(all_clients)] = 0 #Ignore the absolut Minimums
            min_val,min_index = torch.max(V_mat_min_stacked,dim=0)

            flattened_r_tensors = torch.cat([tensor.view(1, -1) for tensor in R_mats], dim=0)
            

        new_tensor_flat = torch.gather(flattened_r_tensors, dim=0, index=max_index.view(1, -1))
        max_tensor = new_tensor_flat.view(global_dict[key].shape)
        new_tensor_flat = torch.gather(flattened_r_tensors, dim=0, index=min_index.view(1, -1))
        min_tensor = new_tensor_flat.view(global_dict[key].shape)

        if 'running_var' in key or 'running_mean' in key:
            global_dict[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))],0).mean(0)
            print(global_dict[key].shape)
        else:
            global_dict[key] = (max_tensor + min_tensor) / 2
        return raw_model
