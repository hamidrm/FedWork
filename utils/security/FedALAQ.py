import numpy as np
import torch
from utils.common import Common
from utils.logger import *
import torch.nn.functional as F
from collections import defaultdict
import copy
from utils.profiler import *

class FedALAQDefense:
    def __init__(self, contribution_ratio, alpha, noise_std = 0):
        self.contribution_ratio = contribution_ratio
        self.alpha = alpha
        self.noise_std = noise_std
        self.probabilities = {}
        self.priv_model = None
        self.counter = 0
        self.priv_model_list = []
        
    # Will be called by every one of the clients after they trained the model in a certain number of local epochs
    def build_packet(self, model, global_model, client_name, use_global_model_as_pattern = True):
        layers = {}
        layer_names = set()
        probabilities = {}

        layers_data_trained_model = {}
        layers_data_pattern_model = {}
        cosim_dict = {}

        #pattern_model = global_model if (use_global_model_as_pattern == True or self.counter == 0) else self.priv_model
        layers_count = 0
        self.counter += 1
        total_layers = 0
        # Assign random probabilities to layer names
        for key in model.keys():
            layer_name = ".".join(key.rsplit('.', 1)[:-1])

            if layer_name not in layers_data_pattern_model.keys():
                layers_data_pattern_model[layer_name] = []
                layers_data_trained_model[layer_name] = []
                total_layers += 1
            layers_data_pattern_model[layer_name].append(global_model[key].flatten())
            layers_data_trained_model[layer_name].append(model[key].flatten())

            if layer_name not in layer_names:
                layer_names.add(layer_name)
                probabilities[layer_name] = torch.rand(1).item()

        for layer in layers_data_pattern_model.keys():
            layers_data_pattern_model[layer] = torch.cat(layers_data_pattern_model[layer], dim=0)
            layers_data_trained_model[layer] = torch.cat(layers_data_trained_model[layer], dim=0)
            cosim_dict[layer] = (F.cosine_similarity(layers_data_pattern_model[layer], layers_data_trained_model[layer], dim=0) + 1) / 2.0

        layer = list(layers_data_pattern_model.keys())[-1]
        profiler.save_variable(f"FedALAQ_Cosim_{client_name}", cosim_dict[layer], self.counter - 1)
        profiler.save_variable(f"FedALAQ_m_{client_name}", torch.exp(self.alpha * (cosim_dict[layer]-1)), self.counter - 1)
        p_k = {}
        # Filter layers based on contribution_ratio
        for key, value in model.items():
            layer_name = ".".join(key.rsplit('.', 1)[:-1])
            # Use Last FC layer similarity (layer -> last layer name)
            m = torch.exp(self.alpha * (cosim_dict[layer]-1))
            p_k[key] = torch.clamp(self.contribution_ratio / m, 0, 1).item() if layer != layer_name else 1.0
            if probabilities[layer_name]  <= p_k[key]:

                layers[key] = value
                if layer_name in layer_names:
                    layers_count += 1
                    layer_names.remove(layer_name)
        
        profiler.save_variable(f"FedALAQ_no_layers_{client_name}", layers_count, self.counter - 1)
        ratio = (layers_count /  total_layers) * 100.0
        logger.log_info(f"Client '{client_name}' masked layers: {layers_count} / {total_layers} ({ratio:.2f}%)")
        for k in layers.keys():
            layers[k] = layers[k] + torch.randn_like(layers[k]) * self.noise_std
        
        
        
        
        # Save the previous model
        self.priv_model = copy.deepcopy(model)
        # Return the filtered layers
        return layers, p_k

    # Will be called by server-side to aggregate received models
    def aggregate(self, clients_partial_delta_models, global_model, weights, p_list):
        delta_model_list = defaultdict(list)
        weights_list = {}
        client_weight = [weights[i] for i in range(len(weights))]

        
        # Collect parameters by key
        for i,client_partial_delta_models in enumerate(clients_partial_delta_models):
            for key, value in client_partial_delta_models[1].items():
                delta_model_list[key].append(value * client_weight[client_partial_delta_models[0]] / p_list[i][key]) #'''  '''
                if key not in weights_list.keys():
                    weights_list[key] = 0
                weights_list[key] += (client_weight[client_partial_delta_models[0]]) / p_list[i][key]
        # Aggregate by averaging tensors
        for key in delta_model_list.keys():
            if Common.is_trainable(global_model, key):
                global_model[key] += torch.sum(torch.stack(delta_model_list[key]), dim=0) #/ weights_list[key]
            else:
                ll = []
                for c in clients_partial_delta_models:
                    ll.append(c[1][key])
                clients_partial_delta_models
                global_model[key] = torch.sum(torch.stack(ll), dim=0) / len(ll)
        return
    
    
