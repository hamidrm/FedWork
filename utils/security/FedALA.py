import numpy as np
import torch
from utils.logger import *
import torch.nn.functional as F
from collections import defaultdict
import copy
from utils.profiler import *

class FedALADefense:
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

        pattern_model = global_model if (use_global_model_as_pattern == True or self.counter == 0) else self.priv_model
        layers_count = 0
        self.counter += 1
        # Assign random probabilities to layer names
        for key in model.keys():
            layer_name = ".".join(key.rsplit('.', 1)[:-1])

            if layer_name not in layers_data_pattern_model.keys():
                layers_data_pattern_model[layer_name] = []
                layers_data_trained_model[layer_name] = []
                
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
        
        # Filter layers based on contribution_ratio
        for key, value in model.items():
            layer_name = ".".join(key.rsplit('.', 1)[:-1])
            # Use Last FC layer similarity (layer -> last layer name)
            m = torch.exp(self.alpha * (cosim_dict[layer]-1))
            if (probabilities[layer_name] * m)  <= (self.contribution_ratio) or layer == layer_name:
                logger.log_info(f"Layer: {layer_name}: Probability: {probabilities[layer_name] : .4f}, S: {probabilities[layer_name] * m  : .4f}, LCR: {self.contribution_ratio}")
                layers[key] = value
                layers_count += 1
        profiler.save_variable(f"FedALAQ_no_layers_{client_name}", layers_count, self.counter - 1)
        
        for k in layers.keys():
            layers[k] = layers[k] + torch.randn_like(layers[k]) * self.noise_std
        
        
        
        
        # Save the previous model
        self.priv_model = copy.deepcopy(model)
        
        # Return the filtered layers
        return layers

    # Will be called by server-side to aggregate received models
    def aggregate(self, partial_models, global_model, weights, beta):
        global_model_list = defaultdict(list)
        weights_list = {}
        client_weight = [weights[i] for i in range(len(weights))]

        # Collect parameters by key
        for partial_model in partial_models:
            for key, value in partial_model[1].items():
                global_model_list[key].append(value * client_weight[partial_model[0]])
                if key not in weights_list.keys():
                    weights_list[key] = 0
                weights_list[key] += (client_weight[partial_model[0]])
        # Aggregate by averaging tensors
        for key in global_model_list.keys():
            global_model[key] = (global_model[key] * beta) + (1-beta) * (torch.sum(torch.stack(global_model_list[key]), dim=0) / weights_list[key])

        return