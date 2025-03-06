import numpy as np
import torch
from utils.logger import *
import torch.nn.functional as F
from collections import defaultdict

class FedALADefense:
    def __init__(self, contribution_ratio, alpha):
        self.contribution_ratio = contribution_ratio
        self.alpha = alpha

    def build_packet(self, model, global_model):
        layers = {}
        layer_names = set()
        probabilities = {}

        layers_data_trained_model = {}
        layers_data_global_model = {}
        cosim_dict = {}

        total_cosim = 0
        # Assign random probabilities to layer names
        for key in model.keys():
            layer_name = ".".join(key.rsplit('.', 1)[:-1])

            if layer_name not in layers_data_global_model.keys():
                layers_data_global_model[layer_name] = []
                layers_data_trained_model[layer_name] = []
                

            layers_data_global_model[layer_name].append(global_model[key].flatten())
            layers_data_trained_model[layer_name].append(model[key].flatten())



            if layer_name not in layer_names:
                layer_names.add(layer_name)
                
                probabilities[layer_name] = torch.rand(1).item()

        for layer in layers_data_global_model.keys():
            layers_data_global_model[layer] = torch.cat(layers_data_global_model[layer], dim=0)
            layers_data_trained_model[layer] = torch.cat(layers_data_trained_model[layer], dim=0)
            cosim_dict[layer] = (F.cosine_similarity(layers_data_global_model[layer], layers_data_trained_model[layer], dim=0) + 1) / 2.0
            total_cosim = total_cosim + cosim_dict[layer]


        # Filter layers based on contribution_ratio
        for key, value in model.items():
            layer_name = ".".join(key.rsplit('.', 1)[:-1])

            m = torch.exp(self.alpha * (cosim_dict[layer]-1))
            if (probabilities[layer_name] * m)  < (self.contribution_ratio):
                layers[key] = value
        
        # Return the filtered layers
        return layers

    def aggregate(self, partial_models, global_model, weights):
        global_model_list = defaultdict(list)  # Use defaultdict for automatic initialization
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
            global_model[key] = torch.sum(torch.stack(global_model_list[key]), dim=0) / weights_list[key]

        return