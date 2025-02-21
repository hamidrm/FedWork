from collections import defaultdict
import numpy as np
import torch
from utils.logger import *
import torch.nn.functional as F

class MixUpDefense:
    def __init__(self, alpha) -> None:
        self.alpha = alpha
        self.mixup_ratio = 0

    def get_data(self, inputs, targets):
        if self.alpha > 0:
            self.mixup_ratio = np.random.beta(self.alpha, self.alpha)
        else:
            self.mixup_ratio = 1.0
    
        batch_size = inputs.size(0)
        shuffled_indices = torch.randperm(batch_size)

        mixed_inputs = self.mixup_ratio * inputs + (1 - self.mixup_ratio) * inputs[shuffled_indices, :]

        return mixed_inputs, targets, targets[shuffled_indices], self.mixup_ratio

    def criterion(self, criterion_fn, pred, y_actual, y_mixed):
        return self.mixup_ratio * criterion_fn(pred, y_actual) + (1 - self.mixup_ratio) * criterion_fn(pred, y_mixed)
    
    def correctness(self, pred, y_actual, y_mixed):
        _, predicted = torch.max(pred, 1)
        correct_a = (predicted == y_actual).sum().item()
        correct_b = (predicted == y_mixed).sum().item()
        return torch.tensor(self.mixup_ratio * correct_a + (1 - self.mixup_ratio) * correct_b)

class InstaHideDataObfuscator:
    def __init__(self, private_loader, public_loader=None, num_mix=3, max_weight=0.7):
        self.private_loader = private_loader
        self.public_loader = public_loader
        self.num_mix = num_mix
        self.max_weight = max_weight

    def generate_mix_weights(self, batch_size):
        weights = np.random.rand(batch_size, self.num_mix)
        weights /= weights.sum(axis=1, keepdims=True)
        weights = np.clip(weights, 0, self.max_weight)
        return torch.from_numpy(weights).float()

    def apply_instahide(self, inputs, targets, public_data=None):
        batch_size = inputs.size(0)
        mix_weights = self.generate_mix_weights(batch_size)
        mixed_inputs = mix_weights[:, 0].view(-1, 1, 1, 1) * inputs
        mixed_labels = mix_weights[:, 0].view(-1, 1) * targets

        for i in range(1, self.num_mix):
            if public_data is not None and np.random.rand() > 0.5:
                rand_data, rand_labels = public_data
            else:
                rand_data, rand_labels = inputs, targets
            perm_indices = torch.randperm(batch_size)
            permuted_data = rand_data[perm_indices]
            permuted_labels = rand_labels[perm_indices]
            mixed_inputs += mix_weights[:, i].view(-1, 1, 1, 1) * permuted_data
            mixed_labels += mix_weights[:, i].view(-1, 1) * permuted_labels

        sign_flip = (torch.randint(2, size=mixed_inputs.shape) * 2.0 - 1).float()
        mixed_inputs *= sign_flip

        return mixed_inputs, mixed_labels

    def generate_obfuscated_batch(self):

        inputs, targets = next(iter(self.private_loader))

        public_data = None
        if self.public_loader:
            pub_inputs, pub_targets = next(iter(self.public_loader))
            public_data = (pub_inputs, pub_targets)

        obfuscated_inputs, obfuscated_labels = self.apply_instahide(inputs, targets.float(), public_data)
        return obfuscated_inputs, obfuscated_labels
    
class FedLADefeat:
    def __init__(self, contribution_ratio):
        self.contribution_ratio = contribution_ratio

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

            alpha = 200.0
            m = torch.exp(alpha * (cosim_dict[layer]-1))
            if (probabilities[layer_name] * m)  < (self.contribution_ratio):
                layers[key] = value
        
        # Return the filtered layers
        return layers

    def aggregate(self, partial_models, global_model):
        global_model_list = defaultdict(list)  # Use defaultdict for automatic initialization

        # Collect parameters by key
        for partial_model in partial_models:
            for key, value in partial_model.items():
                global_model_list[key].append(value)

        # Aggregate by averaging tensors
        for key in global_model_list.keys():
            global_model[key] = torch.mean(torch.stack(global_model_list[key]), dim=0)

        return