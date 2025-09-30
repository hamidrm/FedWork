import torch
import copy

class DINAR:
    def __init__(self, sensitive_layer, noise_std=1e-3):
        self.sensitive_layer = sensitive_layer
        self.noise_std = noise_std
        self.private_layer = {}

    def personalize_model(self, global_model):
        if self.private_layer:
            for key in global_model.keys():
                if self.sensitive_layer in key:
                    global_model[key] = self.private_layer[key]

    def obfuscate_and_store_layer(self, client_model):
        for key in client_model.keys():
            if self.sensitive_layer in key:
                self.private_layer[key] = copy.deepcopy(client_model[key])
                client_model[key] = client_model[key] + torch.rand_like(client_model[key]) * self.noise_std