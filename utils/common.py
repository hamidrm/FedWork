import pickle
import time
import torch
import torch.optim as optim
import os, math, json, hashlib, random, argparse
import torch.nn as nn
import numpy as np
class ClientData:

    def __init__(self, name=None, id=0, addr=None, processing_power=None, connection=None):
        self.name = name
        self.addr = addr
        self.connection = connection
        self.processing_power = processing_power
        self.listener_thread = None
        self.training_count = 0
        self.id = id


class IpAddr:
    def __init__(self, ip="127.0.0.1", port=12345):
        self.ip = ip
        self.port = port
    def get_ip(self):
        return self.ip
    def get_port(self):
        return self.port
    
class TrainingHyperParameters:
    def __init__(self, learning_rate, momentum, weight_decay):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

class Common:

    @staticmethod
    def data_convert_to_bytes(data):
        if isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, bytes):
            return data
        else:
            # Serialize the data structure using pickle
            serialized_data = pickle.dumps(data)
            return serialized_data
        
    @staticmethod
    def data_convert_from_bytes(bytes_data):
        data = pickle.loads(bytes_data)
        return data
    
    @staticmethod
    def get_param_in_args(args_str, param, def_val):
         args = str(args_str).split(",")
         for arg in args:
            param_value = str(arg).split("=")
            if len(param_value) == 2:
                if param_value[0] == param:
                    return param_value[1]
         return def_val     

    @staticmethod
    def time_ns():
        return time.time() * 1000000000  
    
    @staticmethod
    def is_trainable(model_dict, key):
        return model_dict[key].dtype != torch.long and ('running_var' not in key) and ('running_mean' not in key)
    
    
    @staticmethod
    def set_seed_over_everything(seed):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch may not have it; that's fine.
            pass
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    @staticmethod
    def set_seed_over_method(seed):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch may not have it; that's fine.
            pass
    
    @staticmethod
    def get_optimizer_class(optimizer_name):
        optimizers = {
            "SGD": optim.SGD,
            "Adam": optim.Adam,
            "Adagrad": optim.Adagrad,
            "RMSprop": optim.RMSprop,
            "Adadelta": optim.Adadelta,
            "AdamW": optim.AdamW,
            "SparseAdam": optim.SparseAdam,
            # Add more optimizers here if needed
        }

        if optimizer_name in optimizers:
            return optimizers[optimizer_name]
        else:
            raise ValueError("Invalid optimizer name")