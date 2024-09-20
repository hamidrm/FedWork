import pickle
import time
import torch

class ClientData:

    def __init__(self, name=None, addr=None, processing_power=None, connection=None):
        self.name = name
        self.addr = addr
        self.connection = connection
        self.processing_power = processing_power
        self.listener_thread = None
        self.training_count = 0


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