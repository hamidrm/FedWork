
import io
import pickle
import msgpack
import numpy as np
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
        return pickle.dumps(data)

        
    @staticmethod
    def data_convert_from_bytes(bytes_data):
        return pickle.loads(bytes_data)
