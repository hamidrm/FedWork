from abc import ABC, abstractmethod

class FederatedLearningClass(ABC):
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 1e-7

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def aggregate(self, clients_model, global_model):
        pass

    @abstractmethod
    def select_clients(self, all_clients, number):
        pass
    
    @abstractmethod
    def pack_client_model(self, raw_model):
        pass

    @abstractmethod
    def unpack_client_model(self, packed_model):
        pass

    def set_hyperparameters(self, learning_rate : float, momentum : float, weight_decay : float):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        