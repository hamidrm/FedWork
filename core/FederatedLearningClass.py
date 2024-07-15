from abc import ABC, abstractmethod

class FederatedLearningClass(ABC):
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 1e-7

    def set_server(self, server):
        self.server = server
        
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def init_method(self):
        pass

    @abstractmethod
    def aggregate(self, clients_model, global_model):
        pass

    @abstractmethod
    def select_clients_to_train(self, all_clients):
        pass
    
    @abstractmethod
    def select_clients_to_update(self, all_clients):
        pass

    @abstractmethod
    def start_training(self):
        pass
    
    @abstractmethod
    def pack_client_model(self, raw_model, global_model=None):
        pass

    @abstractmethod
    def unpack_client_model(self, packed_model):
        pass

    @abstractmethod
    def pack_server_model(self, raw_model):
        pass

    @abstractmethod
    def unpack_server_model(self, packed_model):
        pass

    def set_hyperparameters(self, learning_rate : float, momentum : float, weight_decay : float):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    @abstractmethod
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        pass