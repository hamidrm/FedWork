#FedAvgMIA
import pickle
import torch
from core.FederatedLearningClass import *
import random
import torch.nn as nn
from utils.logger import *
from utils.profiler import *
from utils.common import Common
from utils.security.MIA import *
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
import math

class FedAvgMIA(FederatedLearningClass):

    def __init__(self, args = ()):
        super().__init__()
        self.clients_epochs, self.num_of_rounds, self.datasets_weights, self.platform, extra_args = args
        self.contributors_percent = int(Common.get_param_in_args(extra_args, "contributors_percent", 100))
        self.num_of_nodes_contributor = 0
        self.round_num = 0
        self.cos_mia = None
        self.fedmia_attack = None
        self.lr = 0.1

    def get_data_loaders(self):

        dir_path = os.path.join("output", "fedavg_miad")
        dir_path = os.path.join(dir_path, "dataset")

        file_path_train      = os.path.join(dir_path, f"dataset_node_0.ds")
        file_path_validation = os.path.join(dir_path, f"dataset_node_1.ds")

        with open(file_path_train,      'rb') as f:
            dataset_loader_train      = pickle.loads(f.read())

        with open(file_path_validation, 'rb') as f:
            dataset_loader_validation = pickle.loads(f.read())

        new_sampler = BatchSampler(SequentialSampler(dataset_loader_train.dataset), batch_size=10, drop_last=False)
        dataset_loader_train = DataLoader(dataset_loader_train.dataset, batch_sampler=new_sampler)

        new_sampler = BatchSampler(SequentialSampler(dataset_loader_validation.dataset), batch_size=10, drop_last=False)
        dataset_loader_validation = DataLoader(dataset_loader_validation.dataset, batch_sampler=new_sampler)

        return dataset_loader_validation, dataset_loader_train


    def get_name(self):
        return "FedAvgMIA"
    
    def init_method(self):
        dataset_loader_validation, dataset_loader_train = self.get_data_loaders()
        self.cos_mia = CosMIA(dataset_loader_train, dataset_loader_validation, torch.optim.SGD, nn.CrossEntropyLoss)
        #self.fedmia_attack = FedMIA(dataset_loader_train, dataset_loader_validation, torch.optim.SGD, nn.CrossEntropyLoss)

    def aggregate(self, clients_models, global_model, global_model_obj, clients_id):


        self.round_num += 1

        if self.round_num % 10 == 0:
            model_class = type(global_model_obj)
            target_model = model_class().to(self.platform)
            global_model_clone = model_class().to(self.platform)

            target_model_name = "Client0"
            target_model_index = clients_id.index(target_model_name)
            target_model.load_state_dict(clients_models[target_model_index]) # Specific model as the target model to inference the member being of experimental data set
            global_model_clone.load_state_dict(global_model)


            self.cos_mia.execute(target_model, global_model_clone, self.platform, 0.1)
            res = self.cos_mia.get_last_auc_metrics()

            # shadow_models=[]
            # for i, client_state_dict in enumerate(clients_models):
            #     shadow_model = model_class().to(self.platform)
            #     shadow_model.load_state_dict(client_state_dict)
            #     if target_model_index != i:
            #         shadow_models.append(shadow_model)

            # self.fedmia_attack.execute(shadow_models, target_model, global_model_clone, self.platform, self.lr)
            # res = self.fedmia_attack.get_last_auc_metrics()

            logger.log_normal(f"Cos Attack on {self.round_num} epochs: {res}, model: {target_model_name}")
            profiler.save_variable("CosII", res["tprs"]["0.01"], self.round_num - 1)

        
        for key in global_model.keys():
            torch_list_weights = torch.stack([clients_models[i][key].float() for i in range(len(clients_models))],0)
            global_model[key] = torch_list_weights.mean(0)

    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != self.num_of_rounds:
            #self.lr *= 0.99
            self.lr = 0.1 * (1 + math.cos(math.pi * self.server.round_number / self.num_of_rounds)) / 2 
            self.server.start_round(self.clients_epochs, self.lr)
            return (eval_loss, eval_accuracy)
        else:
            #res = self.fedmia_attack.get_auc_metrics(self.platform)
            res = self.cos_mia.get_auc_metrics(self.platform)
            logger.log_normal(f"Final Cos Attack on {self.round_num} epochs: {res}")
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None

    def select_clients_to_train(self, all_clients):
        self.num_of_nodes_contributor = int((float(self.contributors_percent) / 100.0) * len(all_clients))
        return dict(random.sample(list(all_clients.items()), self.num_of_nodes_contributor))

    def select_clients_to_update(self, all_clients):
        return all_clients

    def pack_client_model(self, raw_model, global_model):
        return raw_model

    def unpack_client_model(self, packed_model):
        return packed_model
    
    def pack_server_model(self, raw_model):
        return raw_model

    def unpack_server_model(self, packed_model):
        return packed_model
    
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        if num_of_received_model == self.num_of_nodes_contributor:
            return True
        else:
            return False