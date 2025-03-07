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
from utils.security.DataManipulation import *
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
import math


class FedAvgMIA(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        
        self.fedmia_attack = None
        self.lr = 0.1
        self.labels_actual = None
        self.mixup = MixUpDefense(self.get_arg(float, "mixup_alpha", 100))


    def get_data_loaders(self):
        dir_path = self.fl_context["dataset_path"]

        file_path_train      = os.path.join(dir_path, f"dataset_node_0.ds") #Dataset of Client 0 , as the target's dataset
        file_path_validation = os.path.join(dir_path, f"dataset_node_1.ds") #Dataset of Client 1 , as the validation's dataset
        #TODO - MIX all non-targets' dataset to make a mixed dataset for validation's dataset
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
    
    def init_method(self, server):
        dataset_loader_validation, dataset_loader_train = self.get_data_loaders()
        self.fedmia_attack = FedMIA(dataset_loader_train, dataset_loader_validation, torch.optim.SGD, nn.CrossEntropyLoss)
        super().init_method(server)

    def aggregate(self, clients_models, global_model):
        logger.log_info(f"Weights: {self.datasets_weights} , Clients ID: ")
        super().aggregate(clients_models, global_model)

        if self.round_num() % 10 == 0:
            model_class = self.method_dict["arch"]
            global_model_clone = model_class().to(self.platform)
            target_model = model_class().to(self.platform)

            target_model_id = 0
            
            target_model_index = next(
                (i for i, client_state_dict in enumerate(clients_models) if client_state_dict[0] == target_model_id), 
                None
            )

            target_model.load_state_dict(clients_models[target_model_index][1]) # Specific model as the target model to inference the member being of experimental data set
            global_model_clone.load_state_dict(global_model)

            shadow_models = []
            for i, client_state_dict in enumerate(clients_models):
                if i != target_model_index:
                    shadow_model = model_class().to(self.platform)
                    shadow_model.load_state_dict(client_state_dict[1])
                    shadow_models.append(shadow_model)

            self.fedmia_attack.execute(shadow_models, target_model, global_model_clone, self.platform, self.lr)
            res_total = self.fedmia_attack.get_auc_metrics(self.platform)
            if res_total != None:
                logger.log_normal(f"FedMIA Attack on round {self.round_num()}: model id: {target_model_id}, {res_total}")
                profiler.save_variable("MIA_CUMUL", res_total["tprs"]["0.01"], self.round_num() - 1)


    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.round_num() != self.num_of_rounds:
            self.lr = 0.1 * (1 + math.cos(math.pi * self.round_num() / self.num_of_rounds)) / 2 
            self.server.start_round(self.clients_epochs, self.lr)
            return (eval_loss, eval_accuracy)
        else:
            res = self.fedmia_attack.get_auc_metrics(self.platform)
            logger.log_normal(f"Final FedMIA Attack on {self.round_num()} epochs: {res}")
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None
     
    def client_training_get_data(self, inputs, labels):
        inputs, self.labels_actual, labels, _ = self.mixup.get_data(inputs, labels)
        return inputs, labels

    def client_training_correctness(self, outputs, labels):
        return self.mixup.correctness(outputs, self.labels_actual, labels)
    
    def client_training_criterion(self, criterion_fn, outputs, labels):
        return self.mixup.criterion(criterion_fn, outputs, self.labels_actual, labels)