#FedAvg


from utils.security.dp_sgd_optimizer import *
import pickle
import torch
from core.FederatedLearningClass import *
import random
import torch.nn as nn
from utils.logger import *
from utils.profiler import *
from utils.common import Common
from utils.quantization import RandomizedQuantizer
from utils.security.DataManipulation import MixUpDefense
from utils.security.MIAPartial import *
from utils.security.FedALA import *
from utils.security.GradientSparsifier import *
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler

class FedAvgDPSGD(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)
        self.dp_optimizer = DPSGDOptimizer(self.get_arg(float, "noise_multiplier", 1.0), self.get_arg(float, "max_grad_norm", 1.0))
        self.contributors_percent = float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        self.lr = 0.01

    def get_name(self):
        return "FedAvgDPSGD"

    def init_method(self, server):

        dataset_loader_validation, dataset_loader_train = self.get_data_loaders()
        self.fedmia_attack = FedMIA(dataset_loader_train, dataset_loader_validation, torch.optim.SGD, nn.CrossEntropyLoss)
        
        super().init_method(server)

    def select_clients_to_train(self, all_clients):
        return self.select_random_clients(all_clients, self.contributors_percent)                    

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


    def aggregate(self, clients_models, global_model):

        super().aggregate(clients_models, global_model)
        

        if self.round_num() % 10 == 0:
            model_class = self.method_dict["arch"]
            global_model_clone = model_class().to(self.platform)

            target_model_id = 0 # Client 0 will be our target node to evaluate FedMIA attack on it
            
            target_model_index = next(
                (i for i, client_state_dict in enumerate(clients_models) if client_state_dict[0] == target_model_id), 
                None
            )

            global_model_clone.load_state_dict(global_model)

            shadow_models = []
            for i, client_state_dict in enumerate(clients_models):
                if i != target_model_index:
                    shadow_models.append(client_state_dict[1])

            self.fedmia_attack.execute(shadow_models, clients_models[target_model_index][1], global_model_clone, self.platform, self.lr)
            res_total = self.fedmia_attack.get_auc_metrics(self.platform)
            if res_total != None:
                logger.log_normal(f"FedMIA Attack on round {self.round_num()}: model id: {target_model_id}, {res_total}")
                profiler.save_variable("MIA_TPRS_0_1", res_total["tprs"]["0.1"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_02", res_total["tprs"]["0.02"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_01", res_total["tprs"]["0.01"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_001", res_total["tprs"]["0.001"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_0001", res_total["tprs"]["0.001"], self.round_num() - 1)
                profiler.save_variable("MIA_AUC", res_total["auc"], self.round_num() - 1)

    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != self.num_of_rounds:
            #self.lr = 0.1 * (1 + math.cos(math.pi * self.server.round_number / self.num_of_rounds)) / 2 
            #self.server.start_round(self.clients_epochs, self.lr)
            self.server.start_round(self.clients_epochs)

            return (eval_loss, eval_accuracy)
        else:
            res = self.fedmia_attack.get_auc_metrics(self.platform)
            logger.log_normal(f"Final FedMIA Attack on {self.round_num()} epochs: {res}")
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None
    
    def train(self, client_train_dict):
        client_model = client_train_dict["client_model"]
        criterion = client_train_dict["criterion"]
        inputs = client_train_dict["inputs"]
        labels = client_train_dict["labels"]

        self.dp_optimizer.set_platform(self.method_dict["platform"])
        self.dp_optimizer.set_optimizer(client_train_dict["client_optimizer"])
        self.dp_optimizer.zero_grad()
        outputs = client_model(inputs)
        loss = self.client_training_criterion(criterion, outputs, labels)
        loss.backward()
        self.dp_optimizer.step()

        # statistics
        running_corrects = self.client_training_correctness(outputs, labels)
        running_loss = loss.item() * inputs.size(0)


        return running_loss, running_corrects

