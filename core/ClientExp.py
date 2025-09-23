# Client
import threading
import torch
import torch.nn as nn
from dataset.dataset import *
from utils.consts import *
from core.FederatedLearningClass import *
from utils.logger import *
from utils.security.DataManipulation import *
import copy
from utils.common import *
from collections import Counter


    
class ClientExp:
    def __init__(self, name, id, serverInstance, hyperparameters: TrainingHyperParameters, train_ds: torch.utils.data.DataLoader, model: nn.Module, optimizer: torch.optim, loss: nn.Module, method: FederatedLearningClass,executer = "cpu"):
        
        self.client_model = model
        self.global_model = copy.deepcopy(model)
        self.criterion = loss().to(executer)
        self.executer = executer
        self.dataset = train_ds
        self.name = name
        self.id = id
        self.method = method
        self.total_epochs = 0
        self.training_count = 0
        self.serverIns = serverInstance
        self.lr = hyperparameters.learning_rate
        self.momentum = hyperparameters.momentum
        self.weight_decay = hyperparameters.weight_decay
        self.optimizer = optimizer
        logger.log_debug(f"[{name}]: Initialization done.")


    def set_model(self, model):
        self.client_model.load_state_dict(model, strict=False)
        self.global_model.load_state_dict(model, strict=False)

    def get_model_dict(self):
        return self.client_model.state_dict()

    def StartTraining(self, training_conf):
        
        epochs_num = training_conf["epochs_num"]
        lr = training_conf["lr"]

        self.client_model.to(self.executer)
        self.training_count += 1

        if lr != None:
            self.lr = lr

        if self.momentum is None:
            self.client_optimizer = self.optimizer(self.client_model.parameters(), lr=self.lr)
        elif self.weight_decay is None:
            self.client_optimizer = self.optimizer(self.client_model.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            self.client_optimizer = self.optimizer(self.client_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        
        for epoch in range(epochs_num):
            self.total_epochs += 1
            number_of_samples = 0
            self.client_model.train()

            running_loss = 0
            running_corrects = 0
            
            client_train_dict = {}
            client_train_dict["client_optimizer"] = self.client_optimizer
            client_train_dict["client_model"] = self.client_model
            client_train_dict["criterion"] = self.criterion
            client_train_dict["lr"] = self.lr
            client_train_dict["global_model_state"] = self.global_model.state_dict()
            for inputs, labels in self.dataset:
                
                inputs = inputs.to(self.executer)
                labels = labels.to(self.executer)

                inputs, labels = self.method.client_training_get_data(inputs, labels)
                number_of_samples += labels.size(0)
                client_train_dict["inputs"] = inputs
                client_train_dict["labels"] = labels
                

                method_training = self.method.train(client_train_dict)
                if method_training is not None:
                    running_loss_new, running_corrects_new = method_training
                    running_loss += running_loss_new
                    running_corrects += running_corrects_new
                else:
                    self.client_optimizer.zero_grad()
                    outputs = self.client_model(inputs)

                    loss = self.method.client_training_criterion(self.criterion, outputs, labels)
                    loss.backward()
                    self.client_optimizer.step()

                    # statistics
                    running_corrects += self.method.client_training_correctness(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    


            model = self.method.train_after_optimization(client_train_dict, epoch)
            
            if model is not None:
                self.client_model = model
            
            train_loss = running_loss / max(1, number_of_samples)
            train_accuracy = running_corrects / max(1, number_of_samples)
            
            epoch_info = {}
            epoch_info["accuracy"] = train_accuracy.item()
            epoch_info["loss"] = train_loss

            logger.log_debug(f'[{self.name}]: Epoch Done!')
            self.serverIns.EpochDone(self.name, self.id, epoch_info)
        
        if self.method != None:
            packed_data = self.method.pack_client_model(self.client_model.state_dict(), global_model = self.global_model.state_dict(), id = self.id)
            self.serverIns.SetOptimizedModel(self.name, self.id, packed_data)
        self.serverIns.TrainingDone(self.name, self.id)
