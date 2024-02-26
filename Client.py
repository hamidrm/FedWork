# Server
import torch
import torch.nn as nn

import threading
from dataset.dataset import *
from ServerComm import *
from utils.consts import *
from FederatedLearningClass import *
from ClientComm import *


class Client:
    def __init__(self, name, ip_addr: IpAddr, hyperparameters: TrainingHyperParameters, train_ds: torch.utils.data.DataLoader, arch: nn.Module, optimizer: torch.optim, loss: nn.Module, executer = "cpu"):
        self.client_comm = ClientComm(name, ip_addr.get_ip(), ip_addr.get_port(), self.__client_evt_cb)
        self.client_model = arch().to(executer)
        self.client_optimizer = optimizer(self.client_model.parameters(), lr=hyperparameters.learning_rate, momentum=hyperparameters.momentum, weight_decay=hyperparameters.weight_decay)
        self.criterion = loss().to(executer)
        self.executer = executer
        self.dataset = train_ds
        self.name = name

    def __client_evt_cb(self, name, evt, data):
        if evt == COMM_EVT_TRAINING_START:
            self.StartTraining(data["epochs_num"], data["milestone_list"], data["gamma"])
            print(f'[{name}]: Training done (Epochs: {data["epochs_num"]}).')

    def set_model(self, model):
        self.client_model.load_state_dict(model.state_dict())

    def get_model_dict(self):
        return self.client_model.state_dict()

    def StartTraining(self, epochs_num, lr_scheduler_milestone_list : list = None, gamma : float = 0.1):

        if lr_scheduler_milestone_list is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.client_optimizer,
                                                     milestones=lr_scheduler_milestone_list,
                                                     gamma=gamma,
                                                     last_epoch=-1)
        self.client_model.to(self.executer)

        for epoch in range(epochs_num):

            # Training
            self.client_model.train()

            running_loss = 0
            running_corrects = 0

            for inputs, labels in self.dataset:
                inputs = inputs.to(self.executer)
                labels = labels.to(self.executer)

                # zero the parameter gradients
                self.client_optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.client_model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.client_optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_loss = running_loss / len(self.dataset.dataset)
            train_accuracy = running_corrects / len(self.dataset.dataset)
            self.client_comm.send_notification_to_server(COMM_HEADER_NOTI_EPOCH_DONE, int(train_accuracy.item() * 10000))
            if lr_scheduler_milestone_list is not None:
                scheduler.step()
        self.client_comm.send_notification_to_server(COMM_HEADER_CMD_TRAINNING_DONE, 0)
