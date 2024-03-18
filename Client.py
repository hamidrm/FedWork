# Server
import torch
import torch.nn as nn
import utils.logger as logger
import utils.profiler as profiler
from dataset.dataset import *
from ServerComm import *
from utils.consts import *
from FederatedLearningClass import *
from ClientComm import *
from utils.logger import *

class Client:
    def __init__(self, name, ip_addr: IpAddr, hyperparameters: TrainingHyperParameters, train_ds: torch.utils.data.DataLoader, model: nn.Module, optimizer: torch.optim, loss: nn.Module, executer = "cpu"):
        self.client_comm = ClientComm(name, ip_addr.get_ip(), ip_addr.get_port(), self.__client_evt_cb)
        self.client_model = model
        self.client_optimizer = optimizer(self.client_model.parameters(), lr=hyperparameters.learning_rate, momentum=hyperparameters.momentum, weight_decay=hyperparameters.weight_decay)
        self.criterion = loss().to(executer)
        self.executer = executer
        self.dataset = train_ds
        self.name = name
        logger.log_debug(f"[{name}]: Initialization done.")

    def __client_evt_cb(self, name, evt, data):
        if evt == COMM_EVT_TRAINING_START:
            logger.log_info(f'[{name}]: Training is starting...')
            self.StartTraining(data["epochs_num"], data["milestone_list"], data["gamma"])
            logger.log_info(f'[{name}]: Training done (Epochs: {data["epochs_num"]}).')

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
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                self.client_optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_loss = running_loss / len(self.dataset.dataset)
            train_accuracy = running_corrects / len(self.dataset.dataset)
            
            epoch_info = {}
            epoch_info["accuracy"] = train_accuracy.item()
            epoch_info["loss"] = train_loss
            
            self.client_comm.send_notification_to_server(COMM_HEADER_NOTI_EPOCH_DONE, 0, epoch_info)
            if lr_scheduler_milestone_list is not None:
                scheduler.step()
        self.client_comm.send_data_to_server(self.client_model.state_dict())
        self.client_comm.send_notification_to_server(COMM_HEADER_CMD_TRAINNING_DONE, 0)
