# Client
import threading
import torch
import torch.nn as nn
from dataset.dataset import *
from utils.consts import *
from core.FederatedLearningClass import *
from core.ClientComm import *
from utils.logger import *
import copy

class Client:
    def __init__(self, name, ip_addr: IpAddr, hyperparameters: TrainingHyperParameters, train_ds: torch.utils.data.DataLoader, model: nn.Module, optimizer: torch.optim, loss: nn.Module, method: FederatedLearningClass,executer = "cpu"):
        
        self.client_model = model
        self.global_model = copy.deepcopy(model)
        if hyperparameters.momentum is None:
            self.client_optimizer = optimizer(self.client_model.parameters(), lr=hyperparameters.learning_rate)
        elif hyperparameters.weight_decay is None:
            self.client_optimizer = optimizer(self.client_model.parameters(), lr=hyperparameters.learning_rate, momentum=hyperparameters.momentum)
        else:
            self.client_optimizer = optimizer(self.client_model.parameters(), lr=hyperparameters.learning_rate, momentum=hyperparameters.momentum, weight_decay=hyperparameters.weight_decay)

        self.criterion = loss().to(executer)
        self.executer = executer
        self.dataset = train_ds
        self.name = name
        self.method = method
        self.total_epochs = 0
        self.training_count = 0
        self.is_periodic_training_enabled = False
        self.periodic_training_thread = None
        self.periodic_training_wait_evt = None
        self.periodic_training_epochs = 5
        self.periodic_training_lr_mileston = []
        self.periodic_training_gamma = 0
        self.lr = hyperparameters.learning_rate
        self.is_training_lock = threading.Lock()
        self.client_comm = ClientComm(name, ip_addr.get_ip(), ip_addr.get_port(), self.__client_evt_cb)
        logger.log_debug(f"[{name}]: Initialization done.")

    def __client_evt_cb(self, evt, data):
        logger.log_debug(f"[{self.name}]: Event received(Event={evt}).")
        if evt == COMM_EVT_TRAINING_START:
            if self.is_periodic_training_enabled:
                logger.log_warning(f'[{self.name}]: Invalid request! Periodic training is enabled.')
            else:
                logger.log_info(f'[{self.name}]: Training is starting...')
                self.StartTraining(data["epochs_num"], data["milestone_list"], data["gamma"])
                logger.log_info(f'[{self.name}]: Training done (Epochs: {data["epochs_num"]}).')
        elif evt == COMM_EVT_EPOCHS_TOTAL_COUNT_REQ:
            self.client_comm.send_notification_to_server(COMM_EVT_EPOCHS_TOTAL_COUNT_REQ, self.total_epochs)
        elif evt == COMM_EVT_TRAINING_TOTAL_COUNT_REQ:
            self.client_comm.send_notification_to_server(COMM_EVT_TRAINING_TOTAL_COUNT_REQ, self.training_count)
        elif evt == COMM_EVT_START_PERIODIC_TRAINING:
            logger.log_info(f'[{self.name}]: Periodic training is starting...')
            self.start_periodic_training(data["epochs_num"], data["milestone_list"], data["gamma"], data["interval"])
        elif evt == COMM_EVT_STOP_PERIODIC_TRAINING:
            logger.log_info(f'[{self.name}]: Periodic training is stoping...')
            self.stop_periodic_training()
        elif evt == COMM_EVT_MODEL:
            logger.log_info(f'[{self.name}]: New model received.')
            self.set_model(self.method.unpack_server_model(data))
        elif evt == COMM_EVT_CONNECTED:
            logger.log_info(f'[{self.name}]: is connected.')
        elif evt == COMM_EVT_DISCONNECTED:
            logger.log_info(f'[{self.name}]: is disconnected.')
        elif evt == COMM_EVT_TURNOFF:
            logger.log_info(f'[{self.name}]: TurnOff command received.')
        else:
            logger.log_warning(f"Undefined event received (evt={evt})!")

    def set_model(self, model):
        self.client_model.load_state_dict(model)
        self.global_model.load_state_dict(model)

    def get_model_dict(self):
        return self.client_model.state_dict()

    def start_periodic_training(self, epochs, lr_mileston: list, gamma = 0.01, interval = 10):
        if self.is_periodic_training_enabled or ((self.periodic_training_thread is not None) and (self.periodic_training_thread.is_alive())):
            logger.log_warning(f'[{self.name}]: Invalid request! Periodic training has been strated before.')
            return
        self.periodic_training_epochs = epochs
        self.periodic_training_lr_mileston = lr_mileston
        self.periodic_training_gamma = gamma
        self.periodic_training_wait_evt = threading.Event()
        self.periodic_training_thread = threading.Thread(target=self.__periodic_training_thread, args=(self.periodic_training_wait_evt, interval))
        self.periodic_training_thread.start()
        self.is_periodic_training_enabled = True

    def stop_periodic_training(self):
        if self.is_periodic_training_enabled == False:
            logger.log_warning(f'[{self.name}]: Invalid request! Periodic training is not enabled.')
            return
        self.is_periodic_training_enabled = False
        self.periodic_training_wait_evt.set()

    def __periodic_training_thread(self, event, interval):
        while self.is_periodic_training_enabled:
            event.wait(float(interval) / 1000.0)
            if self.is_periodic_training_enabled:
                self.StartTraining(self.periodic_training_epochs, self.periodic_training_lr_mileston, self.periodic_training_gamma)
            logger.log_debug(f'[{self.name}]: Periodic training tick!')


    def StartTraining(self, epochs_num, lr_scheduler_milestone_list : list = None, gamma : float = 0.1):
        self.is_training_lock.acquire()
        if (lr_scheduler_milestone_list is not None) and (len(lr_scheduler_milestone_list) != 0):
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.client_optimizer,
                                                     milestones=lr_scheduler_milestone_list,
                                                     gamma=gamma,
                                                     last_epoch=-1)
        self.client_model.to(self.executer)
        self.training_count += 1



        for epoch in range(epochs_num):
            self.total_epochs += 1

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
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.client_optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            model = self.method.train_after_optimization(client_train_dict, epoch)
            
            if model is not None:
                self.client_model = model
            
            train_loss = running_loss / len(self.dataset.dataset)
            train_accuracy = running_corrects / len(self.dataset.dataset)
            
            epoch_info = {}
            epoch_info["accuracy"] = train_accuracy.item()
            epoch_info["loss"] = train_loss

            logger.log_debug(f'[{self.name}]: Epoch Done!')
            self.client_comm.send_notification_to_server(COMM_HEADER_NOTI_EPOCH_DONE, 0, epoch_info)

            if lr_scheduler_milestone_list is not None:
                scheduler.step()
        
        if self.method != None:
            packed_data = self.method.pack_client_model(self.client_model.state_dict(), global_model = self.global_model.state_dict())
            self.client_comm.send_data_to_server(packed_data)
        self.client_comm.send_notification_to_server(COMM_HEADER_NOTI_TRAINNING_DONE, 0)
        self.is_training_lock.release()
