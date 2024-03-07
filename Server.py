# Server
import torch
import torch.nn as nn
import threading
from dataset.dataset import *
from ServerComm import ServerComm, ClientData
from utils.consts import *
from FederatedLearningClass import *
from utils.common import *

class Server:
    def __init__(self, ip_addr: IpAddr, fl_method: FederatedLearningClass, test_ds : torch.utils.data.DataLoader, contributer_num : int, model : nn.Module, optimizer : torch.optim, loss : nn.Module, executer = "cpu"):
        self.global_model = model
        self.server_comm = ServerComm(ip_addr.get_ip(), ip_addr.get_port(), self.__server_evt_fn)
        self.global_optimizer = optimizer(self.global_model.parameters(), lr=fl_method.learning_rate, momentum=fl_method.momentum, weight_decay=fl_method.weight_decay)
        self.criterion = loss().to(executer)
        self.executer = executer
        self.contributer_num = contributer_num
        self.test_ds = test_ds
        self.received_models = []
        self.fl_method = fl_method


    def start_round_ex(self, epochs, lr_mileston, gamma):
        clients = self.server_comm.get_clients()
        clients_subset = self.fl_method.select_clients(clients, self.contributer_num)
        training_conf = {}

        for client_name in clients_subset.keys():
            training_conf["epochs_num"] = epochs[client_name]
            training_conf["milestone_list"] = lr_mileston[client_name]
            training_conf["gamma"] = gamma[client_name]
            self.server_comm.send_command(client_name, COMM_HEADER_CMD_START_TRAINNING, 0, training_conf)

    def start_round(self, epochs, lr_mileston: list, gamma = 0.01):
        clients = self.server_comm.get_clients()
        clients_subset = self.fl_method.select_clients(clients, self.contributer_num)
        training_conf = {}
        training_conf["epochs_num"] = epochs
        training_conf["milestone_list"] = lr_mileston
        training_conf["gamma"] = gamma
        for client_name in clients_subset.keys():
            self.server_comm.send_command(client_name, COMM_HEADER_CMD_START_TRAINNING, 0, training_conf)

    def __server_evt_fn(self, evt, client, data):
        if evt == COMM_EVT_MODEL:
            self.received_models.append(data)
            print(f"[{self.fl_method.get_name()}]: Trained model received from '{client}'.")
            if len(self.received_models) == self.contributer_num:
                model_list = [model.clone() for model in self.received_models]
                aggregation_thread = threading.Thread(target=self.__aggregation_thread, args=(model_list, ))
                aggregation_thread.start()
                self.received_models.clear()
        elif evt == COMM_EVT_EPOCH_DONE_NOTIFY:
            print(f'[{self.fl_method.get_name()}]: Client {client.name}, Accuracy is {float(data["accuracy"]) / 100.0} %.')
        
                
                
    def __aggregation_thread(self, packed_models_list):
        models_list = [self.fl_method.unpack_client_model(packed_model) for packed_model in packed_models_list]
        self.fl_method.aggregate(models_list, self.global_model)
        print(f"[{self.fl_method.get_name()}]: Aggregration done.")

    def evaluate_model(self):
        self.global_model.eval()

        running_loss = 0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in self.test_ds:

                inputs = inputs.to(self.executer)
                labels = labels.to(self.executer)

                outputs = self.global_model(inputs)
                _, preds = torch.max(outputs, 1)

                if self.criterion is not None:
                    loss = self.criterion(outputs, labels).item()
                else:
                    loss = 0

                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        eval_loss = running_loss / len(self.test_ds.dataset)                      
        eval_accuracy = running_corrects / len(self.test_ds.dataset)

        return eval_loss, eval_accuracy