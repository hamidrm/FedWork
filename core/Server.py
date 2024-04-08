# Server
import torch
import torch.nn as nn
import threading
from dataset.dataset import *
from core.ServerComm import ServerComm, ClientData
from utils.consts import *
from core.FederatedLearningClass import *
from utils.common import *
import copy
from utils.logger import *
from utils.profiler import *

class Server:
    def __init__(self, ip_addr: IpAddr, fl_method: FederatedLearningClass, test_ds : torch.utils.data.DataLoader, model : nn.Module, loss : nn.Module, executer = "cpu"):
        
        self.global_model = model
        self.server_comm = ServerComm(ip_addr.get_ip(), ip_addr.get_port(), self.__server_evt_fn)
        self.criterion = loss().to(executer)
        self.executer = executer
        self.test_ds = test_ds
        self.received_models = []
        self.received_models_lock = threading.Lock()
        self.method_is_processing_lock = threading.Lock()
        self.fl_method = fl_method
        self.round_number = 0
        self.fl_method.server = self
        fl_method.init_method()
        logger.log_debug(f"Server initilization done.")

        profiler.add_var_monitor_changes("no_rcvd_total", self.server_comm, MEASURE_PROBE_CHANGES_TOTAL_RCVD_BYTES)
        profiler.add_var_monitor_changes("no_sent_total", self.server_comm, MEASURE_PROBE_CHANGES_TOTAL_SENT_BYTES)
        profiler.add_var_monitor_changes("no_rcvd_data", self.server_comm, MEASURE_PROBE_CHANGES_DATA_RCVD_BYTES)
        profiler.add_var_monitor_changes("no_rcvd_data", self.server_comm, MEASURE_PROBE_CHANGES_DATA_SENT_BYTES)

    def start_round_ex(self, epochs, lr_mileston, gamma):
        clients = self.server_comm.get_clients()
        clients_subset = self.fl_method.select_clients(clients)
        training_conf = {}
        logger.log_debug(f"Start new round (with dedicated configuration)")
        for client_name in clients_subset.keys():
            training_conf["epochs_num"] = epochs[client_name]
            training_conf["milestone_list"] = lr_mileston[client_name]
            training_conf["gamma"] = gamma[client_name]
            self.server_comm.send_command(client_name, COMM_HEADER_CMD_START_TRAINNING, 0, training_conf)

    def start_round(self, epochs, lr_mileston: list, gamma = 0.01):
        clients = self.server_comm.get_clients()
        clients_subset = self.fl_method.select_clients_to_train(clients)
        training_conf = {}
        self.round_number += 1
        training_conf["epochs_num"] = epochs
        training_conf["milestone_list"] = lr_mileston
        training_conf["gamma"] = gamma
        logger.log_debug(f"Start new round (epochs={epochs}, lr_mileston={lr_mileston}, gamma={gamma})")
        for client_name in clients_subset.keys():
            self.server_comm.send_command(client_name, COMM_HEADER_CMD_START_TRAINNING, 0, training_conf)

    def fetch_clients_pool(self):
        return self.server_comm.clients
    
    def start_periodic_mode(self, client_name, epochs, lr_mileston, gamma, interval):
        periodic_cfg = {}
        periodic_cfg["epochs_num"] = epochs
        periodic_cfg["milestone_list"] = lr_mileston
        periodic_cfg["gamma"] = gamma
        periodic_cfg["interval"] = interval
        self.server_comm.send_command(client_name, COMM_HEADER_CMD_START_PERIODIC_MODE, 0, periodic_cfg)

    def stop_periodic_mode(self, client_name):
        self.server_comm.send_command(client_name, COMM_HEADER_CMD_STOP_PERIODIC_MODE, 0, None)

    def __server_evt_fn(self, evt, client, data):
        if evt == COMM_EVT_MODEL:

            profiler.save_variable(MEASURE_PROBE_TOTAL_RCVD_BYTES, self.server_comm.no_rcvd_total, self.round_number)
            profiler.save_variable(MEASURE_PROBE_TOTAL_SENT_BYTES, self.server_comm.no_sent_total, self.round_number)
            profiler.save_variable(MEASURE_PROBE_DATA_RCVD_BYTES, self.server_comm.no_rcvd_data, self.round_number)
            profiler.save_variable(MEASURE_PROBE_DATA_SENT_BYTES, self.server_comm.no_sent_data, self.round_number)

            logger.log_debug(f"The trained model received from '{client.name}'.")

            with self.received_models_lock:
                self.received_models.append(data)
            logger.log_info(f"[{self.fl_method.get_name()}]: Trained model received from '{client.name}'.")
            with self.received_models_lock:
                if self.fl_method.ready_to_aggregate(len(self.received_models)):
                    logger.log_debug(f"Start to aggregate in a new thread.")
                    model_list = [copy.deepcopy(model) for model in self.received_models]
                    aggregation_thread = threading.Thread(target=self.__aggregation_thread, args=(model_list, ))
                    aggregation_thread.start()
                    self.received_models.clear()
        elif evt == COMM_EVT_EPOCH_DONE_NOTIFY:
            logger.log_debug(f"The notification received from '{client.name}'.")

            profiler.save_variable(MEASURE_PROBE_CLIENT_ACC+client.name, data["accuracy"], self.round_number)
            profiler.save_variable(MEASURE_PROBE_CLIENT_LOSS+client.name, data["loss"], self.round_number)

            logger.log_debug(f'[{self.fl_method.get_name()}]: Client {client.name}, Accuracy is {data["accuracy"]}, Loss: {data["loss"]}.')
        elif evt == COMM_EVT_TRAINING_DONE:
            logger.log_info(f'[{self.fl_method.get_name()}]: Client {client.name}, The round is done.')
        elif evt == COMM_EVT_CONNECTED:
            logger.log_info(f'{client.name} is connected.')
        elif evt == COMM_EVT_DISCONNECTED:
            logger.log_info(f'{client.name} is disconnected.')
        else:
            logger.log_warning(f"Undefined event received (evt={evt})!")
                
    def __aggregation_thread(self, packed_models_list):
        models_list = [self.fl_method.unpack_client_model(packed_model) for packed_model in packed_models_list]

        profiler.start_measuring(MEASURE_PROBE_AGGR_TIME)
        self.fl_method.aggregate(models_list, self.global_model)
        profiler.stop_measuring(MEASURE_PROBE_AGGR_TIME, self.round_number)

        for client in self.fl_method.select_clients_to_update(self.server_comm.clients):
            self.server_comm.send_data_pkg(client, self.global_model)

        logger.log_info(f"[{self.fl_method.get_name()}]: The aggregation has been completed, and clients are now up to date.")
        eval_loss_eval_accuracy = self.fl_method.start_training()
        

        if eval_loss_eval_accuracy is None:
            #Method's procedure has been finished
            logger.log_info(f"[{self.fl_method.get_name()}]: Method's procedure has been finished!")
            self.method_is_processing_lock.release()

        else:
            eval_loss, eval_accuracy = eval_loss_eval_accuracy
            logger.log_info(f"[{self.fl_method.get_name()}]: Evaluation -> Accuracy: {eval_accuracy} , Loss: {eval_loss}")

    def wait_for_method(self):
        self.method_is_processing_lock.acquire()
    
    def start_training(self):
        logger.log_debug(f"Broadcasting start training command...")
        self.fl_method.start_training()
        if not self.method_is_processing_lock.locked():
            self.method_is_processing_lock.acquire()

    def release_all(self):
        self.wait_for_method()
        for client in self.server_comm.clients:
            self.server_comm.send_command(client["name"], COMM_HEADER_CMD_TURNOFF, 0)

    def evaluate_model(self):
        self.global_model.eval()
        logger.log_debug(f"Global model evaluation is started...")
        profiler.start_measuring(MEASURE_PROBE_EVAL_TIME)
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
        profiler.stop_measuring(MEASURE_PROBE_EVAL_TIME, self.round_number)
        logger.log_debug(f"Global model evaluation is done...")
        return eval_loss, eval_accuracy