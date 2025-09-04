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
    def __init__(self, ip_addr: IpAddr, fl_method: FederatedLearningClass, test_ds : torch.utils.data.DataLoader, model : nn.Module, loss : nn.Module, executer = "cpu", experimentMode = False):
        
        self.global_model = model
        if not experimentMode:
            self.server_comm = ServerComm(ip_addr.get_ip(), ip_addr.get_port(), self.__server_evt_fn)
            self.received_models_lock = threading.Lock()
            self.method_is_processing_lock = threading.Lock()
            
            profiler.add_var_monitor_changes("no_rcvd_total", self.server_comm, MEASURE_PROBE_CHANGES_TOTAL_RCVD_BYTES)
            profiler.add_var_monitor_changes("no_sent_total", self.server_comm, MEASURE_PROBE_CHANGES_TOTAL_SENT_BYTES)
            profiler.add_var_monitor_changes("no_rcvd_data", self.server_comm, MEASURE_PROBE_CHANGES_DATA_RCVD_BYTES)
            profiler.add_var_monitor_changes("no_rcvd_data", self.server_comm, MEASURE_PROBE_CHANGES_DATA_SENT_BYTES)
        
        self.criterion = loss().to(executer)
        self.executer = executer
        self.test_ds = test_ds
        self.received_models = []
        self.received_models_id = []
        self.model_list = []
        self.ServerTotalRecvBytesExp = 0
        self.fl_method = fl_method
        self.round_number = 0
        self.experimentMode = experimentMode
        self.clients = {}
        self.fl_method.server = self
        fl_method.init_method(self)
        logger.log_debug(f"Server initilization done.")
        

    def setClientSetExp(self, clients):
        self.clients = clients
        
    def start_round_ex(self, epochs, lr = None):
        if not self.experimentMode:
            clients = self.server_comm.get_clients()
        else:
            clients = self.clients
        clients_subset = self.fl_method.select_clients(clients)
        training_conf = {}
        logger.log_debug(f"Start new round (with dedicated configuration)")
        for client_name in clients_subset.keys():
            training_conf["epochs_num"] = epochs[client_name]
            training_conf["lr"] = lr
            if not self.experimentMode:
                self.server_comm.send_command(client_name, COMM_HEADER_CMD_START_TRAINNING, 0, training_conf)
            else:
                clients[client_name].StartTraining(training_conf)
                if self.Aggregate():
                    self.received_models.clear()

    def start_round(self, epochs, lr = None):
        if not self.experimentMode:
            clients = self.server_comm.get_clients()
        else:
            clients = self.clients
        clients_subset = self.fl_method.select_clients_to_train(clients)
        training_conf = {}
        self.round_number += 1
        training_conf["epochs_num"] = epochs
        training_conf["lr"] = lr
        logger.log_debug(f"Start new round (epochs={epochs}, lr={lr})")
        for client_name in clients_subset.keys():
            logger.log_debug(f"Start training for '{client_name}'.")
            if not self.experimentMode:
                self.server_comm.send_command(client_name, COMM_HEADER_CMD_START_TRAINNING, 0, training_conf)
            else:
                clients[client_name].StartTraining(training_conf)
                if self.Aggregate():
                    self.received_models.clear()
        

    def fetch_clients_pool(self):
        if not self.experimentMode:
            return self.server_comm.get_clients()
        else:
            return self.clients
    
    def start_periodic_mode(self, client_name, epochs, lr = None):
        if self.experimentMode:
            return
        periodic_cfg = {}
        periodic_cfg["epochs_num"] = epochs
        periodic_cfg["lr"] = lr
        self.server_comm.send_command(client_name, COMM_HEADER_CMD_START_PERIODIC_MODE, 0, periodic_cfg)

    def stop_periodic_mode(self, client_name):
        if self.experimentMode:
            return
        self.server_comm.send_command(client_name, COMM_HEADER_CMD_STOP_PERIODIC_MODE, 0, None)

    def __server_evt_fn(self, evt, client, data):
        if evt == COMM_EVT_MODEL:

            profiler.save_variable(MEASURE_PROBE_TOTAL_RCVD_BYTES, self.server_comm.no_rcvd_total, self.round_number)
            profiler.save_variable(MEASURE_PROBE_TOTAL_SENT_BYTES, self.server_comm.no_sent_total, self.round_number)
            profiler.save_variable(MEASURE_PROBE_DATA_RCVD_BYTES, self.server_comm.no_rcvd_data, self.round_number)
            profiler.save_variable(MEASURE_PROBE_DATA_SENT_BYTES, self.server_comm.no_sent_data, self.round_number)

            logger.log_debug(f"The trained model received from '{client.name}'.")

            with self.received_models_lock:
                self.received_models.append((client.id, data))

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
        elif evt == COMM_EVT_DROPEME_REQ:
            logger.log_info(f'{client.name} asked to drope off.')
        else:
            logger.log_warning(f"Undefined event received (evt={evt})!")

    def update_clients(self):
        global_model_pack = self.fl_method.pack_server_model(self.global_model.state_dict())
        if self.experimentMode:
            for client in self.fl_method.select_clients_to_update(self.clients):
                self.clients[client].set_model(global_model_pack)
        else:
            for client in self.fl_method.select_clients_to_update(self.server_comm.clients):
                self.server_comm.send_data_pkg(client, global_model_pack)
            
    def __aggregation_thread(self, packed_models_list):
        models_list = [(packed_model[0],self.fl_method.unpack_client_model(packed_model[1])) for packed_model in packed_models_list]

        profiler.start_measuring(MEASURE_PROBE_AGGR_TIME)
        self.global_model_dict = self.global_model.state_dict()

        self.fl_method.aggregate(models_list, self.global_model_dict)
        self.global_model.load_state_dict(self.global_model_dict)
        profiler.stop_measuring(MEASURE_PROBE_AGGR_TIME, self.round_number)

        self.update_clients()

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
        if self.experimentMode:
            return
        self.method_is_processing_lock.acquire()
    
    def start_training(self):
        logger.log_debug(f"Broadcasting start training command...")

        time.sleep(0.5)
        # Share Global Model among clients before start round
        self.update_clients()

        if not self.experimentMode:
            # Start training procedure
            self.fl_method.start_training()
        else:
            while(True):
                eval_loss_eval_accuracy = self.fl_method.start_training()
                if eval_loss_eval_accuracy is None:
                    #Method's procedure has been finished
                    logger.log_info(f"[{self.fl_method.get_name()}]: Method's procedure has been finished!")
                    break
                else:
                    eval_loss, eval_accuracy = eval_loss_eval_accuracy
                    logger.log_info(f"[{self.fl_method.get_name()}]: Evaluation -> Accuracy: {eval_accuracy} , Loss: {eval_loss}")
                
        if not self.experimentMode:
            if not self.method_is_processing_lock.locked():
                self.method_is_processing_lock.acquire()

    def release_all(self):
        if self.experimentMode:
            del self.clients
            return
        for client in self.server_comm.clients:
            self.server_comm.send_command(client, COMM_HEADER_CMD_TURNOFF, 0)

        while not self.server_comm.send_queue.empty():
            time.sleep(0.1)
        self.server_comm.release_all()
        #time.sleep(5)
        self.server_comm.alive = False
        if self.server_comm.server_socket:
            try:
                self.server_comm.server_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                logger.log_error(f"Network error in new connection arised!")
                pass
            finally:
                self.server_comm.server_socket.close()


    def save_var(self, var_name, var_value):
        profiler.save_variable(str(var_name), var_value, self.round_number)

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
        profiler.save_variable(MEASURE_PROBE_EVAL_ACC, eval_accuracy, self.round_number)
        profiler.save_variable(MEASURE_PROBE_EVAL_LOSS, eval_loss, self.round_number)
        logger.log_debug(f"Global model evaluation is done...")
        return eval_loss, eval_accuracy
    
    
    def EpochDone(self, name, id, epoch_info):
        logger.log_debug(f"The notification received from '{name}'.")
        profiler.save_variable(MEASURE_PROBE_CLIENT_ACC+name, epoch_info["accuracy"], self.round_number)
        profiler.save_variable(MEASURE_PROBE_CLIENT_LOSS+name, epoch_info["loss"], self.round_number)
        logger.log_debug(f'[{self.fl_method.get_name()}]: Client {name}, Accuracy is {epoch_info["accuracy"]}, Loss: {epoch_info["loss"]}.')


    def SetOptimizedModel(self, name, id, packed_data):
        
        self.ServerTotalRecvBytesExp += len(pickle.dumps(packed_data))

        logger.log_debug(f"The trained model received from '{name}'.")

        self.received_models.append((id, packed_data))

        logger.log_info(f"[{self.fl_method.get_name()}]: Trained model received from '{name}'.")

            
    def TrainingDone(self, name, id):
        logger.log_info(f'[{self.fl_method.get_name()}]: Client {name}, The round is done.')
        
    def Aggregate(self):
        if self.fl_method.ready_to_aggregate(len(self.received_models)):
            logger.log_debug(f"Start to aggregate.")
        else:
            return False
        profiler.save_variable(MEASURE_PROBE_TOTAL_RCVD_BYTES, self.ServerTotalRecvBytesExp, self.round_number)

        models_list = [(packed_model[0],self.fl_method.unpack_client_model(packed_model[1])) for packed_model in self.received_models]

        profiler.start_measuring(MEASURE_PROBE_AGGR_TIME)
        self.global_model_dict = self.global_model.state_dict()

        self.fl_method.aggregate(models_list, self.global_model_dict)
        self.global_model.load_state_dict(self.global_model_dict)
        profiler.stop_measuring(MEASURE_PROBE_AGGR_TIME, self.round_number)

        self.update_clients()

        logger.log_info(f"[{self.fl_method.get_name()}]: The aggregation has been completed, and clients are now up to date.")
            
        return True