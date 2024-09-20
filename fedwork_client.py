import pickle
import sys
import xml.etree.ElementTree as ET
from core.Client import Client
from arch.arch import ActivationFunction, BaseArch, FWArch
from utils.common import IpAddr
import utils.consts as const
import utils.logger as util
from utils.profiler import *
import os
import xmltodict
import torch.nn as nn
import dataset.dataset as DS
from core.Server import *
import torch.optim as optim

class fedwork_client:
    def __init__(self):
        self.local_clients = []
        logger().set_log_type(logger_log_type.logger_type_debug.value |
                    logger_log_type.logger_type_error.value |
                    logger_log_type.logger_type_info.value |
                    logger_log_type.logger_type_normal.value |
                    logger_log_type.logger_type_warning.value)


    def get_var(self, var_list, name, type, def_val):
        if not isinstance(var_list, list):
            var_list = [var_list]
        try:
            desired_item_value = next((var for var in var_list if var.get("@name") == name), None)["#text"]
            return type(desired_item_value)
        except TypeError as e:
            return def_val

    def create_datasets(self, dataset_cfg, num_of_nodes, output_dir = const.OUTPUT_DIR):

        dir_path = os.path.join(output_dir, f"dataset")
        if os.path.exists(dir_path):
            dataset_train_list = []
            file_counter = 0
            while True:
                file_path = os.path.join(dir_path, f"dataset_node_{file_counter}.ds")
                file_counter += 1
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        dataset_train_list.append(pickle.loads(f.read()))
                else:
                    file_path = os.path.join(dir_path, f"dataset_test.ds")
                    
                    if not os.path.exists(file_path):
                        break
                    with open(file_path, 'rb') as f:
                        data = f.read()
                        dataset_test = pickle.loads(data)
                    return dataset_train_list, dataset_test

        vars = dataset_cfg["var"]
        heterogeneous = False#self.get_var(vars, "heterogeneous", bool, False)
        non_iid_level = self.get_var(vars, "non_iid_level", float, 0.5)
        train_batch_size = self.get_var(vars, "train_batch_size", int, 128)
        test_batch_size = self.get_var(vars, "test_batch_size", int, 128)
        num_workers = self.get_var(vars, "num_workers", int, 1)
        save_graph = self.get_var(vars, "save_graph", bool, True)
        enclose_info = False#self.get_var(vars, "enclosed_info", bool, False)


        dataset_train_list, dataset_test = DS.create_datasets(num_of_nodes, dataset_cfg["@type"], heterogeneous, non_iid_level, train_batch_size, test_batch_size, num_workers, save_graph, enclose_info, dir_path)


        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        file_counter = 0
        for dataset in dataset_train_list:
            data = pickle.dumps(dataset)
            file_path = os.path.join(dir_path, f"dataset_node_{file_counter}.ds")
            file_counter += 1
            with open(file_path, 'wb') as f:
                f.write(data)
        
        file_path = os.path.join(dir_path, f"dataset_test.ds")
        data = pickle.dumps(dataset_test)
        with open(file_path, 'wb') as f:
            f.write(data)

        return dataset_train_list, dataset_test
        
    def get_loss_function(self, loss_name):
        loss_functions = {
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "BCELoss": nn.BCELoss,
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
            "MSELoss": nn.MSELoss,
            "L1Loss": nn.L1Loss,
            "SmoothL1Loss": nn.SmoothL1Loss,
            "NLLLoss": nn.NLLLoss,
            "PoissonNLLLoss": nn.PoissonNLLLoss,
            "KLDivLoss": nn.KLDivLoss,
            "MarginRankingLoss": nn.MarginRankingLoss,
            "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
            "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
            "SoftMarginLoss": nn.SoftMarginLoss,
            "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
            "CTCLoss": nn.CTCLoss,
        }

        if loss_name in loss_functions:
            return loss_functions[loss_name]
        else:
            raise ValueError("Invalid loss function name")
    
    def get_activation_function(self, loss_function):
        for activation_fn in ActivationFunction:
            if activation_fn.value == loss_function:
                return activation_fn
            
        return None

    def get_optimizer_class(self, optimizer_name):
        optimizers = {
            "SGD": optim.SGD,
            "Adam": optim.Adam,
            "Adagrad": optim.Adagrad,
            "RMSprop": optim.RMSprop,
            "Adadelta": optim.Adadelta,
            "AdamW": optim.AdamW,
            "SparseAdam": optim.SparseAdam,
            # Add more optimizers here as needed
        }

        if optimizer_name in optimizers:
            return optimizers[optimizer_name]
        else:
            raise ValueError("Invalid optimizer name")
    
    def load_method(self, code_string, class_name, args):
        namespace = {}
        exec(code_string, namespace)
        class_obj = namespace.get(class_name)

        if class_obj is None:
            raise ValueError(f"Class '{class_name}' not found in the provided code string.")

        instance = class_obj(args)

        return instance

    def run(self):

        if not os.path.exists(const.OUTPUT_DIR):
            os.mkdir(const.OUTPUT_DIR)

        logger().set_stdout(logger_stdout_type.logger_stdout_console.value |
                logger_stdout_type.logger_stdout_file.value)

        train_dataset_list, _ = DS.create_datasets(1, "CIFAR10", False, 0.1, 128, 256, 8, True, False, ".")


        method_path = os.path.join("methods", "FedPoll2.py")

        with open(method_path, "rb") as f:
            method_class = f.read()

        arch = FWArch("ResNet18")
        arch.Build()

        optimizer = self.get_optimizer_class("SGD")
        loss_func = self.get_loss_function("CrossEntropyLoss")

        model = arch.CreateModel().to("cpu")
        method_obj = self.load_method(method_class, "FedPoll2", (3, 0, 0, "cpu", None))
        
        new_client = Client(f"III73", IpAddr("192.168.166.181", 23335), TrainingHyperParameters(0.001, 0.9, 1e-7), train_dataset_list[0], model, optimizer, loss_func, method_obj, "cpu")
        self.local_clients.append(new_client)


    def get_config(self, file_name):

        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                return file.read()
            
        config_path = os.path.join("configs", file_name)
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return file.read()
        return None

    def start(self):
        self.run()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Configuration file was not determined!\nUse: fedwork.py configuration_xml_file")
        exit()

    fedwork_ins = fedwork_client()
    fedwork_ins.start()
