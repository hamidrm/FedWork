import pickle
import xml.etree.ElementTree as ET
import utils.consts as const
import utils.logger as util
import os
import xmltodict
import torch.nn as nn
import dataset.dataset as DS
#from core.Server import *

class fedwork:
    def __init__(self):
        pass

    def get_var(self, var_list, name, type, def_val):
        if not isinstance(var_list, list):
            var_list = [var_list]
        try:
            desired_item_value = next((var for var in var_list if var.get("@name") == name), None)["#text"]
            return type(desired_item_value)
        except TypeError as e:
            return def_val

    def create_datasets(self, dataset_cfg, num_of_nodes):

        dir_path = os.path.join(const.OUTPUT_DIR, f"dataset")
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
        heterogeneous = self.get_var(vars, "heterogeneous", bool, False)
        non_iid_level = self.get_var(vars, "non_iid_level", float, 0.5)
        train_batch_size = self.get_var(vars, "train_batch_size", int, 128)
        test_batch_size = self.get_var(vars, "test_batch_size", int, 128)
        num_workers = self.get_var(vars, "num_workers", int, 8)
        save_graph = self.get_var(vars, "save_graph", bool, True)
        enclose_info = self.get_var(vars, "enclose_info", bool, False)


        dataset_train_list, dataset_test = DS.create_datasets(num_of_nodes, dataset_cfg["@type"], heterogeneous, non_iid_level, train_batch_size, test_batch_size, num_workers, save_graph, enclose_info)


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

    def get_loss_function(loss_name):
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
        
    def run(self, config_text):

        dict_cfg = xmltodict.parse(config_text)
        fedwork_cfg = dict_cfg.get("fedwork_cfg")
        if fedwork_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'fedwork' tag is not found.")
            return
        
        dataset_cfg = fedwork_cfg.get("dataset")
        if dataset_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'dataset' is not found.")
            return
        
        methods_cfg = fedwork_cfg.get("method")
        if methods_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'method' is not found.")
            return
        
        report_cfg = fedwork_cfg.get("report")
        if report_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'report' is not found.")
            return
        
        train_dataset_list, test_dataset = self.create_datasets(dataset_cfg, int(fedwork_cfg["@num_of_nodes"]))
        
        attr_net_port = "@net_port"
        def_net_port = "12345"
        net_port = int(fedwork_cfg[attr_net_port] if attr_net_port in fedwork_cfg.keys() else def_net_port)

        attr_net_ip = "@net_ip"
        def_net_ip = "127.0.0.1"
        net_ip = fedwork_cfg[attr_net_ip] if attr_net_ip in fedwork_cfg.keys() else def_net_ip

        attr_eval_criterion = "@eval_criterion"
        def_eval_criterion = "CrossEntropyLoss"
        eval_criterion = fedwork_cfg[attr_eval_criterion] if attr_eval_criterion in fedwork_cfg.keys() else def_eval_criterion
        

        # For each method, we have to execute federated learning according to corresponding configuration

        for method in methods_cfg:
            attr_method_type = "@type"
            if not attr_method_type in method.keys():
                util.logger.log_error(f"Method type is not determined!")
                continue

            method_type = method[attr_method_type]

            attr_method_platform = "@platform"
            attr_method_platform_def = "cpu"
            method_platform = method[attr_method_platform] if attr_method_platform in method.keys() else attr_method_platform_def

            method_num_of_epochs = self.get_var(method["var"], "epochs_num", int, 5)

            method_path = os.path.join("methods", f"{method_type}.py")
            if not os.path.exists(method_path):
                util.logger.log_error(f"Method type '{method_type}' is not available!")
                break

            
        # arch = FWArch(BaseArch.FeedForwardNet1)
        # model = arch.CreateModel()

        # loss_func = self.get_loss_function(eval_criterion)
        # server = Server(IpAddr(net_ip, net_port), fedavg, test_dataset, model, loss_func, "cpu")
        # clients_list = []

        # create_local_clients(train_ds_list, clients_list)
        # server.start_training()


    def get_config(self, file_name):

        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                return file.read()
            
        config_path = os.path.join("configs", file_name)
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return file.read()
        return None

    def start(self, config):
        config_text = self.get_config(config)
        if config_text is None:
            util.logger.log_error(f"Unable to find config file! '{config}' is unavailable!")
            return
        
        util.logger.log_debug(f"File '{config}' was found.")
        self.run(config_text)


fedwork_ins = fedwork()
fedwork_ins.start("config_fedavg.xml")
