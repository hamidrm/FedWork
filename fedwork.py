import pickle
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


class fedwork:
    def __init__(self):
        self.local_clients = []
        logger().set_log_type(logger_log_type.logger_type_debug.value |
                    logger_log_type.logger_type_error.value |
                    logger_log_type.logger_type_info.value |
                    logger_log_type.logger_type_normal.value |
                    logger_log_type.logger_type_warning.value)

        logger().set_stdout(logger_stdout_type.logger_stdout_console.value |
                        logger_stdout_type.logger_stdout_file.value |
                        logger_stdout_type.logger_stdout_network.value |
                        logger_stdout_type.logger_stdout_stringio.value)

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
        enclose_info = self.get_var(vars, "enclosed_info", bool, False)


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


    def load_method(self, code_string, class_name):
        namespace = {}
        exec(code_string, namespace)
        class_obj = namespace.get(class_name)

        if class_obj is None:
            raise ValueError(f"Class '{class_name}' not found in the provided code string.")

        instance = class_obj()

        return instance

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
        
        #Step 1.
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
        
        sample_image, _ = test_dataset.dataset[0]
        ds_channels, ds_height, ds_width = sample_image.size()
        ds_num_classes = len(test_dataset.dataset.classes)


        # Step 2.
        # For each method, we have to execute federated learning according to corresponding configuration
        for method in methods_cfg:
        
            attr_method_type = "@type"

            if not attr_method_type in method.keys():
                util.logger.log_error(f"Method type is not determined!")
                continue

            method_type = method[attr_method_type]

            time_probes_path = os.path.join(const.OUTPUT_DIR, f"{method_type}_time_probes_data.data")
            var_probes_path = os.path.join(const.OUTPUT_DIR, f"{method_type}_var_probes_data.data")

            if os.path.exists(time_probes_path) and os.path.exists(var_probes_path):
                util.logger.log_info(f"Information of method '{method_type}' is alreay ready!")
                continue


            attr_method_platform = "@platform"
            attr_method_platform_def = "cpu"
            method_platform = method[attr_method_platform] if attr_method_platform in method.keys() else attr_method_platform_def

            method_num_of_epochs = self.get_var(method["var"], "epochs_num", int, 5)

            method_path = os.path.join("methods", f"{method_type}.py")
            if not os.path.exists(method_path):
                util.logger.log_error(f"Method type '{method_type}' is not available!")
                continue

            with open(method_path, "rb") as f:
                method_class = f.read()
            
            method_obj = self.load_method(method_class, method_type)

            arch_cfg = method["arch"]
            arch_cfg_vars = arch_cfg["var"]

            attr_arch_type = "@type"
            if not attr_arch_type in arch_cfg.keys():
                util.logger.log_error(f"Type of architecture was not determined in method '{method_type}'!")
                break
            
            arch_type = None
            arch_type_str = arch_cfg[attr_arch_type]
            for _arch_type in BaseArch:
                if _arch_type.value == arch_type_str:
                    arch_type = _arch_type
            
            if arch_type == None:
                util.logger.log_error(f"In method '{method_type}', the type of architecture({arch_type_str}) is not defined!")
                break
         

            arch_input_nodes = self.get_var(arch_cfg_vars, "NumberOfInputNodes", int, ds_channels * ds_height * ds_width)
            arch_output_nodes = self.get_var(arch_cfg_vars, "NumberOfOutputNodes", int, ds_num_classes)
            
            arch = FWArch(arch_type)

            arch.SetParameter("NumberOfInputNodes", arch_input_nodes)
            arch.SetParameter("NumberOfOutputNodes", arch_output_nodes)

            vars_list = arch.get_var_list()

            for var in arch_cfg_vars:
                attr_name_key = "@name"
                var_value_key = "#text"
                
                if not attr_name_key in var.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', var name is not available!")
                    break
                if not var_value_key in var.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', var value is not available!")
                    break

                var_name = var[attr_name_key]
                var_text = var[var_value_key]


                if var_name in vars_list:
                    var_type = arch.get_var_type(var_name)
                    if var_type == "integer":
                        arch.SetParameter(var_name, int(var_text))
                    elif var_type == "act_fn":
                        arch.SetParameter(var_name, self.get_activation_function(var_text))
                    else:
                        util.logger.log_error(f"Unexpectedly error in type of the variable '{var_name}'!")
                        break
                
            arch.Build()
            global_model = arch.CreateModel()
            
            loss_func = self.get_loss_function(eval_criterion)
            server = Server(IpAddr(net_ip, net_port), method_obj, test_dataset, global_model, loss_func, method_platform)

            localclients_tag = "localclients"
            if localclients_tag in fedwork_cfg:
                localclients_cfg = fedwork_cfg[localclients_tag]

                attr_learning_rate = "@learning_rate"
                attr_momentum = "@momentum"
                attr_weight_decay = "@weight_decay"
                localclients_num_key = "#text"

                if not attr_learning_rate in localclients_cfg.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', attribute learning_rate not assigned!")
                    break

                if not attr_momentum in localclients_cfg.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', attribute momentum not assigned!")
                    break

                if not attr_weight_decay in localclients_cfg.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', attribute weight_decay not assigned!")
                    break

                if not localclients_num_key in localclients_cfg.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', value is not assigned!")
                    break

                learning_rate = float(localclients_cfg[attr_learning_rate])
                momentum = float(localclients_cfg[attr_momentum])
                weight_decay = float(localclients_cfg[attr_weight_decay])
                localclients_num = int(localclients_cfg[localclients_num_key])

                if localclients_num > len(train_dataset_list):
                    util.logger.log_warning(f"Local clients number must not be greater the total nodes number! Local clients number will be assumed {len(train_dataset_list)}")
                    localclients_num = len(train_dataset_list)
                    break
                
                if localclients_num != 0:
                    for client_id in range(localclients_num):
                        model = arch.CreateModel()
                        new_client = Client(f"Client{client_id}", IpAddr(net_ip, net_port), TrainingHyperParameters(learning_rate, momentum, weight_decay), train_dataset_list[client_id], model, optim.SGD, loss_func, "cpu")
                        self.local_clients.append(new_client)

            server.start_training(method_num_of_epochs)
            server.wait_for_method()

            time_probs = profiler.dump_time_probes()
            var_probs = profiler.dump_var_probes()
            
            time_probs_binary = pickle.loads(time_probs)
            var_probs_binary = pickle.loads(var_probs)
            
            with open(time_probes_path, "wb") as f:
                f.write(time_probs_binary)

            with open(var_probes_path, "wb") as f:
                f.write(var_probs_binary)
        # Step 3.
        





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
