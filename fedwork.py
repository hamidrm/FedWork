import os, math, json, hashlib, random, argparse
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS; set before torch import
os.environ["OMP_NUM_THREADS"] = "1"; 
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import sys
import xml.etree.ElementTree as ET
from core.Client import Client
from core.ClientExp import ClientExp
from arch.arch import ActivationFunction, BaseArch, FWArch
from utils.common import IpAddr
import utils.consts as const
import utils.logger as util
from utils.profiler import *
from utils.common import *
import os
import xmltodict
import torch.nn as nn
import numpy as np
import dataset.dataset as DS
from core.Server import *
import torch.optim as optim
from utils.plotter import Plotter
from methods.FedDrop import FedDrop
    
class fedwork:
    def __init__(self):
        self.local_clients = {}
        self.plotter = Plotter()
        self.fl_context = {}
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
            if type == bool:
                # Convert value to lowercase and check common string representations of true and false
                if desired_item_value.lower() in ['true', '1', 'yes']:
                    return True
                elif desired_item_value.lower() in ['false', '0', 'no']:
                    return False
                else:
                    raise ValueError("Cannot convert the variable to bool.")
            else:
                return type(desired_item_value)
        except TypeError as e:
            return def_val

    def create_datasets(self, dataset_cfg, num_of_nodes, output_dir = const.OUTPUT_DIR, seed = 42):

        dir_path = os.path.join(output_dir, "dataset")
        self.fl_context["dataset_path"] = dir_path

        # 1) read config vars first
        vars = dataset_cfg["var"]
        non_iid_level   = self.get_var(vars, "non_iid_level", float, 0.5)
        non_iid_alpha   = self.get_var(vars, "alpha", float, sys.float_info.min)
        train_batch_size= self.get_var(vars, "train_batch_size", int, 128)
        test_batch_size = self.get_var(vars, "test_batch_size", int, 128)
        num_workers = self.get_var(vars, "num_workers", int, 0)
        save_graph      = self.get_var(vars, "save_graph", bool, True)
        enclose_info    = self.get_var(vars, "enclosed_info", bool, False)
        use_dirichlet   = self.get_var(vars, "dirichlet", bool, False)
        if non_iid_alpha != sys.float_info.min:
            non_iid_level = non_iid_alpha

        # 2) try to load cached partitions
        partitions = []
        i = 0
        while True:
            fpath = os.path.join(dir_path, f"dataset_node_{i}.ds")
            if not os.path.exists(fpath): break
            with open(fpath, 'rb') as f:
                partitions.append(pickle.load(f))
            i += 1

        if partitions:
            dataset_train_list, dataset_test = DS.build_loaders_from_partitions(
                partitions, dataset_cfg["@type"], train_batch_size, test_batch_size, base_seed=seed, num_workers=num_workers
            )
            self.fl_context["dataset_train_list"] = dataset_train_list
            self.fl_context["dataset_train_test"] = dataset_test
            return dataset_train_list, dataset_test

        # 3) build fresh, save partitions
        dataset_train_list, dataset_test, partitions = DS.create_datasets(
            num_of_nodes, dataset_cfg["@type"], non_iid_level,
            train_batch_size, test_batch_size, use_dirichlet, num_workers, save_graph, enclose_info, dir_path, seed
        )
        os.makedirs(dir_path, exist_ok=True)
        for i, idxs in enumerate(partitions):
            with open(os.path.join(dir_path, f"dataset_node_{i}.ds"), 'wb') as f:
                pickle.dump(idxs, f)
        with open(os.path.join(dir_path, "dataset_meta.pkl"), "wb") as f:
            pickle.dump({"type": dataset_cfg["@type"]}, f)

        self.fl_context["dataset_train_list"] = dataset_train_list
        self.fl_context["dataset_train_test"] = dataset_test
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
    
    def load_method(self, code_string, class_name, args):
        namespace = {}
        exec(code_string, namespace)
        class_obj = namespace.get(class_name)

        if class_obj is None:
            raise ValueError(f"Class '{class_name}' not found in the provided code string.")

        instance = class_obj(*tuple(args))

        return instance

    def run(self, config_text):

        

        #TODO-make it configurable through xml config
        

        if not os.path.exists(const.OUTPUT_DIR):
            os.mkdir(const.OUTPUT_DIR)
        self.fl_context["config_str"] = config_text
        dict_cfg = xmltodict.parse(config_text)
        fedwork_cfg = dict_cfg.get("fedwork_cfg")
        if fedwork_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'fedwork' tag is not found.")
            return
        
        attr_seed = "@seed_mode"
        attr_seed_def = 42
        seed_value = fedwork_cfg[attr_seed] if attr_seed in fedwork_cfg.keys() else attr_seed_def
        self.fl_context["seed"] = seed_value
        Common.set_seed_over_everything(seed_value)
        
        self.fl_context["config_xml_fedwork"] = fedwork_cfg
        cfg_name = fedwork_cfg.get("@name")
        if cfg_name is None:
            util.logger.log_error("Invalid config file! A specific name have to be assigned to the configuration.")
            return
        
        output_path = os.path.join(const.OUTPUT_DIR, cfg_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        logger().set_file_path(output_path)
        dataset_cfg = fedwork_cfg.get("dataset")
        if dataset_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'dataset' is not found.")
            return
        self.fl_context["config_xml_dataset"] = dataset_cfg
        methods_cfg = fedwork_cfg.get("method")
        if methods_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'method' is not found.")
            return
        self.fl_context["config_xml_method"] = methods_cfg
        report_cfg = fedwork_cfg.get("report")
        if report_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'report' is not found.")
            return
        self.fl_context["config_xml_report"] = report_cfg
        attr_save_log = "@save_log"
        save_log_def = "True"
        save_log = bool(report_cfg[attr_save_log] if attr_save_log in report_cfg.keys() else save_log_def)

        attr_exp_mode = "@experiment_mode"
        attr_exp_mode_def = "false"
        exp_mode = fedwork_cfg[attr_exp_mode] if attr_exp_mode in fedwork_cfg.keys() else attr_exp_mode_def
        self.fl_context["experiment_mode"] = exp_mode
        
        attr_lon = "@log_over_net"
        lon_opt = None
        if attr_lon in report_cfg.keys():
            lon_opt_temp = report_cfg[attr_lon].split(",")
            lon_opt = lon_opt_temp[0], int(lon_opt_temp[1])
            logger().set_server(utils.common.IpAddr(*lon_opt))
        
        logger().set_stdout(logger_stdout_type.logger_stdout_console.value |
                (logger_stdout_type.logger_stdout_file.value if save_log == True else 0) |
                (logger_stdout_type.logger_stdout_network.value if lon_opt != None else 0))
        
        # Step 1.
        # Generate datasets
        train_dataset_list, test_dataset = self.create_datasets(dataset_cfg, int(fedwork_cfg["@num_of_nodes"]), output_path, seed_value)
        
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

        num_of_rounds = int(fedwork_cfg["@num_of_rounds"])
        self.fl_context["num_of_rounds"] = num_of_rounds
        # Step 2.
        # For each method, we have to execute federated learning according to corresponding configuration
        probes_bin = {}
        
        methods_cfg = methods_cfg if isinstance(methods_cfg, list) else [methods_cfg]
        weights = [(float(len(dataloader.dataset)) / float(sum([len(dataloader.dataset) for  dataloader in train_dataset_list]))) for dataloader in train_dataset_list]
        self.fl_context["dataset_weights"] = weights
        
        self.fl_context["methods_list"] = {}
        for method in methods_cfg:
            self.local_clients = {}
            profiler.reset_profiles()
            attr_method_type = "@type"
            attr_method_name = "@name"
            if not attr_method_type in method.keys():
                util.logger.log_error(f"Method type is not determined!")
                continue

            method_type = method[attr_method_type]
            method_name = method[attr_method_name]

            self.fl_context["methods_list"][method_name] = {}

            self.fl_context["methods_list"][method_name]["type"] = method_type
            probes_data_path = os.path.join(output_path, f"{method_name}_probes_data.data")
            self.fl_context["methods_list"][method_name]["output_path"] = probes_data_path
            if os.path.exists(probes_data_path):
                util.logger.log_info(f"Information for method '{method_name}(type={method_type})' has been found!")

                if os.path.exists(probes_data_path):
                    with open(probes_data_path, "rb") as f:
                        probes_bin[method_name] = f.read()
                else:
                    probes_bin[method_name] = None

                continue

            Common.set_seed_over_method(seed_value)
            train_dataset_list, test_dataset = self.create_datasets(dataset_cfg, int(fedwork_cfg["@num_of_nodes"]), output_path)
            
            attr_method_platform = "@platform"
            attr_method_platform_def = "cpu"
            method_platform = method[attr_method_platform] if attr_method_platform in method.keys() else attr_method_platform_def

            attr_method_seed = "@seed"
            attr_method_seed_def = seed_value
            method_seed = method[attr_method_seed] if attr_method_seed in method.keys() else attr_method_seed_def

            Common.set_seed_over_method(method_seed)
            method_num_of_epochs = self.get_var(method["var"], "epochs_num", int, 5)
            self.fl_context["methods_list"][method_name]["num_of_epochs"] = method_num_of_epochs
            method_args = self.get_var(method["var"], "args", str, "")

            method_path = os.path.join("methods", f"{method_type}.py")
            if not os.path.exists(method_path):
                util.logger.log_error(f"Method type '{method_type}' is not available!")
                continue

            with open(method_path, "rb") as f:
                method_class = f.read()

            arch_cfg = method["arch"]

            tag_var = "var"
            arch_cfg_vars = None
            if tag_var in arch_cfg:
                arch_cfg_vars = arch_cfg["var"]

            attr_arch_type = "@type"
            if not attr_arch_type in arch_cfg.keys():
                util.logger.log_error(f"Type of architecture was not determined in method '{method_type}'!")
                break
            
            arch_type = None
            arch_type_str = arch_cfg[attr_arch_type]
            self.fl_context["methods_list"][method_name]["arch_type"] = arch_type_str
            for _arch_type in BaseArch:
                if _arch_type.value == arch_type_str:
                    arch_type = _arch_type
            
            if arch_type == None:
                util.logger.log_error(f"In method '{method_type}', the type of architecture({arch_type_str}) is not defined!")
                break
         
            if arch_cfg_vars is not None:
                arch_input_nodes = self.get_var(arch_cfg_vars, "NumberOfInputNodes", int, ds_channels * ds_height * ds_width)
                arch_output_nodes = self.get_var(arch_cfg_vars, "NumberOfOutputNodes", int, ds_num_classes)
            else:
                arch_input_nodes = ds_channels * ds_height * ds_width
                arch_output_nodes = ds_num_classes

            arch = FWArch(arch_type)

            arch.SetParameter("NumberOfInputNodes", arch_input_nodes)
            arch.SetParameter("NumberOfOutputNodes", arch_output_nodes)

            vars_list = arch.get_var_list()

            if arch_cfg_vars is not None:
                if type(arch_cfg_vars) == dict:
                    arch_cfg_vars = [arch_cfg_vars]
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
                        arch.SetParameter(var_name, var_text)
        
            
            msg = arch.Build()

            if msg != '':
                util.logger.log_error(f"Model Architecture Error: '{msg}'")
                break
        
            global_model = arch.CreateModel().to(method_platform)
            self.fl_context["methods_list"][method_name]["arch"] = arch.get_class()

            loss_func = self.get_loss_function(eval_criterion)
            self.fl_context["methods_list"][method_name]["loss_func"] = loss_func
            self.fl_context["methods_list"][method_name]["platform"] = method_platform

            # Load the method for Server-side requests
            method_obj = self.load_method(method_class, method_type, (method_name, self.fl_context, method_args))#FedDrop(method_name, self.fl_context, method_args)#

            server = Server(IpAddr(net_ip, net_port), method_obj, test_dataset, global_model, loss_func, method_platform, exp_mode)

            localclients_tag = "localclients"
            if localclients_tag in fedwork_cfg:
                localclients_cfg = fedwork_cfg[localclients_tag]

                attr_learning_rate = "@learning_rate"
                attr_momentum = "@momentum"
                attr_weight_decay = "@weight_decay"
                attr_optimizer = "@optimizer"
                attr_platform = "@platform"
                localclients_num_key = "#text"

                momentum = None
                weight_decay = None

                self.fl_context["methods_list"][method_name]["localclients_xml"] = localclients_cfg
                if not attr_learning_rate in localclients_cfg.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', attribute learning_rate is not assigned!")
                    break

                if attr_momentum in localclients_cfg.keys():
                    momentum = float(localclients_cfg[attr_momentum])

                if not attr_optimizer in localclients_cfg.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', attribute optimizer is not assigned!")
                    break

                if attr_weight_decay in localclients_cfg.keys():
                    weight_decay = float(localclients_cfg[attr_weight_decay])

                if not localclients_num_key in localclients_cfg.keys():
                    util.logger.log_error(f"In method '{method_type}', architecture '{arch_type_str}', value is not assigned!")
                    break

                learning_rate = float(localclients_cfg[attr_learning_rate])
                self.fl_context["methods_list"][method_name]["learning_rate"] = learning_rate
                localclients_num = int(localclients_cfg[localclients_num_key])
                self.fl_context["methods_list"][method_name]["localclients_num"] = localclients_num
                client_platform = localclients_cfg[attr_platform]
                self.fl_context["clients_platform"] = client_platform
                optimizer = Common.get_optimizer_class(localclients_cfg[attr_optimizer])
                self.fl_context["methods_list"][method_name]["optimizer_class"] = optimizer
                if localclients_num > len(train_dataset_list):
                    util.logger.log_warning(f"Local clients number must not be greater the total nodes number! Local clients number will be assumed {len(train_dataset_list)}")
                    localclients_num = len(train_dataset_list)
                    break
                if localclients_num != 0:
                    if exp_mode:
                        for client_id in range(localclients_num):
                            model = arch.CreateModel().to(method_platform)

                            # Load the method for Client-side requests
                            method_obj = self.load_method(method_class, method_type, (method_name, self.fl_context, method_args)) #FedDrop(method_name, self.fl_context, method_args)#
                            
                            new_client = ClientExp(f"Client{client_id}", client_id, server, TrainingHyperParameters(learning_rate, momentum, weight_decay), train_dataset_list[client_id], model, optimizer, loss_func, method_obj, client_platform)
                            self.local_clients[f"Client{client_id}"] = new_client
                        
                        server.setClientSetExp(self.local_clients)
                    else:
                        for client_id in range(localclients_num):
                            model = arch.CreateModel().to(method_platform)

                            # Load the method for Client-side requests
                            method_obj = FedDrop(method_name, self.fl_context, method_args) #self.load_method(method_class, method_type, (method_name, self.fl_context, method_args)) #
                            
                            new_client = Client(f"Client{client_id}", client_id, IpAddr(net_ip, net_port), TrainingHyperParameters(learning_rate, momentum, weight_decay), train_dataset_list[client_id], model, optimizer, loss_func, method_obj, client_platform)
                            self.local_clients[f"Client{client_id}"] = new_client

            server.start_training()
            server.wait_for_method()

            probes = profiler.dump_probes()
            
            if len(probes) != 0:
                method_info = {}
                method_info["cfg_name"] = cfg_name
                method_info["ds_type"] = dataset_cfg["@type"]
                method_info["ds_vars"] = dataset_cfg["var"]
                method_info["method_name"] = method_name
                method_info["method_args"] = method_args
                method_info["method_type"] = method_type
                method_info["method_platform"] = method_platform
                method_info["method_class"] = method_class
                method_info["method_path"] = method_path
                method_info["method_num_of_epochs"] = method_num_of_epochs
                method_info["arch_type"] = arch_type_str
                method_info["arch_cfg_vars"] = arch_cfg_vars

                probes["method_info"] = method_info
                probes_bin[method_name] = pickle.dumps(probes)
                with open(probes_data_path, "wb") as f:
                    f.write(probes_bin[method_name])
            else:
                probes_bin[method_name] = None

            server.release_all()




        
        # Step 3.
        # Generate repoorts

        fig_pf_tag = "fig:pf"
        fig_hv_tag = "fig:hv"
        fig_avg_tag = "fig:avg"

        if fig_avg_tag in report_cfg:
            figs_cfg = report_cfg[fig_avg_tag]

            if not isinstance(figs_cfg, list):
                figs_cfg = [figs_cfg]
                
            for fig in figs_cfg:
                
                attr_name = "@name"
                attr_x_axis = "@x_axis"
                attr_y_axis = "@y_axis"
                attr_methods = "@methods"
                attr_caption = "@caption"
                attr_labels = "@labels"
                attr_x_axis_title = "@x_axis_title"
                attr_y_axis_title = "@y_axis_title"
                attr_x_axis_scale = "@x_axis_scale"
                attr_y_axis_scale = "@y_axis_scale"
                attr_style = "@style"
                attr_senses = "@senses"
                fig_caption = ""

                if not attr_name in fig.keys():
                    util.logger.log_error(f"Figures should have a name attribute!")
                    break

                if not attr_x_axis in fig.keys():
                    x_axis = "Round"
                else:
                    x_axis = fig["@x_axis"]

                if not attr_y_axis in fig.keys():
                    util.logger.log_error(f"Figure '{attr_name}' should have a y_axis attribute!")
                    break

            
                if not attr_methods in fig.keys():
                    util.logger.log_error(f"Figure '{attr_name}' should have a methods attribute!")
                    break



                name = fig[attr_name]
                y_axis = fig[attr_y_axis]
                methods = str(fig[attr_methods]).split(",")

                if not attr_caption in fig.keys():
                    fig_caption = name
                else:
                    fig_caption = fig[attr_caption]

                if not attr_style in fig.keys():
                    style = ""
                else:
                    style = fig[attr_style]

                x_axis_scale = 1.0
                if attr_x_axis_scale in fig.keys():
                    x_axis_scale = float(fig[attr_x_axis_scale])

                y_axis_scale = 1.0
                if attr_y_axis_scale in fig.keys():
                    y_axis_scale = float(fig[attr_y_axis_scale])

                y_labels = None
                if attr_labels in fig.keys():
                    y_labels = str(fig[attr_labels]).split(",")
                
                plot_index = 0
                methods_groups = {}
                for method in methods:
                    method_group = "_".join(method.split("_")[:-1])
                    if method_group in methods_groups.keys():
                        methods_groups[method_group].append(method)
                    else:
                        methods_groups[method_group] = []
                        methods_groups[method_group].append(method)
                
                self.plotter.plot_begin(style_str=style)
                
                
                for group_name, method_group in methods_groups:
                    x_axis_data = []
                    y_axis_data = []
                    for method in methods:

                        if not method in probes_bin:
                            util.logger.log_error(f"Needed method(s) for figure '{name}' was not found!")
                            break
                
                        probes = pickle.loads(probes_bin[method])
                        probes_times_prof = probes["time_profiles"]
                        probes_vars = probes["var_values"]
                        probes_var_changes = probes["var_changes"]

                        y_axis_params = str(fig[attr_y_axis]).split(",")
                        
                        for y_axis in y_axis_params:
                            if y_axis in probes_times_prof:
                                fig_data = probes_times_prof[y_axis]
                            elif y_axis in probes_vars:
                                fig_data = probes_vars[y_axis]
                            elif y_axis in probes_var_changes:
                                fig_data = probes_var_changes[y_axis]
                            else:
                                util.logger.log_error(f"Expected y_axis for figure '{name}' was not found!")
                                break
                            
                            y = []
                            x = []
                            if not attr_x_axis_range in fig.keys():
                                if x_axis == "round":
                                    x = [fig_data_elem[1] for fig_data_elem in fig_data]
                                elif x_axis == "time":
                                    x = [(fig_data_elem[0] - fig_data[0][0]) for fig_data_elem in fig_data]
                                else:
                                    util.logger.log_error(f"'{x_axis}' does not defined for figure '{name}' was not found!")
                                    break
                                y = [fig_data_elem[2] for fig_data_elem in fig_data]
                            else:
                                x_range_str = fig[attr_x_axis_range]
                                x_range = str.split(x_range_str, ",")
                                x_range_start = float(x_range[0])
                                x_range_end = 0
                                if x_axis == "round":
                                    x_range_end = float(x_range[1]) if float(x_range[1]) != -1 else max(fig_data[:][1])

                                    for i in range(len(fig_data)):
                                        if fig_data[i][1] >= x_range_start and fig_data[i][1] <= x_range_end:
                                            x.append(fig_data[i][1])
                                            y.append(fig_data[i][2])
                                    
                                elif x_axis == "time":
                                    x_range_end = float(x_range[1]) if float(x_range[1]) != -1 else max(fig_data[:][0])
                                    for i in range(len(fig_data)):
                                        if (fig_data[i][0] - fig_data[0][0]) >= x_range_start and (fig_data[i][0] - fig_data[0][0]) <= x_range_end:
                                            x.append(fig_data[i][0])
                                            y.append(fig_data[i][2])
                                else:
                                    util.logger.log_error(f"'{x_axis}' does not defined for figure '{name}' was not found!")
                                    break
                            

                            if len(x_axis_data) == 0:
                                x_axis_data = [x_v * x_axis_scale for x_v in x]
                            y = [y_v * y_axis_scale for y_v in y]

                            if y_labels:
                                ylabel=y_labels[plot_index]
                            elif len(y_axis_params) == 1:
                                ylabel=method
                            else:
                                ylabel=f"{method}.{y_axis}"
                            
                            y_axis_data.append(y)
                            plot_index += 1
                    
                    self.plotter.plot_envelope(x_axis_data, y_axis_data, ylabel, style, plot_index)
                x_axis_title = x_axis
                y_axis_title = y_axis

                if attr_x_axis_title in fig.keys():
                    x_axis_title = fig[attr_x_axis_title]

                if attr_y_axis_title in fig.keys():
                    y_axis_title = fig[attr_y_axis_title]

                figure_path = os.path.join(output_path, f'{name}.pdf')
                self.plotter.plot_end(x_axis_title, y_axis_title, fig_caption, style, figure_path)
                
        if fig_pf_tag in report_cfg:
            figs_cfg = report_cfg[fig_pf_tag]

            if not isinstance(figs_cfg, list):
                figs_cfg = [figs_cfg]
                
            for fig in figs_cfg:
                
                attr_name = "@name"
                attr_x_axis = "@x_axis"
                attr_y_axis = "@y_axis"
                attr_methods = "@methods"
                attr_caption = "@caption"
                attr_labels = "@labels"
                attr_x_axis_title = "@x_axis_title"
                attr_y_axis_title = "@y_axis_title"
                attr_x_axis_scale = "@x_axis_scale"
                attr_y_axis_scale = "@y_axis_scale"
                attr_style = "@style"
                attr_senses = "@senses"
                attr_x_count = "@x_count"
                fig_caption = ""

                if not attr_name in fig.keys():
                    util.logger.log_error(f"Figures should have a name attribute!")
                    break

                if not attr_x_axis in fig.keys():
                    x_axis = "Round"
                else:
                    x_axis = fig["@x_axis"]

                if not attr_y_axis in fig.keys():
                    util.logger.log_error(f"Figure '{attr_name}' should have a y_axis attribute!")
                    break

            
                if not attr_methods in fig.keys():
                    util.logger.log_error(f"Figure '{attr_name}' should have a methods attribute!")
                    break



                name = fig[attr_name]
                y_axis = fig[attr_y_axis]
                methods = str(fig[attr_methods]).split(",")

                if not attr_caption in fig.keys():
                    fig_caption = name
                else:
                    fig_caption = fig[attr_caption]

                if not attr_style in fig.keys():
                    style = ""
                else:
                    style = fig[attr_style]

                if not attr_x_count in fig.keys():
                    x_count = -1
                else:
                    x_count = int(fig[attr_x_count])
                    
                
                x_axis_scale = 1.0
                if attr_x_axis_scale in fig.keys():
                    x_axis_scale = float(fig[attr_x_axis_scale])

                y_axis_scale = 1.0
                if attr_y_axis_scale in fig.keys():
                    y_axis_scale = float(fig[attr_y_axis_scale])

                y_labels = None
                if attr_labels in fig.keys():
                    y_labels = str(fig[attr_labels]).split(",")
                
                plot_index = 0

                self.plotter.plot_begin(style_str=style)
                
                for method in methods:

                    if not method in probes_bin:
                        util.logger.log_error(f"Needed method(s) for figure '{name}' was not found!")
                        break
            
                    probes = pickle.loads(probes_bin[method])
                    probes_times_prof = probes["time_profiles"]
                    probes_vars = probes["var_values"]
                    probes_var_changes = probes["var_changes"]

                    y_axis_params = str(fig[attr_y_axis]).split(",")
                    
                    for y_axis in y_axis_params:
                        if y_axis in probes_times_prof:
                            fig_data_y = probes_times_prof[y_axis]
                        elif y_axis in probes_vars:
                            fig_data_y = probes_vars[y_axis]
                        elif y_axis in probes_var_changes:
                            fig_data_y = probes_var_changes[y_axis]
                        else:
                            util.logger.log_error(f"Expected y_axis for figure '{name}' was not found!")
                            break
                        

                        if x_axis in probes_times_prof:
                            fig_data_x = probes_times_prof[x_axis]
                        elif x_axis in probes_vars:
                            fig_data_x = probes_vars[x_axis]
                        elif x_axis in probes_var_changes:
                            fig_data_x = probes_var_changes[x_axis]
                        else:
                            util.logger.log_error(f"Expected y_axis for figure '{name}' was not found!")
                            break


                        x = [fig_data_elem[2] for fig_data_elem in fig_data_x]
                        y = [fig_data_elem[2] for fig_data_elem in fig_data_y]
                        
                    
                        x = [x_v * x_axis_scale for x_v in x]
                        y = [y_v * y_axis_scale for y_v in y]

                        if y_labels:
                            ylabel=y_labels[plot_index]
                        elif len(y_axis_params) == 1:
                            ylabel=method
                        else:
                            ylabel=f"{method}.{y_axis}"
                        
                        reference_point = (1.0, 1.0)
                        self.plotter.plot_tradeoff_2d(x, y, ylabel, style, plot_index, ("min", "max"),show_points=False, number_of_rounds=x_count)
                        #self.plotter.plot_hypervolume2d(x, y, ylabel, reference_point, style, plot_index)
                        plot_index += 1
                

                x_axis_title = x_axis
                y_axis_title = y_axis

                if attr_x_axis_title in fig.keys():
                    x_axis_title = fig[attr_x_axis_title]

                if attr_y_axis_title in fig.keys():
                    y_axis_title = fig[attr_y_axis_title]

                figure_path = os.path.join(output_path, f'{name}.pdf')
                self.plotter.plot_end(x_axis_title, y_axis_title, fig_caption, style, figure_path)

        

        if fig_hv_tag in report_cfg:
            figs_cfg = report_cfg[fig_hv_tag]

            if not isinstance(figs_cfg, list):
                figs_cfg = [figs_cfg]
                
            for fig in figs_cfg:
                
                attr_name = "@name"
                attr_x_axis = "@x_axis"
                attr_y_axis = "@y_axis"
                attr_methods = "@methods"
                attr_caption = "@caption"
                attr_labels = "@labels"
                attr_x_axis_title = "@x_axis_title"
                attr_y_axis_title = "@y_axis_title"
                attr_x_axis_scale = "@x_axis_scale"
                attr_y_axis_scale = "@y_axis_scale"
                attr_style = "@style"
                fig_caption = ""

                if not attr_name in fig.keys():
                    util.logger.log_error(f"Figures should have a name attribute!")
                    break

                if not attr_x_axis in fig.keys():
                    x_axis = "Round"
                else:
                    x_axis = fig["@x_axis"]

                if not attr_y_axis in fig.keys():
                    util.logger.log_error(f"Figure '{attr_name}' should have a y_axis attribute!")
                    break

            
                if not attr_methods in fig.keys():
                    util.logger.log_error(f"Figure '{attr_name}' should have a methods attribute!")
                    break



                name = fig[attr_name]
                y_axis = fig[attr_y_axis]
                methods = str(fig[attr_methods]).split(",")

                if not attr_caption in fig.keys():
                    fig_caption = name
                else:
                    fig_caption = fig[attr_caption]

                if not attr_style in fig.keys():
                    style = ""
                else:
                    style = fig[attr_style]

                x_axis_scale = 1.0
                if attr_x_axis_scale in fig.keys():
                    x_axis_scale = float(fig[attr_x_axis_scale])

                y_axis_scale = 1.0
                if attr_y_axis_scale in fig.keys():
                    y_axis_scale = float(fig[attr_y_axis_scale])

                y_labels = None
                if attr_labels in fig.keys():
                    y_labels = str(fig[attr_labels]).split(",")
                
                plot_index = 0

                self.plotter.plot_begin(style_str=style)
                
                for method in methods:

                    if not method in probes_bin:
                        util.logger.log_error(f"Needed method(s) for figure '{name}' was not found!")
                        break
            
                    probes = pickle.loads(probes_bin[method])
                    probes_times_prof = probes["time_profiles"]
                    probes_vars = probes["var_values"]
                    probes_var_changes = probes["var_changes"]

                    y_axis_params = str(fig[attr_y_axis]).split(",")
                    
                    for y_axis in y_axis_params:
                        if y_axis in probes_times_prof:
                            fig_data_y = probes_times_prof[y_axis]
                        elif y_axis in probes_vars:
                            fig_data_y = probes_vars[y_axis]
                        elif y_axis in probes_var_changes:
                            fig_data_y = probes_var_changes[y_axis]
                        else:
                            util.logger.log_error(f"Expected y_axis for figure '{name}' was not found!")
                            break
                        

                        if x_axis in probes_times_prof:
                            fig_data_x = probes_times_prof[x_axis]
                        elif x_axis in probes_vars:
                            fig_data_x = probes_vars[x_axis]
                        elif x_axis in probes_var_changes:
                            fig_data_x = probes_var_changes[x_axis]
                        else:
                            util.logger.log_error(f"Expected y_axis for figure '{name}' was not found!")
                            break


                        x = [fig_data_elem[2] for fig_data_elem in fig_data_x]
                        y = [fig_data_elem[2] for fig_data_elem in fig_data_y]
                        
                    
                        x = [x_v * x_axis_scale for x_v in x]
                        y = [y_v * y_axis_scale for y_v in y]

                        if y_labels:
                            ylabel=y_labels[plot_index]
                        elif len(y_axis_params) == 1:
                            ylabel=method
                        else:
                            ylabel=f"{method}.{y_axis}"
                        
                        reference_point = (1.0, 1.0)
                        self.plotter.plot_hypervolume2d(x, y, ylabel, reference_point, style, plot_index)
                        plot_index += 1
                

                x_axis_title = x_axis
                y_axis_title = y_axis

                if attr_x_axis_title in fig.keys():
                    x_axis_title = fig[attr_x_axis_title]

                if attr_y_axis_title in fig.keys():
                    y_axis_title = fig[attr_y_axis_title]

                figure_path = os.path.join(output_path, f'{name}.pdf')
                self.plotter.plot_end(x_axis_title, y_axis_title, fig_caption, figure_path)


        fig_tag = "fig"
        if not fig_tag in report_cfg:
            util.logger.log_warning(f"It seems no figure as output is needed!")
            return
        
        figs_cfg = report_cfg[fig_tag]

        if not isinstance(figs_cfg, list):
            figs_cfg = [figs_cfg]
            
        for fig in figs_cfg:
            
            attr_name = "@name"
            attr_x_axis = "@x_axis"
            attr_x_axis_range = "@x_axis_range"
            attr_y_axis = "@y_axis"
            attr_methods = "@methods"
            attr_caption = "@caption"
            attr_labels = "@labels"
            attr_x_axis_title = "@x_axis_title"
            attr_y_axis_title = "@y_axis_title"
            attr_x_axis_scale = "@x_axis_scale"
            attr_y_axis_scale = "@y_axis_scale"
            attr_style = "@style"
            fig_caption = ""

            if not attr_name in fig.keys():
                util.logger.log_error(f"Figures should have a name attribute!")
                break

            if not attr_x_axis in fig.keys():
                x_axis = "Round"
            else:
                x_axis = fig["@x_axis"]

            if not attr_y_axis in fig.keys():
                util.logger.log_error(f"Figure '{attr_name}' should have a y_axis attribute!")
                break

        
            if not attr_methods in fig.keys():
                util.logger.log_error(f"Figure '{attr_name}' should have a methods attribute!")
                break



            name = fig[attr_name]
            y_axis = fig[attr_y_axis]
            methods = str(fig[attr_methods]).split(",")

            if not attr_caption in fig.keys():
                fig_caption = name
            else:
                fig_caption = fig[attr_caption]

            if not attr_style in fig.keys():
                style = ""
            else:
                style = fig[attr_style]

            x_axis_scale = 1.0
            if attr_x_axis_scale in fig.keys():
                x_axis_scale = float(fig[attr_x_axis_scale])

            y_axis_scale = 1.0
            if attr_y_axis_scale in fig.keys():
                y_axis_scale = float(fig[attr_y_axis_scale])

            y_labels = None
            if attr_labels in fig.keys():
                y_labels = str(fig[attr_labels]).split(",")
            
            plot_index = 0

            self.plotter.plot_begin(style_str=style)
            
            for method in methods:

                if not method in probes_bin:
                    util.logger.log_error(f"Needed method(s) for figure '{name}' was not found!")
                    break
        
                probes = pickle.loads(probes_bin[method])
                probes_times_prof = probes["time_profiles"]
                probes_vars = probes["var_values"]
                probes_var_changes = probes["var_changes"]

                y_axis_params = str(fig[attr_y_axis]).split(",")
                
                for y_axis in y_axis_params:
                    if y_axis in probes_times_prof:
                        fig_data = probes_times_prof[y_axis]
                    elif y_axis in probes_vars:
                        fig_data = probes_vars[y_axis]
                    elif y_axis in probes_var_changes:
                        fig_data = probes_var_changes[y_axis]
                    else:
                        util.logger.log_error(f"Expected y_axis for figure '{name}' was not found!")
                        break
                    
                    x = []
                    y = []

                    if not attr_x_axis_range in fig.keys():
                        if x_axis == "round":
                            x = [fig_data_elem[1] for fig_data_elem in fig_data]
                        elif x_axis == "time":
                            x = [(fig_data_elem[0] - fig_data[0][0]) for fig_data_elem in fig_data]
                        else:
                            util.logger.log_error(f"'{x_axis}' does not defined for figure '{name}' was not found!")
                            break
                        y = [fig_data_elem[2] for fig_data_elem in fig_data]
                    else:
                        x_range_str = fig[attr_x_axis_range]
                        x_range = str.split(x_range_str, ",")
                        x_range_start = float(x_range[0])
                        x_range_end = 0
                        if x_axis == "round":
                            x_range_end = float(x_range[1]) if float(x_range[1]) != -1 else max(fig_data[:][1])

                            for i in range(len(fig_data)):
                                if fig_data[i][1] >= x_range_start and fig_data[i][1] <= x_range_end:
                                    x.append(fig_data[i][1])
                                    y.append(fig_data[i][2])
                            
                        elif x_axis == "time":
                            x_range_end = float(x_range[1]) if float(x_range[1]) != -1 else max(fig_data[:][0])
                            for i in range(len(fig_data)):
                                if (fig_data[i][0] - fig_data[0][0]) >= x_range_start and (fig_data[i][0] - fig_data[0][0]) <= x_range_end:
                                    x.append(fig_data[i][0])
                                    y.append(fig_data[i][2])
                        else:
                            util.logger.log_error(f"'{x_axis}' does not defined for figure '{name}' was not found!")
                            break
                    

                   
                    x = [x_v * x_axis_scale for x_v in x]
                    y = [y_v * y_axis_scale for y_v in y]

                    if y_labels:
                        ylabel=y_labels[plot_index]
                    elif len(y_axis_params) == 1:
                        ylabel=method
                    else:
                        ylabel=f"{method}.{y_axis}"
                    
                    self.plotter.plot(x, y, ylabel, style, plot_index)
                    plot_index += 1
            

            x_axis_title = x_axis
            y_axis_title = y_axis

            if attr_x_axis_title in fig.keys():
                x_axis_title = fig[attr_x_axis_title]

            if attr_y_axis_title in fig.keys():
                y_axis_title = fig[attr_y_axis_title]

            figure_path = os.path.join(output_path, f'{name}.pdf')
            self.plotter.plot_end(x_axis_title, y_axis_title, fig_caption, style, figure_path)


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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Configuration file was not determined!\nUse: fedwork.py configuration_xml_file")
        exit()

    fedwork_ins = fedwork()
    fedwork_ins.start(sys.argv[1])
