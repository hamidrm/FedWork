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
from utils.plotter import Plotter
from methods.FedPipe import FedPipe


class fedwork:
    def __init__(self):
        self.local_clients = []
        self.plotter = Plotter()
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
                    raise ValueError(f"Cannot convert '{desired_item_value}' to bool.")
            else:
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
        heterogeneous = self.get_var(vars, "heterogeneous", bool, False)
        non_iid_level = self.get_var(vars, "non_iid_level", float, 0.5)
        train_batch_size = self.get_var(vars, "train_batch_size", int, 128)
        test_batch_size = self.get_var(vars, "test_batch_size", int, 128)
        num_workers = self.get_var(vars, "num_workers", int, 1)
        save_graph = self.get_var(vars, "save_graph", bool, True)
        enclose_info = self.get_var(vars, "enclosed_info", bool, False)


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

    def run(self, config_text):

        if not os.path.exists(const.OUTPUT_DIR):
            os.mkdir(const.OUTPUT_DIR)

        dict_cfg = xmltodict.parse(config_text)
        fedwork_cfg = dict_cfg.get("fedwork_cfg")
        if fedwork_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'fedwork' tag is not found.")
            return
        
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
        
        methods_cfg = fedwork_cfg.get("method")
        if methods_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'method' is not found.")
            return
        
        report_cfg = fedwork_cfg.get("report")
        if report_cfg is None:
            util.logger.log_error("Invalid config file! Tag 'report' is not found.")
            return
        
        attr_save_log = "@save_log"
        save_log_def = "True"
        save_log = bool(report_cfg[attr_save_log] if attr_save_log in report_cfg.keys() else save_log_def)


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
        train_dataset_list, test_dataset = self.create_datasets(dataset_cfg, int(fedwork_cfg["@num_of_nodes"]), output_path)
        
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

        # Step 2.
        # For each method, we have to execute federated learning according to corresponding configuration
        probes_bin = {}
        methods_cfg = methods_cfg if isinstance(methods_cfg, list) else [methods_cfg]
        for method in methods_cfg:
        
            attr_method_type = "@type"
            attr_method_name = "@name"

            if not attr_method_type in method.keys():
                util.logger.log_error(f"Method type is not determined!")
                continue

            method_type = method[attr_method_type]
            method_name = method[attr_method_name]

            probes_data_path = os.path.join(output_path, f"{method_name}_probes_data.data")

            if os.path.exists(probes_data_path):
                util.logger.log_info(f"Information for method '{method_name}(type={method_type})' has been found!")

                if os.path.exists(probes_data_path):
                    with open(probes_data_path, "rb") as f:
                        probes_bin[method_name] = f.read()
                else:
                    probes_bin[method_name] = None

                continue


            attr_method_platform = "@platform"
            attr_method_platform_def = "cpu"
            method_platform = method[attr_method_platform] if attr_method_platform in method.keys() else attr_method_platform_def

            method_num_of_epochs = self.get_var(method["var"], "epochs_num", int, 5)
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
                        elif var_type == "bool":
                            arch.SetParameter(var_name, bool(var_text))
                        else:
                            util.logger.log_error(f"Unexpectedly error in type of the variable '{var_name}'!")
                            break
                
            arch.Build()
            global_model = arch.CreateModel().to(method_platform)

            loss_func = self.get_loss_function(eval_criterion)

            weights = [(float(len(dataloader.dataset)) / float(sum([len(dataloader.dataset) for  dataloader in train_dataset_list]))) for dataloader in train_dataset_list]

            if method_args == "":
                method_obj = self.load_method(method_class, method_type, (method_num_of_epochs, num_of_rounds, weights, method_platform))
            else:
                method_obj = self.load_method(method_class, method_type, (method_num_of_epochs, num_of_rounds, weights, method_platform, method_args))
            
            server = Server(IpAddr(net_ip, net_port), method_obj, test_dataset, global_model, loss_func, method_platform)

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

                localclients_num = int(localclients_cfg[localclients_num_key])
                client_platform = localclients_cfg[attr_platform]
                optimizer = self.get_optimizer_class(localclients_cfg[attr_optimizer])

                if localclients_num > len(train_dataset_list):
                    util.logger.log_warning(f"Local clients number must not be greater the total nodes number! Local clients number will be assumed {len(train_dataset_list)}")
                    localclients_num = len(train_dataset_list)
                    break
                
                if localclients_num != 0:
                    for client_id in range(localclients_num):
                        model = arch.CreateModel().to(method_platform)
                        if method_args == "":
                            method_obj = self.load_method(method_class, method_type, (method_num_of_epochs, num_of_rounds, weights, method_platform))
                        else:
                            method_obj = self.load_method(method_class, method_type, (method_num_of_epochs, num_of_rounds, weights, method_platform, method_args))
                        
                        new_client = Client(f"Client{client_id}", IpAddr(net_ip, net_port), TrainingHyperParameters(learning_rate, momentum, weight_decay), train_dataset_list[client_id], model, optimizer, loss_func, method_obj, client_platform)
                        self.local_clients.append(new_client)

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
            self.plotter.plot_end(x_axis_title, y_axis_title, fig_caption, figure_path)


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
    if len(sys.argv) == 1:
        print("Configuration file was not determined!\nUse: fedwork.py configuration_xml_file")
        exit()

    fedwork_ins = fedwork()
    fedwork_ins.start(sys.argv[1])
