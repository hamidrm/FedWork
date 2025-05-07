#FedALA
import pickle
import torch
from core.FederatedLearningClass import *
import random
import torch.nn as nn
from utils.logger import *
from utils.profiler import *
from utils.common import Common
from utils.quantization import RandomizedQuantizer
from utils.security.DataManipulation import MixUpDefense
from utils.security.MIAPartial import *
from utils.security.FedALA import *
from utils.security.GradientSparsifier import *

from torch.utils.data import ConcatDataset, DataLoader, Subset, RandomSampler, BatchSampler, SequentialSampler
import math

class FedALA(FederatedLearningClass):

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.fedmia_attack = None
        self.lr = 0.01
        self.lcr = self.get_arg(float, "lcr", 1.0)
        self.alpha = self.get_arg(float, "alpha", 0.0)
        self.ldp_noise_std = self.get_arg(float, "ldp_noise_std", 0)
        self.mixup_alpha = self.get_arg(float, "mixup_alpha", 0)
        self.gradient_sparsifier_ratio = self.get_arg(float, "gradient_sparsifier", 1.0)
        self.q_levels = self.get_arg(int, "randomized_quantization_levels", 0)


        self.fla = FedALADefense(self.lcr, self.alpha, self.ldp_noise_std)
        self.mixup = MixUpDefense(self.mixup_alpha)
        self.gradient_sparsifier = GradientSparsifier(self.gradient_sparsifier_ratio) if self.gradient_sparsifier_ratio != 1.0 else None
        self.quantization = RandomizedQuantizer(self.q_levels) if self.q_levels != 0 else None


    def get_data_loaders(self):
        dataset_list = self.fl_context["dataset_train_list"]
        dataset_train = self.fl_context["dataset_train"]
        dir_path = self.fl_context["dataset_path"]
        
        mia_dataset_train_path      = os.path.join(dir_path, f"mia_dataset_train.ds")
        mia_dataset_validation_path      = os.path.join(dir_path, f"mia_dataset_validation.ds")
        
        if os.path.isfile(mia_dataset_train_path) and os.path.isfile(mia_dataset_validation_path):
            with open(mia_dataset_validation_path,      'rb') as f:
                dataset_loader_validation      = pickle.loads(f.read())
            with open(mia_dataset_train_path,      'rb') as f:
                dataset_loader_train      = pickle.loads(f.read())
            return dataset_loader_validation, dataset_loader_train
        
        file_path_train      = os.path.join(dir_path, f"dataset_node_0.ds") #Dataset of Client 0 , as the target's dataset

        with open(file_path_train,      'rb') as f:
            dataset_loader_train      = pickle.loads(f.read())


        
        new_sampler = BatchSampler(SequentialSampler(dataset_loader_train.dataset), batch_size=10, drop_last=False)
        dataset_loader_train = DataLoader(dataset_loader_train.dataset, batch_sampler=new_sampler, num_workers=8)

        client0_indices = dataset_loader_train.dataset.indices if hasattr(dataset_loader_train.dataset, 'indices') \
            else list(range(len(dataset_loader_train.dataset)))

        all_indices = set(range(len(dataset_train)))

        non_member_pool = list(all_indices - set(client0_indices))

        same_length = len(client0_indices)
        random.seed(1)
        non_member_sample = random.sample(non_member_pool, same_length)

        non_member_dataset = Subset(dataset_train, non_member_sample)

        dataset_loader_validation = DataLoader(non_member_dataset, batch_size=10, shuffle=True, num_workers=8)


        with open(mia_dataset_validation_path,      'wb') as f:
            f.write(pickle.dumps(dataset_loader_validation))
        with open(mia_dataset_train_path,      'wb') as f:
            f.write(pickle.dumps(dataset_loader_train))
                
        return dataset_loader_validation, dataset_loader_train


    def get_name(self):
        return "FedALA"
    
    def init_method(self, server):

        separator = "-" * 55
        title = "Federated Adaptive Layer Aggregation"
        info1 = f"Layers Contribution Ratio: {self.lcr * 100:.2f}%, Alpha: {self.alpha}"
        info2 = f"LDP Noise Std.: {self.ldp_noise_std:.2f}, MixUp Alpha: {self.mixup_alpha}"
        info3 = f"Sparsifier Ratio: {self.gradient_sparsifier_ratio:.2f}, Quantization Levels: {self.q_levels}"
        logger.log_normal(separator)
        logger.log_normal(f"|{title.center(53)}|")
        logger.log_normal(separator)
        logger.log_normal(f"|{info1.center(53)}|")
        logger.log_normal(f"|{info2.center(53)}|")
        logger.log_normal(f"|{info3.center(53)}|")
        logger.log_normal(separator)
        dataset_loader_validation, dataset_loader_train = self.get_data_loaders()
        self.fedmia_attack = FedMIA(dataset_loader_train, dataset_loader_validation, torch.optim.SGD, nn.CrossEntropyLoss)
        
        super().init_method(server)


    def aggregate(self, clients_models, global_model):
        if self.quantization is not None or self.gradient_sparsifier is not None:
            for i, client_model in enumerate(clients_models):
                for key in client_model[1].keys():
                    if Common.is_trainable(global_model, key):
                        clients_models[i][1][key] += global_model[key]


        self.fla.aggregate(clients_models, global_model, self.datasets_weights, 0.0)
        

        if self.round_num() % 10 == 0:
            model_class = self.method_dict["arch"]
            global_model_clone = model_class().to(self.platform)

            target_model_id = 0 # Client 0 will be our target node to evaluate FedMIA attack on it
            
            target_model_index = next(
                (i for i, client_state_dict in enumerate(clients_models) if client_state_dict[0] == target_model_id), 
                None
            )

            global_model_clone.load_state_dict(global_model)

            shadow_models = []
            for i, client_state_dict in enumerate(clients_models):
                if i != target_model_index:
                    shadow_models.append(client_state_dict[1])

            self.fedmia_attack.execute(shadow_models, clients_models[target_model_index][1], global_model_clone, self.platform, self.lr)
            res_total = self.fedmia_attack.get_auc_metrics(self.platform)
            if res_total != None:
                logger.log_normal(f"FedMIA Attack on round {self.round_num()}: model id: {target_model_id}, {res_total}")
                profiler.save_variable("MIA_TPRS_0_1", res_total["tprs"]["0.1"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_02", res_total["tprs"]["0.02"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_01", res_total["tprs"]["0.01"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_001", res_total["tprs"]["0.001"], self.round_num() - 1)
                profiler.save_variable("MIA_TPRS_0_0001", res_total["tprs"]["0.001"], self.round_num() - 1)
                profiler.save_variable("MIA_AUC", res_total["auc"], self.round_num() - 1)


    def start_training(self):
        logger.log_normal(f"===================================================")
        eval_loss, eval_accuracy = self.server.evaluate_model()
        logger.log_normal(f"Round {self.server.round_number} is starting...")
        logger.log_normal(f"Current situation:\n\tAccuracy: {eval_accuracy}, Loss: {eval_loss}")
        if self.server.round_number != self.num_of_rounds:
            self.lr = 0.1 * (1 + math.cos(math.pi * self.server.round_number / self.num_of_rounds)) / 2 
            #self.server.start_round(self.clients_epochs, self.lr)
            self.server.start_round(self.clients_epochs)

            return (eval_loss, eval_accuracy)
        else:
            res = self.fedmia_attack.get_auc_metrics(self.platform)
            logger.log_normal(f"Final FedMIA Attack on {self.round_num()} epochs: {res}")
            logger.log_normal(f"Training done! last global model accuracy is: {eval_accuracy}")
            return None


    def pack_client_model(self, raw_model, global_model):
        raw_model = self.fla.build_packet(raw_model, global_model)


        if self.gradient_sparsifier is not None:
            raw_model = self.gradient_sparsifier.sparsify(raw_model, global_model)

        if self.quantization is not None:
            quantized_model = {}
            packet_to_send = {}
            scale = {}
            mins = {}
            for key in raw_model.keys():
                #Skip the statstical parameters
                if not Common.is_trainable(raw_model, key):
                    quantized_tensor, mins[key], scale[key] = raw_model[key], 0, 0
                    quantized_model[key] = quantized_tensor.to(torch.long)
                else:
                    quantized_tensor, mins[key], scale[key] = self.quantization.quantize(raw_model[key] - global_model[key])
                    #if torch.any(quantized_tensor < 0) or torch.any(quantized_tensor > 255):
                    #    raise ValueError("Quantization values outside uint8 range detected!")
                    quantized_model[key] = quantized_tensor.to(torch.uint8)      

            packet_to_send["tensors"] = quantized_model
            packet_to_send["scales"] = scale
            packet_to_send["mins"] = mins
            return packet_to_send
    
        return raw_model

    def unpack_client_model(self, packed_model):

        if self.quantization is not None:
            quantized_model = packed_model["tensors"]
            scale = {}
            mins = {}
            dequantized_model = {}
            scale = packed_model["scales"]
            mins = packed_model["mins"]
            for key in quantized_model.keys():
                if scale[key] == 0:
                    dequantized_model[key] = quantized_model[key]
                else:
                    dequantized_model[key] = self.quantization.dequantize(quantized_model[key], mins[key], scale[key])

            return dequantized_model
        return packed_model
    
    def ready_to_aggregate(self, num_of_received_model: int) -> bool:
        logger.log_normal(f"Number of trained models: {num_of_received_model}")
        return super().ready_to_aggregate(num_of_received_model)
    


    def client_training_get_data(self, inputs, labels):
        inputs, self.labels_actual, labels, _ = self.mixup.get_data(inputs, labels)
        return inputs, labels

    def client_training_correctness(self, outputs, labels):
        return self.mixup.correctness(outputs, self.labels_actual, labels)
    
    def client_training_criterion(self, criterion_fn, outputs, labels):
        return self.mixup.criterion(criterion_fn, outputs, self.labels_actual, labels)