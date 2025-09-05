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
from dataset.dataset import _make_eval_train_dataset_from_base
from torch.utils.data import ConcatDataset, DataLoader, Subset, RandomSampler, BatchSampler, SequentialSampler
import math
import os, pickle, torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, SequentialSampler, BatchSampler
from torchvision import datasets, transforms
from torch.utils.data import Subset as TorchSubset
from torch.utils.data import TensorDataset

def _empty_like_subset(subset):
    x_shape = subset[0][0].shape if len(subset) else (1,1,1)
    return torch.utils.data.TensorDataset(torch.empty(0, *x_shape), torch.empty(0, dtype=torch.long))

def _infer_base_dataset(dl):
    """Unwrap DataLoader.dataset -> (Subset ->) base torchvision/MedMNIST dataset."""
    ds = dl.dataset
    # unwrap nested Subset(...) -> dataset
    while isinstance(ds, TorchSubset):
        base_indices = ds.indices  # we keep these elsewhere
        ds = ds.dataset
    return ds

def _indices_from_loader(dl):
    """Extract Subset.indices from a client training loader."""
    ds = dl.dataset
    while isinstance(ds, TorchSubset):
        idxs = ds.indices
        inner = ds.dataset
        ds = inner
    return idxs

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

    
    def get_data_loaders(self, batch_size: int = 10):
        Common.set_seed_over_method(self.fl_context["seed"])
        train_loader_list = self.fl_context.get("dataset_train_list")
        if not train_loader_list or len(train_loader_list) == 0:
            raise RuntimeError("Call create_datasets() first: self.fl_context['dataset_train_list'] is empty.")

        # 1) Recreate an eval (non-random) training dataset matching the base type
        base_ds = _infer_base_dataset(train_loader_list[0])
        eval_train_ds = _make_eval_train_dataset_from_base(base_ds)   # your helper from earlier

        # 2) Indices
        idxs_target = list(_indices_from_loader(train_loader_list[0]))  # client 0
        target_len = len(idxs_target)

        # Build round-robin pool from all other clients
        other_lists = [list(_indices_from_loader(dl)) for dl in train_loader_list[1:]]
        # deterministically interleave
        mixed = []
        ptrs = [0] * len(other_lists)
        while len(mixed) < target_len and len(other_lists) > 0:
            progressed = False
            for i in range(len(other_lists)):
                if ptrs[i] < len(other_lists[i]):
                    mixed.append(other_lists[i][ptrs[i]])
                    ptrs[i] += 1
                    progressed = True
                    if len(mixed) == target_len:
                        break
            if not progressed:
                # ran out of pool (e.g., only one tiny other client) -> stop
                break

        # 3) Build subsets
        train_subset = Subset(eval_train_ds, idxs_target)
        val_subset = Subset(eval_train_ds, mixed) if len(mixed) > 0 else _empty_like_subset(train_subset)

        # 4) Deterministic evaluation loaders: sequential sampling, single worker
        train_loader = DataLoader(
            train_subset,
            batch_sampler=BatchSampler(SequentialSampler(train_subset), batch_size=batch_size, drop_last=False),
            num_workers=0,
        )
        val_loader = DataLoader(
            val_subset,
            batch_sampler=BatchSampler(SequentialSampler(val_subset), batch_size=batch_size, drop_last=False),
            num_workers=0,
        )
        Common.set_seed_over_method(self.fl_context["seed"])
        return val_loader, train_loader


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
            Common.set_seed_over_method(self.fl_context["seed"])
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
            Common.set_seed_over_method(self.fl_context["seed"])


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


    def pack_client_model(self, raw_model, global_model, id):
        raw_model = self.fla.build_packet(raw_model, global_model, id)


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
                    quantized_tensor, mins[key], scale[key] = raw_model[key], 0, -1
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
                if scale[key] == -1:
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