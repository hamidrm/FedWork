import os
import random
import math
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from arch.arch import BaseArch, FWArch
from dataset import MedMNIST
from utils.logger import *
import utils.consts as consts
from dataset.dataset import _make_eval_transforms_and_datasets
from dataset.MedMNIST import *
from utils.security.YeomBlackBoxAttack import YeomBlackBoxAttack
import pickle

def split_loader(loader, frac_calib=0.3, seed=0):
    ds = loader.dataset
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    n_cal = int(n * frac_calib)
    cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]
    cal_ds, eval_ds = Subset(ds, cal_idx), Subset(ds, eval_idx)

    def clone(dl_ds):
        return DataLoader(
            dl_ds,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=getattr(loader, "pin_memory", False),
            collate_fn=getattr(loader, "collate_fn", None),
            drop_last=False,
            persistent_workers=getattr(loader, "persistent_workers", False),
        )
    return clone(cal_ds), clone(eval_ds)

def make_equal_loaders(dataset_name, batch_size=128, seed=42):
    g = torch.Generator().manual_seed(seed)

    transform = transforms.ToTensor()
    train_ds, test_ds, _ = _make_eval_transforms_and_datasets(dataset_name)

    x = len(test_ds)

    if len(train_ds) >= x:
        idx = torch.randperm(len(train_ds), generator=g)[:x].tolist()
        train_subset = Subset(train_ds, idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        weights = torch.ones(len(train_ds))
        sampler = WeightedRandomSampler(weights, num_samples=x, replacement=True, generator=g)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    assert len(train_loader) == math.ceil(x / batch_size) == len(test_loader)

    return train_loader, test_loader

def calc_yoem(model, member_dataset, nonmember_dataset, seed = 123):
    
    
    member_calib, member_eval = split_loader(member_dataset, frac_calib=0.3, seed=seed)
    nonmember_calib, nonmember_eval = split_loader(nonmember_dataset, frac_calib=0.3, seed=seed)

    model.eval().to()
    def predict_fn(x):
        with torch.no_grad():
            return model(x.to())
    
    attack = YeomBlackBoxAttack(predict_fn)
    attack.fit(member_calib=member_calib, nonmember_calib=nonmember_calib)

    metrics = attack.evaluate(member_eval, nonmember_eval)
    print(metrics)

def load_pickle_file(filename):
    """Load a dictionary from a pickle file."""
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
def measure_mia(output_file):
    
    data_file = load_pickle_file(output_file)
    data = data_file["method_info"]
    ds_type = data["ds_type"]
    arch_type = data["arch_type"]

    member_loader, nonmember_loader = make_equal_loaders(ds_type)

    for _arch_type in BaseArch:
        if _arch_type.value == arch_type:
            arch_type = _arch_type
            
    arch = FWArch(arch_type)
    
    
    arch.SetParameter("NumberOfOutputNodes", len(member_loader.dataset.classes))
    
    msg = arch.Build()

    if msg != '':
        print(f"Model Architecture Error: '{msg}'")
    
    global_model = arch.CreateModel()

    global_model.load_state_dict(pickle.loads(data_file["global_model"]))
    
    calc_yoem(global_model, member_loader, nonmember_loader)
    
    
measure_mia("output/fedalaq_hyperparams_compare/FedALAQ_0_9_100_probes_data.data")