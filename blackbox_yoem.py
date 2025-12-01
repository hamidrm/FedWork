import os
import random
import math
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utils.security.ShokriShadowAttack import ShokriShadowAttack

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

    train_ds, test_ds, _ = _make_eval_transforms_and_datasets(dataset_name)

    x = len(test_ds)


    idx = torch.randperm(len(train_ds), generator=g)[:x].tolist()
    train_subset = Subset(train_ds, idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, drop_last=False)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)


    assert len(train_loader) == math.ceil(x / batch_size) == len(test_loader)

    return train_loader, test_loader, train_ds

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


def infer_num_classes(model, loader, device):
    model.eval().to(device)
    for x, y in loader:
        with torch.no_grad():
            out = model(x.to(device))
        return out.shape[1]
    raise ValueError("Empty loader")

# --- (B) minimal shadow training loop (replace with your own if you have one) ---
def default_train_shadow_model(model, train_loader, device, epochs=5, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()

# --- (C) build K shadow triplets from a pool dataset ---
def build_shadow_triplets(pool_dataset, ModelCtor, device, num_shadows=8, train_frac=0.5,
                          batch_size=128, seed=0, train_epochs=5):
    g = torch.Generator().manual_seed(seed)
    n = len(pool_dataset)
    triplets = []
    for _ in range(num_shadows):
        idx = torch.randperm(n, generator=g)
        n_train = int(train_frac * n)
        mem_idx = idx[:n_train].tolist()
        nonmem_idx = idx[n_train:].tolist()

        shadow_train_loader = DataLoader(Subset(pool_dataset, mem_idx), batch_size=batch_size, shuffle=True)
        shadow_nonmember_loader = DataLoader(Subset(pool_dataset, nonmem_idx), batch_size=batch_size, shuffle=False)

        m = ModelCtor().to(device)
        default_train_shadow_model(m, shadow_train_loader, device=device, epochs=train_epochs)

        @torch.no_grad()
        def pf(x, _m=m):   # bind model
            return _m(x.to(device))   # logits

        triplets.append((pf, shadow_train_loader, shadow_nonmember_loader))
    return triplets

# --- (D) full wrapper: train attack and evaluate on target member/nonmember ---
def calc_shokri(target_model, member_loader, nonmember_loader,
                pool_dataset, ModelCtor, device=None,
                num_shadows=8, shadow_train_frac=0.5,
                attack_epochs=10, attack_lr=1e-3, batch_size=128, seed=0):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model.to(device).eval()

    # 1) Build shadow triplets (shadow predict_fn, shadow member loader, shadow nonmember loader)
    shadow_triplets = build_shadow_triplets(
        pool_dataset=pool_dataset,
        ModelCtor=ModelCtor,
        device=device,
        num_shadows=num_shadows,
        train_frac=shadow_train_frac,
        batch_size=batch_size,
        seed=seed,
        train_epochs=5
    )

    # 2) Create and fit the Shokri attack (your class)
    C = infer_num_classes(target_model, member_loader, device)

    attack = ShokriShadowAttack(num_classes=C, device=device, hidden=128)
    attack.fit(shadow_triplets, epochs=attack_epochs, lr=attack_lr, batch_size=batch_size)

    # 3) Evaluate on the target model
    @torch.no_grad()
    def target_predict_fn(x):
        return target_model(x.to(device))  # logits

    metrics = attack.evaluate(target_predict_fn, member_loader, nonmember_loader)
    print(metrics)
    
def measure_mia(output_file):
    
    data_file = load_pickle_file(output_file)
    data = data_file["method_info"]
    ds_type = data["ds_type"]
    arch_type = data["arch_type"]

    member_loader, nonmember_loader, train_ds = make_equal_loaders(ds_type)

    for _arch_type in BaseArch:
        if _arch_type.value == arch_type:
            arch_type = _arch_type
            
    arch = FWArch(arch_type)
    
    
    arch.SetParameter("NumberOfOutputNodes", 10)
    
    msg = arch.Build()

    if msg != '':
        print(f"Model Architecture Error: '{msg}'")
    
    global_model = arch.CreateModel()

    global_model.load_state_dict(data_file["global_model"])
    
    calc_yoem(global_model, member_loader, nonmember_loader)
    
    calc_shokri(
        target_model=global_model,
        member_loader=member_loader,
        nonmember_loader=nonmember_loader,
        pool_dataset=train_ds,   # the pool used to sample shadow members/nonmembers
        ModelCtor=arch.CreateModel,          # e.g., lambda: SmallCNN()
        device="cpu",
        num_shadows=8,
        shadow_train_frac=0.5,
        attack_epochs=10,
        attack_lr=1e-3,
        batch_size=128,
        seed=0
    )
    
    
measure_mia("output/fedalaq_hyperparams_compare/FedAvg_probes_data.data")