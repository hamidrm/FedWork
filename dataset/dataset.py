import os
import random
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from dataset import MedMNIST
from utils.logger import *
import utils.consts as consts


def _make_eval_transforms_and_datasets(ds_type: str):
    if ds_type == "MNIST":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root="./dataset/data", train=True, transform=tf, download=True)
        test_dataset  = datasets.MNIST(root="./dataset/data", train=False, transform=tf)
        dataset_label_list = train_dataset.targets.tolist()

    elif ds_type == "CIFAR10":
        stats = ((0.49139968, 0.48215841, 0.44653091),
                 (0.24703223, 0.24348513, 0.26158784))
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_dataset = datasets.CIFAR10(root="./dataset/data", train=True, transform=tf, download=True)
        test_dataset  = datasets.CIFAR10(root="./dataset/data", train=False, transform=tf)
        dataset_label_list = train_dataset.targets

    elif ds_type == "CIFAR100":
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2761)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = datasets.CIFAR100(root="./dataset/data", train=True, transform=tf, download=True)
        test_dataset  = datasets.CIFAR100(root="./dataset/data", train=False, transform=tf, download=True)
        dataset_label_list = train_dataset.targets

    elif ds_type == "FashionMNIST":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(root="./dataset/data", train=True, transform=tf, download=True)
        test_dataset  = datasets.FashionMNIST(root="./dataset/data", train=False, transform=tf)
        dataset_label_list = train_dataset.targets.tolist()

    elif ("-" in ds_type) and ds_type.split("-")[0].lower() == "medmnist":
        tf = transforms.Compose([transforms.ToTensor()])
        name = ds_type.split("-")[1]
        train_dataset = MedMNIST.MedMNIST(dataset_name=name, root="./dataset/data",
                                          train=True, transform=tf, download=True)
        test_dataset  = MedMNIST.MedMNIST(dataset_name=name, root="./dataset/data",
                                          train=False, transform=tf)
        dataset_label_list = train_dataset.targets.tolist()

    else:
        raise ValueError(f"Dataset '{ds_type}' not recognized.")

    return train_dataset, test_dataset, dataset_label_list


def _make_train_transforms_and_datasets(ds_type: str):
    if ds_type == "MNIST":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root="./dataset/data", train=True, transform=tf, download=True)
    elif ds_type == "CIFAR10":
        stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        train_dataset = datasets.CIFAR10(root="./dataset/data", train=True, transform=tf, download=True)
    elif ds_type == "CIFAR100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2761)
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = datasets.CIFAR100(root="./dataset/data", train=True, transform=tf, download=True)
    elif ds_type == "FashionMNIST":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(root="./dataset/data", train=True, transform=tf, download=True)
    elif ("-" in ds_type) and ds_type.split("-")[0].lower() == "medmnist":
        tf = transforms.Compose([transforms.ToTensor()])
        name = ds_type.split("-")[1]
        train_dataset = MedMNIST.MedMNIST(dataset_name=name, root="./dataset/data", train=True, transform=tf, download=True)
    else:
        raise ValueError(f"Dataset '{ds_type}' not recognized.")
    return train_dataset


def _cpu_dl_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def build_loaders_from_partitions(
    partitions, ds_type: str, train_batch_size: int, test_batch_size: int, base_seed: int = 0, num_workers: int = 0, augment: bool = True
):
    train_dataset = _make_train_transforms_and_datasets(ds_type) if augment else _make_eval_transforms_and_datasets(ds_type)[0]
    _, test_dataset, _ = _make_eval_transforms_and_datasets(ds_type)

    train_loaders = []
    for cid, idxs in enumerate(partitions):
        subset = Subset(train_dataset, idxs)
        g = _cpu_dl_generator(base_seed + 1000 + cid)
        loader = DataLoader(
            subset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            generator=g,
            persistent_workers=bool(num_workers),
        )
        train_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=bool(num_workers),
    )

    return train_loaders, test_loader


def worker_init_fn(worker_id):
    s = 10_000 + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator("cpu")
    g.manual_seed(seed)
    return g

def _infer_ds_type_from_base(base_ds):
    if isinstance(base_ds, datasets.CIFAR10):   return "CIFAR10"
    if isinstance(base_ds, datasets.CIFAR100):  return "CIFAR100"
    if isinstance(base_ds, datasets.MNIST):     return "MNIST"
    if isinstance(base_ds, datasets.FashionMNIST): return "FashionMNIST"
    try:
        from dataset import dataset
        if isinstance(base_ds, dataset.MedMNIST.MedMNIST):
            return f"medmnist-{base_ds.flag}"
    except Exception:
        raise ValueError(f"Unsupported base dataset type: {type(base_ds)}")
    
def _make_eval_train_dataset_from_base(base_ds):
    ds_type = _infer_ds_type_from_base(base_ds)
    eval_train_ds, _, _ = _make_eval_transforms_and_datasets(ds_type)
    return eval_train_ds

def build_client_loader(train_dataset, indices, batch_size, base_seed, client_id, num_workers: int = 0):
    subset = Subset(train_dataset, indices)
    g = make_generator(base_seed + 1000 + client_id)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
        persistent_workers=bool(num_workers),
    )


def create_datasets(
    train_ds_num=5,
    ds_type="MNIST",
    non_iid_level_alpha=0.1,
    train_batch_size=64,
    test_batch_size=64,
    use_dirichlet=False,
    num_workers=0,
    save_graph=True,
    add_info_to_figure=False,
    path=None,
    base_seed: int = 0,
):
    logger.log_info(
        f"Dirichlet: {use_dirichlet}, Non-i.i.d Level: {non_iid_level_alpha}, Train Batch Size: {train_batch_size}, Test Batch Size: {test_batch_size}"
    )

    rng = np.random.default_rng(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    train_dataset_eval, _, dataset_label_list = _make_eval_transforms_and_datasets(ds_type)

    train_classes_num = int(len(np.unique(dataset_label_list)))
    train_total_dataset_size = len(train_dataset_eval)
    train_groups_eq_size = train_total_dataset_size // train_ds_num

    classes = []
    client_distributions = []

    if not use_dirichlet:
        all_indices = rng.permutation(train_total_dataset_size).tolist()
        split_indices = [all_indices[i * train_groups_eq_size : (i + 1) * train_groups_eq_size] for i in range(train_ds_num)]
        remainder = train_total_dataset_size % train_ds_num
        for i in range(remainder):
            split_indices[i].append(all_indices[train_ds_num * train_groups_eq_size + i])
        partitions = split_indices
        for subset_indices in split_indices:
            if isinstance(train_dataset_eval.targets, list):
                classes.append([train_dataset_eval.targets[i] for i in subset_indices])
            else:
                classes.append([train_dataset_eval.targets[i].item() for i in subset_indices])
            client_labels = [dataset_label_list[i] for i in subset_indices]
            class_counts = Counter(client_labels)
            client_distributions.append(class_counts)
    else:
        if non_iid_level_alpha <= 0:
            raise ValueError("non_iid_level_alpha must be > 0 when use_dirichlet=True.")
        class_proportions = rng.dirichlet([non_iid_level_alpha] * train_ds_num, train_classes_num)
        subset_indices_lists = [[] for _ in range(train_ds_num)]
        for class_index in range(train_classes_num):
            class_indices = [i for i, lbl in enumerate(dataset_label_list) if lbl == class_index]
            class_indices = rng.permutation(class_indices).tolist()
            sizes = rng.multinomial(len(class_indices), class_proportions[class_index])
            offset = 0
            for client_id, size in enumerate(sizes):
                assigned_indices = class_indices[offset : offset + size]
                offset += size
                subset_indices_lists[client_id].extend(assigned_indices)
        partitions = subset_indices_lists
        for client_id in range(train_ds_num):
            assigned_indices = subset_indices_lists[client_id]
            client_labels = [dataset_label_list[i] for i in assigned_indices]
            class_counts = Counter(client_labels)
            client_distributions.append(class_counts)
            if isinstance(train_dataset_eval.targets, list):
                classes.append([train_dataset_eval.targets[i] for i in assigned_indices])
            else:
                classes.append([train_dataset_eval.targets[i].item() for i in assigned_indices])

    train_datasets, test_dataset_loader = build_loaders_from_partitions(
        partitions, ds_type, train_batch_size, test_batch_size, base_seed=base_seed, num_workers=num_workers, augment=True
    )

    if save_graph:
        unique_classes = sorted(set(dataset_label_list))
        graph_map = [[0 for _ in range(len(train_datasets))] for _ in range(len(unique_classes))]
        for client_index in range(len(train_datasets)):
            for class_index in unique_classes:
                graph_map[class_index][client_index] = classes[client_index].count(class_index)
        max_val = len(train_dataset_eval.targets)
        normalized_matrix = [[val / max_val for val in row] for row in graph_map]
        figure_width = max(len(train_datasets) / 2, 8)
        figure_height = max(len(unique_classes) / 2, 6)
        plt.figure(figsize=(figure_width, figure_height))
        plt.imshow(normalized_matrix, cmap="gray_r", interpolation="nearest")
        if add_info_to_figure:
            plt.text(
                len(train_datasets) + 1,
                len(unique_classes) / 2,
                f"{type(train_dataset_eval).__name__} \nNon-i.i.d level: {non_iid_level_alpha}\nTrain batch size: {train_batch_size}\nTest batch size: {test_batch_size}",
                fontsize=8,
                color="red",
                rotation=90,
                va="center",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.5),
            )
        plt.xlim(-1, len(train_datasets))
        plt.ylim(-1, len(unique_classes))
        output_directory = os.path.join(os.getcwd(), consts.OUTPUT_DIR) if path is None else path
        dir_path = os.path.join(output_directory, "dataset")
        os.makedirs(dir_path, exist_ok=True)
        plt.yticks([i for i in range(len(unique_classes))], train_dataset_eval.classes)
        plt.xticks([i for i in range(train_ds_num)], [(i + 1) for i in range(train_ds_num)])
        current_time = datetime.now()
        time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        full_path = os.path.join(dir_path, f"dataset_distribution_{time_str}.pdf")
        plt.savefig(full_path, format="pdf", bbox_inches="tight")
        plt.close()

        all_classes = sorted({cls for dist in client_distributions for cls in dist})
        distribution_matrix = np.zeros((len(client_distributions), len(all_classes)))
        for i, dist in enumerate(client_distributions):
            for cls, count in dist.items():
                class_index = all_classes.index(cls)
                distribution_matrix[i, class_index] = count

        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(client_distributions))
        for class_index, cls in enumerate(all_classes):
            class_distribution = distribution_matrix[:, class_index]
            ax.bar(range(len(client_distributions)), class_distribution, bottom=bottom, label=f"Class {cls}")
            bottom += class_distribution
        plt.tick_params(axis="x", labelsize=20)
        plt.tick_params(axis="y", labelsize=20)
        plt.xticks([i for i in range(len(train_datasets))], [f"{i}" for i in range(len(train_datasets))])
        full_path = os.path.join(dir_path, f"dataset_distribution_{time_str}_sbp.pdf")
        plt.savefig(full_path, format="pdf", bbox_inches="tight")
        plt.close()

    return train_datasets, test_dataset_loader, partitions
