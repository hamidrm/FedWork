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
class ScaleToMinusOneOne:
    def __call__(self, x):
        return x.mul(2.0).sub(1.0)

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
    elif ds_type == "CIFAR10_BOP":
        tf = transforms.Compose([
            transforms.ToTensor(),
            ScaleToMinusOneOne(),
        ])

        train_dataset = datasets.CIFAR10(
            root="./dataset/data",
            train=True,
            transform=tf,
            download=True,
        )

        test_dataset = datasets.CIFAR10(
            root="./dataset/data",
            train=False,
            transform=tf,
        )

        dataset_label_list = train_dataset.targets
    elif ds_type == "CIFAR100_N":
        tf = transforms.Compose([
            transforms.ToTensor(),
            ScaleToMinusOneOne(),
        ])

        train_dataset = datasets.CIFAR100(root="./dataset/data", train=True, transform=tf, download=True)
        test_dataset  = datasets.CIFAR100(root="./dataset/data", train=False, transform=tf, download=True)

        dataset_label_list = train_dataset.targets
    elif ds_type == "CIFAR100":
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2761)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = datasets.CIFAR100(root="./dataset/data", train=True, transform=tf, download=True)
        test_dataset  = datasets.CIFAR100(root="./dataset/data", train=False, transform=tf, download=True)
        dataset_label_list = train_dataset.targets
        
    elif ds_type == "SVHN":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=tf)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=tf)
        train_dataset.targets = train_dataset.labels
        dataset_label_list = train_dataset.labels
        train_dataset.classes = [str(i) for i in range(10)]
        train_dataset.class_to_idx = {str(i): i for i in range(10)}
        test_dataset.classes = [str(i) for i in range(10)]
        test_dataset.class_to_idx = {str(i): i for i in range(10)}
        
    elif ds_type == "IMAGENETTE":
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)

        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        tf_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        try:
            train_dataset = datasets.Imagenette(root='./data', split='train', download=True, transform=tf_train)
        except RuntimeError as e:
            if "already exists" in str(e) and "imagenette2" in str(e):
                train_dataset = datasets.Imagenette(root="./data", split="train", download=False, transform=tf_train)
            else:
                raise
        try:
            test_dataset = datasets.Imagenette(root='./data', split='val', download=True, transform=tf_test)
        except RuntimeError as e:
            if "already exists" in str(e) and "imagenette2" in str(e):
                test_dataset = datasets.Imagenette(root='./data', split='val', download=False, transform=tf_test)
            else:
                raise
        
        train_dataset.targets = [y for _, y in train_dataset._samples]
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
    elif ds_type == "CIFAR10_BOP":
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ScaleToMinusOneOne(),
        ])

        train_dataset = datasets.CIFAR10(
            root="./dataset/data",
            train=True,
            transform=tf,
            download=True,
        )
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
    elif ds_type == "CIFAR100_N":
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ScaleToMinusOneOne(),
        ])

        train_dataset = datasets.CIFAR100(root="./dataset/data", train=True, transform=tf, download=True)
    elif ds_type == "FashionMNIST":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(root="./dataset/data", train=True, transform=tf, download=True)
    elif ds_type == "SVHN":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=tf)
        train_dataset.targets = train_dataset.labels
        train_dataset.classes = [str(i) for i in range(10)]
        train_dataset.class_to_idx = {str(i): i for i in range(10)}
    elif ds_type == "IMAGENETTE":
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)

        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        try:
            train_dataset = datasets.Imagenette(root='./data', split='train', download=True, transform=tf_train)
        except RuntimeError as e:
            if "already exists" in str(e) and "imagenette2" in str(e):
                train_dataset = datasets.Imagenette(root="./data", split="train", download=False, transform=tf_train)
            else:
                raise

        train_dataset.targets = [y for _, y in train_dataset._samples]
        dataset_label_list = train_dataset.targets
        
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
            worker_init_fn=make_worker_init_fn(base_seed, cid) if num_workers > 0 else None,
            generator=g,
            persistent_workers=bool(num_workers),
        )
        train_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=make_worker_init_fn(base_seed, 0) if num_workers > 0 else None,
        persistent_workers=bool(num_workers),
    )

    return train_loaders, test_loader


def make_worker_init_fn(base_seed: int, client_id: int):
    def _init_fn(worker_id):
        s = 10_000 + 97*base_seed + 31*client_id + worker_id
        import random, numpy as np, torch
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
    return _init_fn


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator("cpu")
    g.manual_seed(seed)
    return g

def _infer_ds_type_from_base(base_ds):
    if isinstance(base_ds, datasets.CIFAR10):   return "CIFAR10"
    if isinstance(base_ds, datasets.CIFAR100):  return "CIFAR100"
    if isinstance(base_ds, datasets.MNIST):     return "MNIST"
    if isinstance(base_ds, datasets.FashionMNIST): return "FashionMNIST"
    if isinstance(base_ds, datasets.SVHN): return "SVHN"
    if isinstance(base_ds, datasets.Imagenette): return "IMAGENETTE"
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
        worker_init_fn=make_worker_init_fn(base_seed, client_id) if num_workers > 0 else None,
        generator=g,
        persistent_workers=bool(num_workers),
    )



def _dirichlet_partition_equal_client_sizes(
    dataset_label_list,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
):
    """
    Create a label-skewed Dirichlet partition while keeping client dataset
    sizes equal (or differing by at most one when N is not divisible by K).

    A Dirichlet distribution is sampled independently for each class. Samples
    are then assigned in a globally shuffled order. Once a client's quota is
    full, that client is removed from the candidate set and the remaining
    probabilities are renormalized. A remaining-capacity factor is included
    to avoid filling one client too early and distorting the final classes.

    No sample is dropped or duplicated.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for Dirichlet partitioning.")
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0.")

    if torch.is_tensor(dataset_label_list):
        labels = dataset_label_list.detach().cpu().numpy().reshape(-1)
    else:
        labels = np.asarray(dataset_label_list).reshape(-1)

    total_size = int(labels.shape[0])
    if total_size < num_clients:
        raise ValueError(
            f"Dataset has only {total_size} samples for {num_clients} clients."
        )

    unique_labels = np.unique(labels)
    label_to_row = {label: row for row, label in enumerate(unique_labels)}

    # Equal capacity. If N is not divisible by K, sizes differ by at most one.
    base_size = total_size // num_clients
    remainder = total_size % num_clients
    target_sizes = np.full(num_clients, base_size, dtype=np.int64)

    # Randomize who receives the +1 samples to avoid client-id bias.
    if remainder:
        extra_clients = rng.permutation(num_clients)[:remainder]
        target_sizes[extra_clients] += 1

    remaining = target_sizes.copy()
    partitions = [[] for _ in range(num_clients)]

    # One client-probability vector per class.
    class_proportions = rng.dirichlet(
        [alpha] * num_clients,
        size=len(unique_labels),
    )

    # Global shuffling prevents a systematic bias toward classes processed last.
    for idx in rng.permutation(total_size):
        row = label_to_row[labels[idx]]
        probs = class_proportions[row].astype(np.float64, copy=True)

        active = remaining > 0
        probs[~active] = 0.0

        # Smoothly discourage clients that are close to their capacity.
        capacity_fraction = remaining.astype(np.float64) / target_sizes
        probs *= capacity_fraction

        prob_sum = probs.sum()
        if prob_sum <= 0.0:
            # Defensive fallback; normally only relevant to floating-point
            # underflow for very small alpha.
            probs = remaining.astype(np.float64)
            probs[~active] = 0.0
            prob_sum = probs.sum()

        probs /= prob_sum
        client_id = int(rng.choice(num_clients, p=probs))

        partitions[client_id].append(int(idx))
        remaining[client_id] -= 1

    if np.any(remaining != 0):
        raise RuntimeError(
            f"Balanced Dirichlet partition failed; remaining capacities: {remaining.tolist()}"
        )

    # Shuffle each client's index list so partition order does not carry class info.
    for client_id in range(num_clients):
        rng.shuffle(partitions[client_id])

    # Strong sanity checks: full coverage, no duplicates, expected client sizes.
    actual_sizes = np.asarray([len(p) for p in partitions], dtype=np.int64)
    if not np.array_equal(actual_sizes, target_sizes):
        raise RuntimeError(
            f"Unexpected client sizes. expected={target_sizes.tolist()}, "
            f"actual={actual_sizes.tolist()}"
        )

    flat = [idx for part in partitions for idx in part]
    if len(flat) != total_size or len(set(flat)) != total_size:
        raise RuntimeError(
            "Partition must contain every training sample exactly once."
        )

    return partitions

def create_datasets(
    train_ds_num=5,
    ds_type="MNIST",
    non_iid_level_alpha=0.1,
    train_batch_size=64,
    test_batch_size=64,
    use_dirichlet=False,
    equal_client_sizes=False,
    num_workers=0,
    save_graph=True,
    add_info_to_figure=False,
    path=None,
    base_seed: int = 0,
):
    logger.log_info(
        f"Dirichlet: {use_dirichlet}, Non-i.i.d Level: {non_iid_level_alpha}, "
        f"Equal Client Sizes: {equal_client_sizes}, Train Batch Size: {train_batch_size}, "
        f"Test Batch Size: {test_batch_size}"
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

        if equal_client_sizes:
            subset_indices_lists = _dirichlet_partition_equal_client_sizes(
                dataset_label_list=dataset_label_list,
                num_clients=train_ds_num,
                alpha=non_iid_level_alpha,
                rng=rng,
            )
        else:
            # Original FedWork behavior: Dirichlet label skew may also create
            # unequal client dataset sizes.
            class_proportions = rng.dirichlet(
                [non_iid_level_alpha] * train_ds_num,
                train_classes_num,
            )
            subset_indices_lists = [[] for _ in range(train_ds_num)]
            for class_index in range(train_classes_num):
                class_indices = [
                    i for i, lbl in enumerate(dataset_label_list)
                    if lbl == class_index
                ]
                class_indices = rng.permutation(class_indices).tolist()
                sizes = rng.multinomial(
                    len(class_indices),
                    class_proportions[class_index],
                )
                offset = 0
                for client_id, size in enumerate(sizes):
                    assigned_indices = class_indices[offset : offset + size]
                    offset += size
                    subset_indices_lists[client_id].extend(assigned_indices)

        partitions = subset_indices_lists

        logger.log_info(
            f"Client dataset sizes: {[len(p) for p in partitions]}"
        )

        for client_id in range(train_ds_num):
            assigned_indices = subset_indices_lists[client_id]
            client_labels = [dataset_label_list[i] for i in assigned_indices]
            class_counts = Counter(client_labels)
            client_distributions.append(class_counts)

            logger.log_info(
                f"Client{client_id}: n={len(assigned_indices)}, "
                f"class_counts={dict(sorted(class_counts.items()))}"
            )

            if isinstance(train_dataset_eval.targets, list):
                classes.append([
                    train_dataset_eval.targets[i] for i in assigned_indices
                ])
            else:
                classes.append([
                    train_dataset_eval.targets[i].item() for i in assigned_indices
                ])

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
                f"{type(train_dataset_eval).__name__} \nDirichlet: {use_dirichlet}\n"
                f"Non-i.i.d level: {non_iid_level_alpha}\n"
                f"Equal client sizes: {equal_client_sizes}\n"
                f"Train batch size: {train_batch_size}\n"
                f"Test batch size: {test_batch_size}",
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
