import torch
import random
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from datetime import datetime
from dataset import MedMNIST
from utils.logger import *
import utils.consts as consts

#from sklearn.manifold import TSNE
from collections import Counter
import numpy as np

def create_datasets(train_ds_num=5, ds_type="MNIST", heterogeneous=False, non_iid_level_alpha=0.1, train_batch_size=64, test_batch_size=64, use_dirichlet=False, num_workers=8, save_graph=True, add_info_to_figure=False, path=None):
    logger.log_info(f'Dirichlet: {use_dirichlet}, Heterogeneous: {heterogeneous}, Non-i.i.d Level: {non_iid_level_alpha}, Train Batch Size: {train_batch_size}, Test Batch Size: {test_batch_size}')
    classes = []
    if ds_type == "MNIST":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ),(0.5, ))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ),(0.5, ))
        ])
        train_dataset = datasets.MNIST(root='./dataset/data', train=True, transform=transform_train, download=True)
        test_dataset = datasets.MNIST(root='./dataset/data', train=False, transform=transform_test)
        dataset_label_list = train_dataset.targets.tolist()
    elif ds_type == "CIFAR10":
        stats = (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        train_dataset = datasets.CIFAR10(root='./dataset/data', train=True, transform=transform_train, download=True)
        test_dataset  = datasets.CIFAR10(root='./dataset/data', train=False, transform=transform_test)
        dataset_label_list = train_dataset.targets
    elif ds_type == "CIFAR100":
        stats = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        train_dataset = datasets.CIFAR100(root='./dataset/data', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR100(root='./dataset/data', train=False, transform=transform_test)
        dataset_label_list = train_dataset.targets
    elif ds_type == "FashionMNIST":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ),(0.5, ))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ),(0.5, ))
        ])
        train_dataset = datasets.FashionMNIST(root='./dataset/data', train=True, transform=transform_train, download=True)
        test_dataset  = datasets.FashionMNIST(root='./dataset/data', train=False, transform=transform_test)
        dataset_label_list = train_dataset.targets.tolist()
    elif ("-" in ds_type) and ds_type.split("-")[0].lower() == "medmnist":
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = MedMNIST.MedMNIST(dataset_name=ds_type.split("-")[1], root='./dataset/data', train=True, transform=transform_train, download=True)
        test_dataset = MedMNIST.MedMNIST(dataset_name=ds_type.split("-")[1], root='./dataset/data', train=False, transform=transform_test)
        dataset_label_list = train_dataset.targets.tolist()
    else:
        raise ValueError(f"Dataset '{ds_type}' not recognized.")
    

    train_classes_num = len(train_dataset.classes)
    train_datasets = []
    train_dataset_subsets_len = []
    train_total_dataset_size = len(train_dataset)
    train_groups_eq_size = train_total_dataset_size // train_ds_num
    client_distributions = []

    if non_iid_level_alpha == 0 and not use_dirichlet:
        # IID datasets
        # Generate and shuffle all indices
        all_indices = list(range(train_total_dataset_size))
        random.shuffle(all_indices)


        split_indices = [all_indices[i * train_groups_eq_size:(i + 1) * train_groups_eq_size] for i in range(train_ds_num)]


        remainder = train_total_dataset_size % train_ds_num
        for i in range(remainder):
            split_indices[i].append(all_indices[train_ds_num * train_groups_eq_size + i])

        for subset_indices in split_indices:
            subset = Subset(train_dataset, subset_indices)
            loader = DataLoader(subset, batch_size=test_batch_size, shuffle=True,  num_workers=num_workers)
            train_dataset_subsets_len.append(len(subset_indices))
            train_datasets.append(loader)
        
            if isinstance(train_dataset.targets, list):
                classes.append([train_dataset.targets[i] for i in subset_indices])
            else:
                classes.append([train_dataset.targets[i].item() for i in subset_indices])

            client_labels = [train_dataset[i][1] for i in subset_indices]
            # Count the occurrences of each class
            class_counts = Counter(client_labels)
            client_distributions.append(class_counts)
    else:

        # Compute Dirichlet distribution for non-iid partitioning if enabled
        if use_dirichlet:
            dirichlet_alpha = non_iid_level_alpha
            # For each client, sample proportions for each class from a Dirichlet distribution
            class_proportions = np.random.dirichlet([dirichlet_alpha] * train_ds_num, train_classes_num)

            subset_indices_lists = [[] for _ in range(train_ds_num)]
            for class_index in range(train_classes_num):
                class_indices = [i for i, lbl in enumerate(dataset_label_list) 
                                if lbl == class_index]

                random.shuffle(class_indices)

                frac_sizes = class_proportions[class_index] * len(class_indices)
                sizes = frac_sizes.astype(int) 
                leftover = len(class_indices) - np.sum(sizes)

                sizes[-1] += leftover

                offset = 0
                for client_id, size in enumerate(sizes):
                    assigned_indices = class_indices[offset : offset + size]
                    offset += size
                    subset_indices_lists[client_id].extend(assigned_indices)
            train_datasets = []
            for client_id in range(train_ds_num):

                random.shuffle(subset_indices_lists[client_id])

                # Create a Subset and then a DataLoader
                dataset_client = Subset(train_dataset, subset_indices_lists[client_id])
                loader_client = DataLoader(dataset_client,
                                        batch_size=train_batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)
                train_datasets.append(loader_client)

                assigned_indices = subset_indices_lists[client_id]

                client_labels = [dataset_label_list[i] for i in assigned_indices]

                class_counts = Counter(client_labels)
                client_distributions.append(class_counts)

                if isinstance(train_dataset.targets, list):
                    classes.append([train_dataset.targets[i] for i in assigned_indices])
                else:
                    classes.append([train_dataset.targets[i].item() for i in assigned_indices])
        
    test_dataset_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,  num_workers=num_workers)

    if save_graph:
        unique_classes = sorted(set(dataset_label_list))

        graph_map = [[0 for _ in range(len(train_datasets))] for _ in range(len(unique_classes))]
        for client_index in range(len(train_datasets)):
            for class_index in unique_classes:
                graph_map[class_index][client_index] = classes[client_index].count(class_index)
                
        max_val = len(train_dataset.targets)
        normalized_matrix = [[val / max_val for val in row] for row in graph_map]
        

        # Calculate figure size based on the number of x labels
        figure_width = max(len(train_datasets) / 2, 8)  # Adjust this factor as needed
        figure_height = max(len(unique_classes) / 2, 6)  # Adjust this factor as needed

        plt.figure(figsize=(figure_width, figure_height))
        plt.imshow(normalized_matrix, cmap='gray_r', interpolation='nearest')

        if add_info_to_figure:
            plt.text(len(train_datasets) + 1, len(unique_classes) / 2 , f'{type(train_dataset).__name__} \nHeterogeneous: {heterogeneous}\nNon-i.i.d level: {non_iid_level_alpha}\nTrain batch size: {train_batch_size}\nTest batch size: {test_batch_size}', fontsize=8, color='red', rotation=90, va='center', ha='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.xlim(-1, len(train_datasets))
        plt.ylim(-1, len(unique_classes))
        if path is None:
            output_directory = os.path.join(os.getcwd(), consts.OUTPUT_DIR)
            dir_path = os.path.join(output_directory, f"dataset")
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        else:
            dir_path = path
        plt.yticks([i for i in range(len(unique_classes))], train_dataset.classes)
        plt.xticks([i for i in range(train_ds_num)], [(i+1) for i in range(train_ds_num)])
        # Check if the directory exists, if not, create it
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        current_time = datetime.now()
        time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        full_path = os.path.join(dir_path, f"dataset_distribution_{time_str}.pdf")
        plt.savefig(full_path, format="pdf", bbox_inches="tight")

        client_num=0
        # Convert to arrays for easy plotting
        all_classes = sorted({cls for dist in client_distributions for cls in dist})
        distribution_matrix = np.zeros((len(client_distributions), len(all_classes)))

        for i, dist in enumerate(client_distributions):
            for cls, count in dist.items():
                class_index = all_classes.index(cls)
                distribution_matrix[i, class_index] = count

        # Normalize distributions to show proportions
        #distribution_matrix /= distribution_matrix.sum(axis=1, keepdims=True)

        # Plotting the stacked bar plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each class as a separate section in the bar
        bottom = np.zeros(len(client_distributions))
        colors = ["#35478c","#495696","#5c65a1","#6e75ab","#8085b5","#9295c0","#a4a6ca","#b6b7d5","#c8c9df","#dadbea"]#plt.cm.tab20(np.linspace(0, 1, len(all_classes)))  # Use a colormap for colors

        for class_index, cls in enumerate(all_classes):
            class_distribution = distribution_matrix[:, class_index]
            ax.bar(range(len(client_distributions)), class_distribution, bottom=bottom, label=f'Class {cls}')
            bottom += class_distribution
        
        # Customize the plot
        #ax.set_xlabel('Client Index')
        #ax.set_ylabel('Proportion of Classes')
        #ax.set_title('Non-IID Data Distribution Across Clients')
        #ax.legend(title='Class')

        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.xticks([i for i in range(len(train_datasets))], [f"{i}" for i in range(len(train_datasets))])

        full_path = os.path.join(dir_path, f"dataset_distribution_{time_str}_sbp.pdf")
        plt.savefig(full_path, format="pdf", bbox_inches="tight")

    return train_datasets, test_dataset_loader