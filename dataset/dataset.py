import torch
import random
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from datetime import datetime
from utils.logger import *
import utils.consts as consts
#from sklearn.manifold import TSNE
from collections import Counter
import numpy as np

def create_datasets(train_ds_num=5, ds_type="MNIST", heterogeneous=False, non_iid_level=0.1, train_batch_size=64, test_batch_size=64, num_workers=8, save_graph=True, add_info_to_figure=False, path=None):
    logger.log_info(f'Heterogeneous: {heterogeneous}, Non-i.i.d Level: {non_iid_level}, Train Batch Size: {train_batch_size}, Test Batch Size: {test_batch_size}')
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
    train_classes_num = len(train_dataset.classes)
    train_datasets = []
    train_dataset_subsets_len = []
    train_total_dataset_size = len(train_dataset)
    train_groups_eq_size = train_total_dataset_size // train_ds_num

    

    for _ in range(train_ds_num - 1):
        group_dataset_len = train_groups_eq_size + (torch.randint(int((non_iid_level / 2.0) * train_groups_eq_size) // -2, int((non_iid_level / 2.0) * train_groups_eq_size) // 2, (1,)).item() if heterogeneous and non_iid_level > 0.0 else 0)
        train_dataset_subsets_len.append(group_dataset_len)
        train_total_dataset_size -= group_dataset_len
    
    train_dataset_subsets_len.append(train_total_dataset_size)

    
    train_dataset_subsets_expected_len = list(train_dataset_subsets_len)
    if train_classes_num > train_ds_num:
        train_groups_labels = [[((train_classes_num // train_ds_num) * i + j) for j in range(train_classes_num // train_ds_num)] for i in range(train_ds_num)]
    else:
        train_groups_labels = [[torch.randint(0, train_classes_num, (1,)).item()] for i in range(train_ds_num)]

    for label in range(train_classes_num):
        if any(label in group_labels for group_labels in train_groups_labels) == False:
            train_groups_labels[torch.randint(0, train_ds_num, (1,)).item()].append(label)

    
    
    client_distributions = []
    for subset_index in range(train_ds_num):
        train_group_index_list = []
        label_len_list = []
        labels_list_size = len(train_groups_labels[subset_index])

        each_label_standard_size = train_dataset_subsets_len[subset_index] // labels_list_size
        for label in range(labels_list_size - 1):
            label_len = each_label_standard_size + (torch.randint(int((non_iid_level / 2.0) * each_label_standard_size) // -2, int((non_iid_level / 2.0) * each_label_standard_size) // 2, (1,)).item() if non_iid_level > 0.0 else 0)
            label_len_list.append(label_len)
            train_dataset_subsets_len[subset_index] -= label_len
        label_len_list.append(train_dataset_subsets_len[subset_index])



        for label_index,label in enumerate(train_groups_labels[subset_index]):
            
            indices = [index for index, value in enumerate(dataset_label_list) if ((value == label) and ((index not in train_group_index_list) or (torch.rand(1).item() < 0.25)))]
            
            needed_length = int(label_len_list[label_index] * non_iid_level)
            random_indices = random.sample(range(len(indices)), min(len(indices), needed_length) )
            needed_indices = [indices[i] for i in random_indices]

            train_group_index_list += needed_indices

        
        remainded_indices = train_dataset_subsets_expected_len[subset_index] - len(train_group_index_list)
        
        for _ in range(remainded_indices):
            rand_index = torch.randint(0, len(dataset_label_list), (1,)).item()
            train_group_index_list.append(rand_index)


        if isinstance(train_dataset.targets, list):
            classes.append([train_dataset.targets[i] for i in train_group_index_list])
        else:
            classes.append([train_dataset.targets[i].item() for i in train_group_index_list])


        random.shuffle(train_group_index_list)
        dataset_client = Subset(train_dataset, train_group_index_list)
        train_datasets.append(DataLoader(dataset_client, batch_size=train_batch_size,
                                   shuffle=True, num_workers=num_workers))
        
        # Get the targets/labels for the current client
        client_labels = [train_dataset[i][1] for i in train_group_index_list]
        # Count the occurrences of each class
        class_counts = Counter(client_labels)
        client_distributions.append(class_counts)
    
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
            plt.text(len(train_datasets) + 1, len(unique_classes) / 2 , f'{type(train_dataset).__name__} \nHeterogeneous: {heterogeneous}\nNon-i.i.d level: {non_iid_level}\nTrain batch size: {train_batch_size}\nTest batch size: {test_batch_size}', fontsize=8, color='red', rotation=90, va='center', ha='center', bbox=dict(facecolor='white', alpha=0.5))
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
            ax.bar(range(len(client_distributions)), class_distribution, bottom=bottom, color=colors[class_index], label=f'Class {cls}')
            bottom += class_distribution
        
        # Customize the plot
        #ax.set_xlabel('Client Index')
        #ax.set_ylabel('Proportion of Classes')
        #ax.set_title('Non-IID Data Distribution Across Clients')
        #ax.legend(title='Class')

        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.xticks([i for i in range(len(all_classes))], [f"C#{i}" for i in range(len(all_classes))])

        full_path = os.path.join(dir_path, f"dataset_distribution_{time_str}_sbp.pdf")
        plt.savefig(full_path, format="pdf", bbox_inches="tight")

    return train_datasets, test_dataset_loader
