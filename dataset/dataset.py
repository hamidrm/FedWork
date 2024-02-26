import torch
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def create_datasets(train_ds_num=5, ds_type="MNIST", heterogeneous=False, non_iid_level=0.1, train_batch_size=64, test_batch_size=64):
    
    if ds_type == "MNIST":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ),(0.5, ))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ),(0.5, ))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test)
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
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
        
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
        if torch.rand(1).item() < non_iid_level:
            train_groups_labels = [[torch.randint(0, train_classes_num, (1,)).item()] for i in range(train_ds_num)]
        else:
            train_groups_labels = [[(i+int( (torch.rand(1).item() - 0.5) * non_iid_level * train_ds_num * 2)) % train_ds_num] for i in range(train_ds_num)]

    for label in range(train_classes_num):
        if any(label in group_labels for group_labels in train_groups_labels) == False:
            train_groups_labels[torch.randint(0, train_ds_num, (1,)).item()].append(label)

    dataset_label_list = train_dataset.targets

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
            indices = [index for index, value in enumerate(dataset_label_list) if value == label]
            needed_indices = indices[0:int(label_len_list[label_index] * non_iid_level)]
            train_group_index_list += needed_indices

            _dataset_label_list = [item for item in dataset_label_list if (item not in needed_indices) or (torch.rand(1).item() < 0.75)]
            dataset_label_list = _dataset_label_list

        remainded_indices = train_dataset_subsets_expected_len[subset_index] - len(train_group_index_list)
        
        for _ in range(remainded_indices):
            rand_index = torch.randint(0, len(dataset_label_list), (1,)).item()
            train_group_index_list.append(rand_index)

            _dataset_label_list = [item for item in dataset_label_list if (item is not rand_index) or (torch.rand(1).item() < 0.75)]
            dataset_label_list = _dataset_label_list

        random.shuffle(train_group_index_list)
        dataset_client = Subset(train_dataset, train_group_index_list)
        train_datasets.append(DataLoader(dataset_client, batch_size=train_batch_size,
                                   shuffle=True, num_workers=16))
    test_dataset_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,  num_workers=16)
    return train_datasets, test_dataset_loader
