import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

# MedMNIST imports
import medmnist
from medmnist import INFO, PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST
from medmnist import  OrganAMNIST, OrganSMNIST, OrganMNIST3D
from medmnist import TissueMNIST, BloodMNIST
from medmnist.dataset import MedMNIST2D

MEDMNIST2D_LIST = {
    'pathmnist',
    'chestmnist',
    'dermamnist',
    'octmnist',
    'pneumoniamnist',
    'retinamnist',
    'breastmnist',
    'organamnist',
    'organcmnist',
    'organsmnist',
    'tissuemnist',
    'bloodmnist'
}


class MedMNISTDataset(MedMNIST2D):
    def __init__(self, dataset_name: str, root: str, split: str, transform: transforms.Compose, download: bool = False):
        self.flag = dataset_name
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            download=download
        )

class MedMNIST(MedMNIST2D):
    def __init__(self, dataset_name: str, root: str, train: bool, transform: transforms.Compose, download: bool = False):
        dataset_name = dataset_name.lower()
        if dataset_name not in MEDMNIST2D_LIST:
            raise ValueError(f"Dataset '{dataset_name}' not recognized.")
        self.flag = dataset_name
        labels = INFO[dataset_name]["label"]
        
        self.classes = [c for c in labels.values()]

        
        if train:
            super().__init__(
                root=root,
                split="train",
                transform=transform,
                download=download
            )
            val_dataset = MedMNISTDataset(
                dataset_name=dataset_name,
                root=root,
                split="val",
                transform=transform,
                download=download
            )

            self.imgs = np.concatenate([self.imgs, val_dataset.imgs], axis=0)
            self.labels = np.concatenate([self.labels, val_dataset.labels], axis=0)


            self.targets = [int(l[0].item()) for l in self.labels]
            self.targets = torch.tensor(self.targets, dtype=torch.int)
        else:
            
            super().__init__(
                root=root,
                split="test",
                transform=transform,
                download=download
            )
            self.targets = [int(l[0].item()) for l in self.labels]
            self.targets = torch.tensor(self.targets, dtype=torch.int)
    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index):

        img, target = super().__getitem__(index)

        return img, int(target[0].item())