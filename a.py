# Re-import necessary libraries after execution state reset
import torch
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define the cifar_beta function using the user's code
def cifar_beta(dataset, beta, n_clients):  
    print("The dataset is split with non-iid param ", beta)
    label_distributions = []
    
    for y in range(len(dataset.classes)):  # FIX: dataset.dataset.classes → dataset.classes
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))  
    
    labels = np.array(dataset.targets).astype(np.int32)  # FIX: dataset.dataset.targets → dataset.targets
    client_idx_map = {i:{} for i in range(n_clients)}
    client_size_map = {i:{} for i in range(n_clients)}

    for y in range(len(dataset.classes)):  # FIX: dataset.dataset.classes → dataset.classes
        label_y_idx = np.where(labels == y)[0]  
        label_y_size = len(label_y_idx)
        
        sample_size = (label_distributions[y] * label_y_size).astype(np.int32)
        sample_size[n_clients-1] += label_y_size - np.sum(sample_size)

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)

        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i-1] if i > 0 else 0):sample_interval[i]]
            client_size_map[i][y] = sample_size[i]

    client_distributions = []
    client_datasets = []
    all_idxs = [i for i in range(len(dataset))]

    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset, client_i_idx)  # FIX: dataset.dataset → dataset
        client_datasets.append(subset)

        # Count the occurrences of each class
        client_labels = [dataset.targets[idx] for idx in client_i_idx]  # FIX: dataset.dataset.targets → dataset.targets
        class_counts = {cls: client_labels.count(cls) for cls in range(len(dataset.classes))}  # FIX: dataset.dataset.classes → dataset.classes
        client_distributions.append(class_counts)

    return client_datasets, client_distributions


# Load CIFAR-100 dataset
stats = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Load dataset
cifar100_train = datasets.CIFAR100(root='./dataset/data', train=True, transform=transform, download=True)

# Apply Dirichlet distribution
beta = 0.1
num_clients = 10
_, client_distributions = cifar_beta(cifar100_train, beta, num_clients)

# Prepare data for visualization
distribution_matrix = np.zeros((num_clients, len(cifar100_train.classes)))

for client_idx, dist in enumerate(client_distributions):
    for cls, count in dist.items():
        distribution_matrix[client_idx, cls] = count

# Create a stacked bar plot
fig, ax = plt.subplots(figsize=(12, 6))

# Bottom tracker for stacking bars
bottom = np.zeros(num_clients)

# Plot stacked bars
for cls in range(len(cifar100_train.classes)):
    ax.bar(range(num_clients), distribution_matrix[:, cls], bottom=bottom, label=f'Class {cls}' if cls < 10 else "", alpha=0.8)
    bottom += distribution_matrix[:, cls]

# Labels and title
ax.set_xlabel('Clients')
ax.set_ylabel('Number of Samples')
ax.set_title(f'Non-IID Label Distribution Across Clients (Dirichlet, Beta={beta})')
ax.set_xticks(range(num_clients))
ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)

# Display plot
plt.tight_layout()
plt.savefig("223.pdf", format="pdf", bbox_inches="tight")

