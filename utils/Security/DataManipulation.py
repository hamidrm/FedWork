import numpy as np
import torch


class MixUpDefense:
    def __init__(self, alpha) -> None:
        if alpha > 0:
            self.mixup_ratio = np.random.beta(alpha, alpha)
        else:
            self.mixup_ratio = 1.0

    def get_data(self, inputs, targets):
        batch_size = inputs.size(0)
        shuffled_indices = torch.randperm(batch_size)

        mixed_inputs = self.mixup_ratio * inputs + (1 - self.mixup_ratio) * inputs[shuffled_indices, :]

        mixed_targets = self.mixup_ratio * targets + (1 - self.mixup_ratio) * targets[shuffled_indices]
        
        return mixed_inputs, mixed_targets, targets[shuffled_indices], self.mixup_ratio

    def criterion(self, criterion_fn, pred, y_actual, y_mixed):
        return self.mixup_ratio * criterion_fn(pred, y_actual) + (1 - self.mixup_ratio) * criterion_fn(pred, y_mixed)
    
    def correctness(self, pred, y_actual, y_mixed):
        _, predicted = torch.max(pred, 1)
        correct_a = (predicted == y_actual).sum().item()
        correct_b = (predicted == y_mixed).sum().item()
        return self.mixup_ratio * correct_a + (1 - self.mixup_ratio) * correct_b

class InstaHideDataObfuscator:
    def __init__(self, private_loader, public_loader=None, num_mix=3, max_weight=0.7):
        self.private_loader = private_loader
        self.public_loader = public_loader
        self.num_mix = num_mix
        self.max_weight = max_weight

    def generate_mix_weights(self, batch_size):
        weights = np.random.rand(batch_size, self.num_mix)
        weights /= weights.sum(axis=1, keepdims=True)
        weights = np.clip(weights, 0, self.max_weight)
        return torch.from_numpy(weights).float()

    def apply_instahide(self, inputs, targets, public_data=None):
        batch_size = inputs.size(0)
        mix_weights = self.generate_mix_weights(batch_size)
        mixed_inputs = mix_weights[:, 0].view(-1, 1, 1, 1) * inputs
        mixed_labels = mix_weights[:, 0].view(-1, 1) * targets

        for i in range(1, self.num_mix):
            if public_data is not None and np.random.rand() > 0.5:
                rand_data, rand_labels = public_data
            else:
                rand_data, rand_labels = inputs, targets
            perm_indices = torch.randperm(batch_size)
            permuted_data = rand_data[perm_indices]
            permuted_labels = rand_labels[perm_indices]
            mixed_inputs += mix_weights[:, i].view(-1, 1, 1, 1) * permuted_data
            mixed_labels += mix_weights[:, i].view(-1, 1) * permuted_labels

        sign_flip = (torch.randint(2, size=mixed_inputs.shape) * 2.0 - 1).float()
        mixed_inputs *= sign_flip

        return mixed_inputs, mixed_labels

    def generate_obfuscated_batch(self):

        inputs, targets = next(iter(self.private_loader))

        public_data = None
        if self.public_loader:
            pub_inputs, pub_targets = next(iter(self.public_loader))
            public_data = (pub_inputs, pub_targets)

        obfuscated_inputs, obfuscated_labels = self.apply_instahide(inputs, targets.float(), public_data)
        return obfuscated_inputs, obfuscated_labels