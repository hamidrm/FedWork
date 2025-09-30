import torch
import math

class LDPDefense:
    def __init__(self, epsilon, delta, clipping_bound):
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_bound = clipping_bound
        self.noise_std = self.compute_noise_std()

    def compute_noise_std(self):
        return (self.clipping_bound * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon

    def clip_and_add_noise(self, state_dict):

        # Flatten all parameters for norm calculation
        all_params = torch.cat([p.flatten() for p in state_dict.values()])
        norm = torch.norm(all_params, p=2)

        # Clip if needed
        scale = min(1.0, self.clipping_bound / (norm + 1e-6))  # small epsilon for stability

        noisy_state = {}

        for k, v in state_dict.items():
            clipped_param = v * scale
            noise = torch.randn_like(clipped_param) * self.noise_std
            noisy_state[k] = clipped_param + noise

        return noisy_state
