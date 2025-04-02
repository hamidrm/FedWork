import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DPSGDOptimizer(optim.Optimizer):

    def __init__(self, noise_multiplier, max_grad_norm, device="cpu", optimizer=None):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.optimizer = optimizer


    def set_platform(self, device: str):
        self.device = device
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups  # Retain optimizer parameter groups

    def clip_and_add_noise(self):

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    # Compute norm of gradient
                    grad_norm = torch.norm(param.grad, p=2)
                    
                    # Clip gradient if necessary
                    clip_coef = self.max_grad_norm / (grad_norm + 1e-6)
                    if clip_coef < 1:
                        param.grad *= clip_coef
                    
                    # Add Gaussian noise
                    noise = torch.normal(0, self.noise_multiplier * self.max_grad_norm, param.grad.shape).to(self.device)
                    param.grad += noise

    def step(self, closure=None):
        if self.optimizer is None:
            raise ValueError(f"Optimizer object not found!")
        self.clip_and_add_noise()  # Apply DP mechanisms
        self.optimizer.step(closure)  # Step with original optimizer

    def zero_grad(self, set_to_none=True):
        if self.optimizer is None:
            raise ValueError(f"Optimizer object not found!")
        self.optimizer.zero_grad(set_to_none)