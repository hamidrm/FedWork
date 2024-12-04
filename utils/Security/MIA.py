import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.stats import norm

class MIACommon:
    @staticmethod
    def calculate_auc_metrics(val_scores, train_scores):
        # Labels and scores concatenation
        labels = torch.cat([torch.zeros_like(val_scores), torch.ones_like(train_scores)])
        scores = torch.cat([val_scores, train_scores])

        # Compute ROC curve using PyTorch
        sorted_indices = torch.argsort(scores, descending=True)
        labels = labels[sorted_indices]
        tps = torch.cumsum(labels, dim=0)
        fps = torch.cumsum(1 - labels, dim=0)

        # Calculate FPR and TPR
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        # Calculate AUC
        auc = torch.trapz(tpr, fpr).item()

        # Log-space AUC calculation
        log_tpr = torch.log10(torch.clamp(tpr, 1e-5, 1))
        log_fpr = torch.log10(torch.clamp(fpr, 1e-5, 1))
        log_tpr = (log_tpr + 5) / 5.0
        log_fpr = (log_fpr + 5) / 5.0
        log_auc = torch.trapz(log_tpr, log_fpr).item()

        # TPRs at specific FPR thresholds
        fpr_thresholds = torch.tensor([0.1, 0.02, 0.01, 0.001, 0.0001], device=fpr.device)
        tprs_at_thresholds = {}
        for threshold in fpr_thresholds:
            valid_indices = fpr < threshold
            if torch.any(valid_indices):
                tprs_at_thresholds[str(threshold.item())] = tpr[torch.where(valid_indices)[0][-1]].item()
            else:
                tprs_at_thresholds[str(threshold.item())] = 0.0

        return {
            "auc": auc,
            "log_auc": log_auc,
            "tprs": tprs_at_thresholds
        }

    @staticmethod
    def optimize_experimental_data(data_loader, model, optimizer, loss_fn):
        model.train()
        experimental_grad_batch_list = []

        for x, y in data_loader:
            optimizer.zero_grad()

            # Forward pass
            pred = model(x)

            # Compute loss
            loss = loss_fn(pred, y)

            # Backward pass
            loss.backward()

            # Collect gradients for each parameter
            experimental_grad_batch = []
            for param in model.parameters():
                if param.grad is not None:
                    experimental_grad_batch.append(param.grad.view(-1))  # Flatten gradients
            if experimental_grad_batch:  # Ensure list is not empty
                experimental_grad_batch_list.append(torch.cat(experimental_grad_batch))

        if experimental_grad_batch_list:
            return torch.stack(experimental_grad_batch_list)
        else:
            return None

    @staticmethod
    def calculate_gradient_difference(model, global_model):
        grad_diff = []
        with torch.no_grad():  # Ensure no gradients are computed
            for (name, param), (_, global_param) in zip(model.named_parameters(), global_model.named_parameters()):
                if param.requires_grad:
                    grad_diff.append((param.data - global_param.data).view(-1))  # Flatten differences
        return torch.cat(grad_diff) if grad_diff else torch.tensor([], device=next(model.parameters()).device)

class CosMIA:
    def __init__(self, train_data_loader, validation_data_loader, optimizer, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scores = []

    def get_auc_metrics(self, device):
        # Compute mean validation and training scores directly on the same device
        validation_scores = torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        train_scores = torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)

        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        # Calculate AUC metrics for the most recent scores
        return MIACommon.calculate_auc_metrics(self.scores[-1][0], self.scores[-1][1])

    def execute(self, target_model, global_model, device):
        # Optimize experimental data to compute gradients
        train_gradients = MIACommon.optimize_experimental_data(
            self.train_data_loader, global_model, self.optimizer, self.loss_fn
        ).to(device)
        validation_gradients = MIACommon.optimize_experimental_data(
            self.validation_data_loader, global_model, self.optimizer, self.loss_fn
        ).to(device)

        # Compute target gradient difference
        target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model).to(device)

        # Calculate cosine similarity for training and validation
        target_cosine_similarity_train_list = F.cosine_similarity(
            train_gradients, target_grad_diff.unsqueeze(0), dim=1
        )

        target_cosine_similarity_validation_list = F.cosine_similarity(
            validation_gradients, target_grad_diff.unsqueeze(0), dim=1
        )

        # Append scores to the list
        self.scores.append((target_cosine_similarity_validation_list, target_cosine_similarity_train_list))


class GradDiffMIA:
    def __init__(self, train_data_loader, validation_data_loader, optimizer, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scores = []

    def get_auc_metrics(self, device):
        # Compute mean validation and training scores on the same device
        validation_scores = torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        train_scores = torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        # Calculate AUC metrics for the most recent scores
        return MIACommon.calculate_auc_metrics(self.scores[-1][0], self.scores[-1][1])

    def execute(self, target_model, global_model, device):
        # Optimize experimental data to compute gradients
        train_gradients = MIACommon.optimize_experimental_data(
            self.train_data_loader, global_model, self.optimizer, self.loss_fn
        ).to(device)
        validation_gradients = MIACommon.optimize_experimental_data(
            self.validation_data_loader, global_model, self.optimizer, self.loss_fn
        ).to(device)

        # Compute target gradient difference
        target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model).to(device)

        # Compute gradient differences for training and validation
        grad_diff_train = (
            torch.norm(target_grad_diff.unsqueeze(0), p=2, dim=1) ** 2
            - torch.norm(target_grad_diff.unsqueeze(0) - train_gradients, p=2, dim=1) ** 2
        )
        grad_diff_validation = (
            torch.norm(target_grad_diff.unsqueeze(0), p=2, dim=1) ** 2
            - torch.norm(target_grad_diff.unsqueeze(0) - validation_gradients, p=2, dim=1) ** 2
        )

        # Append scores to the list
        self.scores.append((grad_diff_validation, grad_diff_train))


class LossMIA:
    def __init__(self, train_data_loader, validation_data_loader, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.loss_fn = loss_fn
        self.scores = []

    def get_auc_metrics(self, device):
        # Compute mean validation and training scores directly on the same device
        validation_scores = torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        train_scores = torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        # Calculate AUC metrics for the most recent scores
        return MIACommon.calculate_auc_metrics(self.scores[-1][0], self.scores[-1][1])

    def execute(self, target_model, global_model, device):
        # Compute losses for training and validation datasets
        train_loss = self.get_model_res(self.train_data_loader, target_model, self.loss_fn, device)
        validation_loss = self.get_model_res(self.validation_data_loader, target_model, self.loss_fn, device)
        self.scores.append((validation_loss, train_loss))

    def get_model_res(self, data_loader, model, loss_fn, device):
        model.eval()
        batch_losses = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                # Ensure inputs and targets are moved to the same device as the model
                inputs, targets = inputs.to(device), targets.to(device)

                # Compute predictions and loss
                pred = model(inputs)
                loss = loss_fn(pred, targets).item()
                batch_losses.append(loss)
        
        # Return losses as a tensor on the correct device
        return torch.tensor(batch_losses, device=device)


class FedMIA:
    def __init__(self, train_data_loader, validation_data_loader, optimizer, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scores = []

    def get_auc_metrics(self, device):
        # Convert scores to tensors and move to the appropriate device
        validation_scores = torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).to(device)
        train_scores = torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).to(device)
        
        # Compute the scores with tensor-based operations
        train_scores = -torch.nanmean(1 - torch.log(train_scores), dim=0)
        validation_scores = -torch.nanmean(1 - torch.log(validation_scores), dim=0)

        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        return MIACommon.calculate_auc_metrics(self.scores[-1][0], self.scores[-1][1])

    def execute(self, shadow_models, target_model, global_model, device):
        # Ensure all tensors are on the correct device
        train_gradients = MIACommon.optimize_experimental_data(
            self.train_data_loader, global_model, self.optimizer, self.loss_fn
        ).to(device)
        validation_gradients = MIACommon.optimize_experimental_data(
            self.validation_data_loader, global_model, self.optimizer, self.loss_fn
        ).to(device)

        # Calculate scores
        train_scores = self.calculate_fedmia_score(shadow_models, target_model, global_model, train_gradients, device)
        validation_scores = self.calculate_fedmia_score(shadow_models, target_model, global_model, validation_gradients, device)

        self.scores.append((validation_scores, train_scores))

    def calculate_fedmia_score(self, shadow_models, target_model, global_model, experimental_grad_batch_list, device):
        if experimental_grad_batch_list is None:
            return None

        # Calculate gradients and cosine similarities on the same device
        target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model).to(device)
        shadow_grad_diffs = [MIACommon.calculate_gradient_difference(shadow_model, global_model).to(device) for shadow_model in shadow_models]

        shadow_cosine_similarity_list = []
        for shadow_grad_diff in shadow_grad_diffs:
            shadow_cosine_similarity_list_per_batch = F.cosine_similarity(
                experimental_grad_batch_list, shadow_grad_diff.unsqueeze(0), dim=1
            )
            shadow_cosine_similarity_list.append(shadow_cosine_similarity_list_per_batch)

        target_cosine_similarity_list = F.cosine_similarity(
            experimental_grad_batch_list, target_grad_diff.unsqueeze(0), dim=1
        )

        shadow_cosine_similarity_tensor = torch.stack(shadow_cosine_similarity_list, dim=0).to(device)
        mean = torch.mean(shadow_cosine_similarity_tensor, dim=0)
        variance = torch.var(shadow_cosine_similarity_tensor, dim=0) + 1e-8

        # Replace np functions with PyTorch's equivalent
        normal_dist = Normal(mean, torch.sqrt(variance))
        fedmia_scores = 1 - normal_dist.cdf(target_cosine_similarity_list)

        return fedmia_scores
