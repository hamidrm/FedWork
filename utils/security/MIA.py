import torch
import numpy as np
import torch.nn.functional as F
from opacus import PrivacyEngine
from torch.distributions.normal import Normal

class MIACommon:
    @staticmethod
    def calculate_auc_metrics(val_scores, train_scores):
        

        # Labels and scores concatenation
        labels = torch.cat([torch.zeros_like(val_scores), torch.ones_like(train_scores)])
        scores = torch.cat([val_scores, train_scores])

        # Compute ROC curve similar to sklearn.metrics.roc_curve
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()
        sorted_indices = np.argsort(-scores_np)  # Descending sort
        sorted_labels = labels_np[sorted_indices]

        tps = np.cumsum(sorted_labels)  # True Positives
        fps = np.cumsum(1 - sorted_labels)  # False Positives

        # Calculate FPR and TPR
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)

        # Log-space AUC calculation
        log_tpr = np.log10(np.clip(tpr, 1e-5, 1))
        log_fpr = np.log10(np.clip(fpr, 1e-5, 1))
        log_tpr = (log_tpr + 5) / 5.0
        log_fpr = (log_fpr + 5) / 5.0
        log_auc = np.trapz(log_tpr, log_fpr)

        # TPRs at specific FPR thresholds
        fpr_thresholds = [0.1, 0.02, 0.01, 0.001, 0.0001]
        fpr_str = ["0.1", "0.02", "0.01", "0.001", "0.0001"]
        tprs_at_thresholds = {}
        for i, threshold in enumerate(fpr_thresholds):
            indices_below_threshold = np.where(fpr < threshold)[0]
            if len(indices_below_threshold) > 0:
                tprs_at_thresholds[fpr_str[i]] = tpr[indices_below_threshold[-1]]
            else:
                tprs_at_thresholds[fpr_str[i]] = 0.0

        return {
            "auc": auc,
            "log_auc": log_auc,
            "tprs": tprs_at_thresholds
        }
    @staticmethod
    def compute_batch_gradients(data_loader, model, loss_fn, device):
        model.train()
        per_sample_gradients = []

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(x)
            loss = loss_fn(outputs, y)

            # Backward pass
            loss.backward()

            # Collect per-sample gradients
            batch_gradients = []
            for param in model.parameters():
                if hasattr(param, "grad_sample") and param.grad_sample is not None:
                    # Flatten gradients per sample
                    batch_gradients.append(param.grad_sample.view(param.grad_sample.size(0), -1).cpu())

            if batch_gradients:  # Concatenate all parameters' gradients for each sample
                per_sample_gradients.append(torch.cat(batch_gradients, dim=1))

            # Clear Opacus' gradient storage to avoid memory issues
            model.zero_grad()

        if per_sample_gradients:
            return torch.cat(per_sample_gradients, dim=0)
        else:
            return None

    @staticmethod
    def evaluate_model_on_experimental_data(data_loader, model, loss_fn, optimizer, device, score_function):
        # Move model to the specified device
        model.to(device)
        model.train()
        score_list = []

        for x, y in data_loader:

            # Move data to the specified device
            x, y = x.to(device), y.to(device)
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
                if param.grad_sample is not None:  # Opacus stores per-sample gradients
                    experimental_grad_batch.append(param.grad_sample.flatten(start_dim=1))
            
            experimental_grad_batch=torch.cat(experimental_grad_batch,1)

            for experimental_grad in experimental_grad_batch:
                score_list.append(score_function(experimental_grad))


        return score_list


    @staticmethod
    def calculate_gradient_difference(model, global_model, device):
        grad_diff = []

        with torch.no_grad():  # Ensure no gradients are computed
            for (name, param), (_, global_param) in zip(model.named_parameters(), global_model.named_parameters()):
                if param.requires_grad:
                    # Ensure both tensors are on the same device
                    param_diff = param.detach().to(device) - global_param.detach().to(device)
                    grad_diff.append(param_diff.view(-1))  # Flatten differences

        return torch.cat(grad_diff) if grad_diff else torch.tensor([], device=device)
class CosMIA:
    def __init__(self, train_data_loader, validation_data_loader, optimizer, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scores = []
        self.target_grad_diff = None

    def get_auc_metrics(self, device):
        # Compute mean validation and training scores directly on the same device
        validation_scores = -torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        train_scores = -torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)

        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        # Calculate AUC metrics for the most recent scores
        return MIACommon.calculate_auc_metrics(-self.scores[-1][0], -self.scores[-1][1])

    def score_function(self, experimental_grads):
        return F.cosine_similarity(experimental_grads, self.target_grad_diff, dim=0)
    
    def execute(self, target_model, global_model, device, lr):

        optimizer_inst = self.optimizer(global_model.parameters(), lr)
        loss_fn_inst = self.loss_fn()

        privacy_engine = PrivacyEngine()
        global_model, optimizer_inst, _ = privacy_engine.make_private(
            module=global_model,
            optimizer=optimizer_inst,
            data_loader=self.train_data_loader,
            noise_multiplier=0,
            max_grad_norm=1e10,
        )


        # Compute target gradient difference
        self.target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model, device)

        # Optimize experimental data to compute gradients
        train_cos = MIACommon.evaluate_model_on_experimental_data(
            self.train_data_loader, global_model, loss_fn_inst, optimizer_inst, device, self.score_function
        )
        val_cos = MIACommon.evaluate_model_on_experimental_data(
            self.validation_data_loader, global_model, loss_fn_inst, optimizer_inst, device, self.score_function
        )

        # Append scores to the list
        self.scores.append((torch.tensor(val_cos).cpu(), torch.tensor(train_cos).cpu()))

class GradDiffMIA:
    def __init__(self, train_data_loader, validation_data_loader, optimizer, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scores = []
        self.target_grad_diff = None

    def get_auc_metrics(self, device):
        # Compute mean validation and training scores directly on the same device
        validation_scores = -torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        train_scores = -torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)

        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        # Calculate AUC metrics for the most recent scores
        return MIACommon.calculate_auc_metrics(-self.scores[-1][0], -self.scores[-1][1])

    def score_function(self, experimental_grads):
        return (torch.norm(self.target_grad_diff.unsqueeze(0), p=2, dim=1) ** 2 - torch.norm(self.target_grad_diff.unsqueeze(0) - experimental_grads, p=2, dim=1) ** 2)
    
    def execute(self, target_model, global_model, device, lr):

        optimizer_inst = self.optimizer(global_model.parameters(), lr)
        loss_fn_inst = self.loss_fn()

        # Compute target gradient difference
        self.target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model, device)

        # Ensure all tensors are on the correct device
        grad_diff_train = MIACommon.evaluate_model_on_experimental_data(
            self.train_data_loader, global_model, loss_fn_inst, optimizer_inst, device, self.score_function
        )
        grad_diff_validation = MIACommon.evaluate_model_on_experimental_data(
            self.validation_data_loader, global_model, loss_fn_inst, optimizer_inst, device, self.score_function
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
        validation_scores = -torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        train_scores = -torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)

        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        # Calculate AUC metrics for the most recent scores
        return MIACommon.calculate_auc_metrics(-self.scores[-1][0], -self.scores[-1][1])
    
    def execute(self, target_model, device):
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
        self.target_grad_diff = None
        self.shadow_grad_diffs = None
        self.device = None

    def get_auc_metrics(self, device):
        # Compute mean validation and training scores directly on the same device
        validation_scores = torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        train_scores = torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)

        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        return MIACommon.calculate_auc_metrics(self.scores[-1][0], self.scores[-1][1])

    def score_function(self, experimental_grads):
        return self.calculate_fedmia_score(self.target_grad_diff, self.shadow_grad_diffs, experimental_grads, self.device)
    
    def execute(self, shadow_models, target_model, global_model, device, lr):

        optimizer_inst = self.optimizer(global_model.parameters(), lr)
        loss_fn_inst = self.loss_fn()

        privacy_engine = PrivacyEngine()
        global_model, optimizer_inst, _ = privacy_engine.make_private(
            module=global_model,
            optimizer=optimizer_inst,
            data_loader=self.train_data_loader,
            noise_multiplier=0,
            max_grad_norm=1e10,
        )

        self.device = device

        # Calculate gradients and cosine similarities on the same device
        self.target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model, device)
        self.shadow_grad_diffs = [MIACommon.calculate_gradient_difference(shadow_model, global_model, device) for shadow_model in shadow_models]

        # Ensure all tensors are on the correct device
        train_scores = MIACommon.evaluate_model_on_experimental_data(
            self.train_data_loader, global_model, loss_fn_inst, optimizer_inst, device, self.score_function
        )
        validation_scores = MIACommon.evaluate_model_on_experimental_data(
            self.validation_data_loader, global_model, loss_fn_inst, optimizer_inst, device, self.score_function
        )

        train_scores_shadows = torch.stack([t[0].squeeze() for t in train_scores])
        validation_scores_shadows = torch.stack([t[0].squeeze() for t in validation_scores])
        train_scores_target = torch.stack([t[1].squeeze() for t in train_scores])
        validation_scores_target = torch.stack([t[1].squeeze() for t in validation_scores])

        mean_train = torch.mean(train_scores_shadows, dim=1)
        variance_train = torch.var(train_scores_shadows, dim=1) + 1e-8

        normal_dist_train = Normal(mean_train, torch.sqrt(variance_train))
        fedmia_score_train = 1 - normal_dist_train.cdf(train_scores_target)

        mean_validation = torch.mean(validation_scores_shadows, dim=1)
        variance_validation = torch.var(validation_scores_shadows, dim=1) + 1e-8

        normal_dist_validation = Normal(mean_validation, torch.sqrt(variance_validation))
        fedmia_score_validation = 1 - normal_dist_validation.cdf(validation_scores_target)

        self.scores.append((fedmia_score_validation, fedmia_score_train))

    def calculate_fedmia_score(self, target_grad_diff, shadow_grad_diffs, experimental_grad_batch_list, device):
        if experimental_grad_batch_list is None:
            return None

        shadow_cosine_similarity_list = []
        for shadow_grad_diff in shadow_grad_diffs:
            shadow_cosine_similarity_list_per_batch = F.cosine_similarity(
                experimental_grad_batch_list, shadow_grad_diff, dim=0)
            shadow_cosine_similarity_list.append(shadow_cosine_similarity_list_per_batch)

        target_cosine_similarity_list = F.cosine_similarity(
            experimental_grad_batch_list, target_grad_diff, dim=0)

        shadow_cosine_similarity_tensor = torch.stack(shadow_cosine_similarity_list, dim=0).to(device)


        return (shadow_cosine_similarity_tensor,target_cosine_similarity_list)
