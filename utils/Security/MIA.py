import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.stats import norm

class MIACommon:
    @staticmethod
    def calculate_auc_metrics(val_scores, train_scores):
        labels = torch.cat([torch.zeros_like(val_scores), torch.ones_like(train_scores)])
        scores = torch.cat([val_scores, train_scores])

        fpr, tpr, _ = metrics.roc_curve(labels.cpu().numpy(), scores.cpu().numpy())
        auc = metrics.auc(fpr, tpr)

        log_tpr = np.log10(np.clip(tpr, 1e-5, 1))
        log_fpr = np.log10(np.clip(fpr, 1e-5, 1))

        log_tpr = (log_tpr + 5) / 5.0
        log_fpr = (log_fpr + 5) / 5.0

        log_auc = metrics.auc(log_fpr, log_tpr)

        fpr_thresholds = [0.1, 0.02, 0.01, 0.001, 0.0001]
        tprs_at_thresholds = {}
        for threshold in fpr_thresholds:
            valid_indices = fpr < threshold
            if np.any(valid_indices):
                tprs_at_thresholds[str(threshold)] = tpr[np.max(np.where(valid_indices))]
            else:
                tprs_at_thresholds[str(threshold)] = 0.0

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
            loss = loss_fn(pred, y)
            loss.backward()

            # Collect gradients for each parameter
            experimental_grad_batch = []
            for param in model.parameters():
                if param.grad is not None:
                    experimental_grad_batch.append(param.grad.view(-1))  # Flatten gradients
            experimental_grad_batch_list.append(torch.cat(experimental_grad_batch))

        return torch.stack(experimental_grad_batch_list)
    
    @staticmethod
    def calculate_gradient_difference(model, global_model):
        grad_diff = []
        for (name, param), (_, global_param) in zip(model.named_parameters(), global_model.named_parameters()):
            if param.requires_grad:
                grad_diff.append((param.data - global_param.data).view(-1))  # Flatten differences
        return torch.cat(grad_diff)

class CosMIA:
    def __init__(self, train_data_loader, validation_data_loader, optimizer, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def execute(self, target_model, global_model):

        # Optimize experimental data for training and validation sets
        train_gradients = MIACommon.optimize_experimental_data(self.train_data_loader, global_model)
        validation_gradients = MIACommon.optimize_experimental_data(self.validation_data_loader, global_model)

        # Calculate gradient differences for target model
        target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model)

        # Compute cosine similarities for target model

        target_cosine_similarity_train_list = F.cosine_similarity(
            train_gradients, target_grad_diff.unsqueeze(0), dim=1
        )

        target_cosine_similarity_validation__list = F.cosine_similarity(
            validation_gradients, target_grad_diff.unsqueeze(0), dim=1
        )

        # Return AUC metrics
        return MIAMetrics.calculate_auc_metrics(target_cosine_similarity_validation__list, target_cosine_similarity_train_list)



class GradDiffMIA:
    def __init__(self, train_data_loader, validation_data_loader, optimizer, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def execute(self, target_model, global_model):

        # Optimize experimental data for training and validation sets
        train_gradients = MIACommon.optimize_experimental_data(self.train_data_loader, global_model)
        validation_gradients = MIACommon.optimize_experimental_data(self.validation_data_loader, global_model)

        # Calculate gradient differences for target model
        target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model)

        # Compute cosine similarities for target model
        grad_diff_train = torch.norm(target_grad_diff.unsqueeze(0), p=2, dim=0)**2 - torch.norm(target_grad_diff.unsqueeze(0)-train_gradients, p=2, dim=0)**2
        grad_diff_validation = torch.norm(target_grad_diff.unsqueeze(0), p=2, dim=0)**2 - torch.norm(target_grad_diff.unsqueeze(0)-validation_gradients, p=2, dim=0)**2

        # Return AUC metrics
        return MIAMetrics.calculate_auc_metrics(grad_diff_validation, grad_diff_train)

class LossMIA:
    def __init__(self, train_data_loader, validation_data_loader, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.loss_fn = loss_fn

    def execute(self, target_model, global_model):

        # Optimize experimental data for training and validation sets
        train_loss= self.get_model_res(self.train_data_loader, target_model, self.loss_fn)
        validation_loss = self.get_model_res(self.validation_data_loader, target_model, self.loss_fn)

        # Return AUC metrics
        return MIAMetrics.calculate_auc_metrics(validation_loss, train_loss)

    def get_model_res(self, data_loader, model, loss_fn):
        model.eval()
        experimental_logits_batch_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                # Forward pass
                pred = model(inputs)
    
                experimental_logits_batch_list.append(loss_fn(pred, targets))

        return experimental_logits_batch_list

class FedMIA:
    def __init__(self, train_data_loader, validation_data_loader, optimizer, loss_fn) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def execute(self, shadow_models, target_model, global_model):
        # Optimize experimental data for training and validation sets
        train_gradients = MIACommon.optimize_experimental_data(self.train_data_loader, global_model)
        validation_gradients = MIACommon.optimize_experimental_data(self.validation_data_loader, global_model)

        # Calculate scores for training and validation
        train_scores = MIACommon.calculate_fedmia_score(shadow_models, target_model, global_model, train_gradients)
        validation_scores = MIACommon.calculate_fedmia_score(shadow_models, target_model, global_model, validation_gradients)

        # Return AUC metrics
        return MIAMetrics.calculate_auc_metrics(validation_scores, train_scores)

    def calculate_fedmia_score(self, shadow_models, target_model, global_model, experimental_grad_batch_list):
        if experimental_grad_batch_list is None:
            return None

        shadow_cosine_similarity_list = []
        target_cosine_similarity_list = []

        # Calculate gradient differences for shadow and target models
        target_grad_diff = self.calculate_gradient_difference(target_model, global_model)
        shadow_grad_diffs = [self.calculate_gradient_difference(shadow_model, global_model) for shadow_model in shadow_models]

        # Compute cosine similarities for shadow models
        for shadow_grad_diff in shadow_grad_diffs:
            shadow_cosine_similarity_list_per_batch = F.cosine_similarity(
                experimental_grad_batch_list, shadow_grad_diff.unsqueeze(0), dim=1
            )
            shadow_cosine_similarity_list.append(shadow_cosine_similarity_list_per_batch)

        # Compute cosine similarities for target model
        target_cosine_similarity_list = F.cosine_similarity(
            experimental_grad_batch_list, target_grad_diff.unsqueeze(0), dim=1
        )

        # Stack shadow cosine similarities and calculate statistics
        shadow_cosine_similarity_tensor = torch.stack(shadow_cosine_similarity_list)
        mean = torch.mean(shadow_cosine_similarity_tensor, dim=0)
        variance = torch.var(shadow_cosine_similarity_tensor, dim=0) + 1e-8

        # Compute FedMIA scores
        fedmia_scores = 1 - norm.cdf(
            target_cosine_similarity_list.cpu().numpy(),
            mean.cpu().numpy(),
            torch.sqrt(variance).cpu().numpy()
        )

        return torch.tensor(fedmia_scores)