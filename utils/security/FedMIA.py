from sklearn import metrics
import torch
import numpy as np
import torch.nn.functional as F
from opacus import PrivacyEngine
from torch.distributions.normal import Normal
import copy

class MIACommon:

    @staticmethod
    def filter_mean_var_tensor(tensor_list):
        filtered_tensor_mean=torch.zeros(len(tensor_list), device=tensor_list[0].device)
        filtered_tensor_var=torch.zeros(len(tensor_list), device=tensor_list[0].device)

        for i in range(len(tensor_list)):
            mean = tensor_list[i].mean(dim=0)
            std = tensor_list[i].std(dim=0)
            threshold = mean + 3 * std

            filtered_tensor_mean[i] = torch.mean(tensor_list[i][tensor_list[i][:] < threshold], dim=0)
            filtered_tensor_var[i] = torch.var(tensor_list[i][tensor_list[i][:] < threshold], dim=0) + 1e-8

        return filtered_tensor_mean, filtered_tensor_var

    @staticmethod
    def calculate_auc_metrics(val_scores, train_scores, low_fpr_cutoff=0.01):
        """
        val_scores: tensor of non-member scores (e.g., -loss on validation)
        train_scores: tensor of member scores (e.g., -loss on train)
        """
        # labels: 0 = non-member (val), 1 = member (train)
        labels = torch.cat([torch.zeros_like(val_scores), torch.ones_like(train_scores)])
        scores = torch.cat([val_scores, train_scores])

        # to numpy
        y = labels.detach().cpu().numpy().astype(np.int32)
        s = scores.detach().cpu().numpy().astype(np.float64)

        # Sort by score desc; stable so ties keep order (ok for ROC)
        order = np.argsort(-s, kind="mergesort")
        y_sorted = y[order]
        s_sorted = s[order]

        # Cum sums give ROC steps at each distinct threshold
        tps = np.cumsum(y_sorted)                 # positives predicted positive
        fps = np.cumsum(1 - y_sorted)             # negatives predicted positive
        P = tps[-1] if tps.size else 0            # total positives
        N = fps[-1] if fps.size else 0            # total negatives

        if P == 0 or N == 0:
            # Degenerate case: cannot build ROC
            return {"auc": float("nan"), "log_auc": float("nan"),
                    "tprs": {k: float("nan") for k in ["0.1","0.02","0.01","0.001","0.0001"]},
                    "pauc_<=1%": float("nan")}

        # Collapse ties (distinct thresholds) to get proper ROC
        # Indices where score changes
        score_changes = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
        tps_u = tps[score_changes]
        fps_u = fps[score_changes]

        # Build ROC with endpoints
        fpr = np.r_[0.0, fps_u / N, 1.0]
        tpr = np.r_[0.0, tps_u / P, 1.0]

        # Standard AUC
        auc = np.trapz(tpr, fpr)

        # Log-space AUC (non-standard, but sometimes reported in MIAs)
        eps = 1e-12
        fpr_clip = np.clip(fpr, eps, 1.0)
        tpr_clip = np.clip(tpr, eps, 1.0)
        log_fpr = (np.log10(fpr_clip) + 12) / 12.0  # rescale ~[0,1]
        log_tpr = (np.log10(tpr_clip) + 12) / 12.0
        log_auc = np.trapz(log_tpr, log_fpr)

        # TPR at fixed FPRs via interpolation (monotone vectors)
        fpr_targets = np.array([0.1, 0.02, 0.01, 0.001, 0.0001], dtype=float)
        tprs_at = np.interp(fpr_targets, fpr, tpr, left=0.0, right=1.0)
        tprs = {k: v for k, v in zip(["0.1","0.02","0.01","0.001","0.0001"], tprs_at)}

        # Partial AUC up to a low-FPR cutoff (default 1%)
        # Interpolate curve up to cutoff and integrate
        cutoff = float(low_fpr_cutoff)
        if cutoff < fpr[-1]:
            tpr_at_cut = np.interp(cutoff, fpr, tpr)
            fpr_p = np.r_[fpr[fpr <= cutoff], cutoff]
            tpr_p = np.r_[tpr[fpr <= cutoff], tpr_at_cut]
            pauc = np.trapz(tpr_p, fpr_p) / cutoff  # normalized pAUC in [0,1]
        else:
            pauc = 1.0

        return {
            "auc": float(auc),
            "log_auc": float(log_auc),
            "tprs": tprs,
            "pauc_<=1%": float(pauc),
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
    def evaluate_model_on_experimental_data(data_loader, shadow_models, target_model,  model, loss_fn, optimizer, device, score_function):
        
        # Move model to the specified device
        model.to(device)
        model.train()
        score_list = []

        for x, y in data_loader:
            shadow_models_pair = [None] * len(shadow_models)
            target_model_pair = []

            # Move data to the specified device
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # Forward pass
            pred = model(x)

            # Compute loss
            loss = loss_fn(pred, y)

            # Backward pass
            loss.backward()
            #if len(y) < data_loader.batch_sampler.batch_size:
            #    continue
            # Collect gradients for each parameter
            experimental_grad_batch = []
            for name, param in model.named_parameters():
                name = str.replace(name, "_module.", "")
                if param.grad_sample is not None:  # Opacus stores per-sample gradients
                    experimental_grad_batch.append(param.grad_sample.flatten(start_dim=1))
                    if name in target_model.keys():
                        target_model_pair.append(param.grad_sample.flatten(start_dim=1))
                    for i, shadow_model in enumerate(shadow_models):
                        if name in shadow_model.keys():

                            if shadow_models_pair[i] is None:
                                shadow_models_pair[i] = []
                            shadow_models_pair[i].append(param.grad_sample.flatten(start_dim=1))

            if target_model_pair == []:
                continue
            target_model_pair=torch.cat(target_model_pair,1)
            for i in range(len(shadow_models)):
                if shadow_models_pair[i] is not None:
                    shadow_models_pair[i]=torch.cat(shadow_models_pair[i],1)
            score_pair = score_function(target_model_pair, shadow_models_pair)

            if (score_pair[0] is not None) and (score_pair[1] is not None):
                score_list.append(score_pair)

        return score_list


    @staticmethod
    def calculate_gradient_difference(model, global_model, device):
        grad_diff = []
        # Iterate over the model's state_dict keys
        for name, param in model.items():
            name = "_module."+name
            if name in global_model:  # Ensure the parameter exists in the global model
                global_param = global_model[name]

                # Ensure both tensors are on the same device
                param_diff = global_param.detach().to(device) - param.detach().to(device)
                grad_diff.append(param_diff.view(-1))  # Flatten differences

        return grad_diff


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
        if self.scores == []:
            return {
                "auc": 0,
                "log_auc": 0,
                "tprs": {'0.1': 0.0, '0.02': 0.0, '0.01': 0.0, '0.001': 0.0, '0.0001': 0.0}
            }
        # Compute mean validation and training scores directly on the same device
        validation_scores = torch.cat([i[0].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)
        train_scores = torch.cat([i[1].unsqueeze(0) for i in self.scores], dim=0).mean(dim=0).to(device)

        return MIACommon.calculate_auc_metrics(validation_scores, train_scores)

    def get_last_auc_metrics(self):
        return MIACommon.calculate_auc_metrics(self.scores[-1][0], self.scores[-1][1])

    def score_function(self, target_model_pair, shadow_models_pair):

        shadow_cosine_similarity_list = []
        for i, shadow_grad_diff in enumerate(self.shadow_grad_diffs):
            if shadow_grad_diff != []:
                shadow_grad_diff = torch.cat(shadow_grad_diff)
                for shadow_grad_batch in shadow_models_pair[i]:
                    
                    shadow_cosine_similarity_list_per_batch = F.cosine_similarity(shadow_grad_diff, shadow_grad_batch, dim=0)
                    shadow_cosine_similarity_list.append(shadow_cosine_similarity_list_per_batch)

        for target_grad_batch in target_model_pair:
            target_grad_diff_tensor = torch.cat(self.target_grad_diff)
            target_cosine_similarity_list = F.cosine_similarity(
                target_grad_batch, target_grad_diff_tensor, dim=0)


        if not shadow_cosine_similarity_list:
            return (None, None)
        
        shadow_cosine_similarity_tensor = torch.stack(shadow_cosine_similarity_list, dim=0)

        return (shadow_cosine_similarity_tensor,target_cosine_similarity_list)

    
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

        # Calculate gradients and cosine similarities
        self.target_grad_diff = MIACommon.calculate_gradient_difference(target_model, global_model.state_dict(), device)
        self.shadow_grad_diffs = [MIACommon.calculate_gradient_difference(shadow_model, global_model.state_dict(), device) for shadow_model in shadow_models]

        train_scores = MIACommon.evaluate_model_on_experimental_data(
            self.train_data_loader, shadow_models, target_model, global_model, loss_fn_inst, optimizer_inst, device, self.score_function
        )
        validation_scores = MIACommon.evaluate_model_on_experimental_data(
            self.validation_data_loader, shadow_models, target_model, global_model, loss_fn_inst, optimizer_inst, device, self.score_function
        )


        train_scores_shadows_list = [t[0].squeeze() for t in train_scores if t != []]
        validation_scores_shadows_list =[t[0].squeeze() for t in validation_scores if t != []]
        train_scores_target_list = [t[1].squeeze() for t in train_scores if t != []]
        validation_scores_target_list = [t[1].squeeze() for t in validation_scores if t != []]

        if train_scores_target_list != []:
            train_scores_target = torch.stack(train_scores_target_list)
        else:
            return
        if validation_scores_target_list != []:
            validation_scores_target = torch.stack(validation_scores_target_list)
        else:
            return
        

        mean_train,variance_train = MIACommon.filter_mean_var_tensor(train_scores_shadows_list)
        mean_validation,variance_validation = MIACommon.filter_mean_var_tensor(validation_scores_shadows_list)


        normal_dist_train = Normal(mean_train, torch.sqrt(variance_train))
        fedmia_score_train = normal_dist_train.cdf(train_scores_target)

        normal_dist_validation = Normal(mean_validation, torch.sqrt(variance_validation))
        fedmia_score_validation = normal_dist_validation.cdf(validation_scores_target)

        self.scores.append((fedmia_score_validation, fedmia_score_train))

    def calculate_fedmia_score(self, target_grad_diff, shadow_grad_diffs, name, experimental_grad_batch_list, device):
        if experimental_grad_batch_list is None:
            return None

        shadow_cosine_similarity_list = []
        for shadow_grad_diff in shadow_grad_diffs:
            shadow_grad_list = []
            for shadow_layer in shadow_grad_diff:
                shadow_name = shadow_layer[0]
                grad_diff = shadow_layer[1]

                if name == shadow_name:
                    shadow_grad_list.append(grad_diff)
                    break
            
            if shadow_grad_list != []:
                shadow_grad_list = torch.cat(shadow_grad_list, 0)
                shadow_cosine_similarity_list_per_batch = F.cosine_similarity(
                    shadow_grad_list, experimental_grad_batch_list, dim=0)
                shadow_cosine_similarity_list.append(shadow_cosine_similarity_list_per_batch)

        for target_layer in target_grad_diff:
            target_name = target_layer[0]
            grad_diff = target_layer[1]

            if name == target_name:
                target_cosine_similarity_list = F.cosine_similarity(experimental_grad_batch_list, grad_diff, dim=0)
                break

        shadow_cosine_similarity_tensor = torch.stack(shadow_cosine_similarity_list, dim=0).to(device)


        return (shadow_cosine_similarity_tensor,target_cosine_similarity_list)
