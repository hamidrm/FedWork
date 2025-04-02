import torch

class GradientSparsifier:
    def __init__(self, sparsity_ratio=0.1):
        assert 0 < sparsity_ratio <= 1, "sparsity_ratio must be in (0,1]"
        self.sparsity_ratio = sparsity_ratio

    def sparsify(self, local_model, global_model):

        sparsified_gradients = {}

        for key in local_model:
            grad_diff = local_model[key] - global_model[key]

            # Flatten gradient
            flat_grad = grad_diff.view(-1)
            k = max(1, int(self.sparsity_ratio * flat_grad.numel()))

            # Find top-k magnitude elements
            _, indices = torch.topk(torch.abs(flat_grad), k)

            # Create sparse gradient tensor initialized to zero
            sparse_grad = torch.zeros_like(flat_grad)

            # Set top-k elements
            sparse_grad[indices] = flat_grad[indices]

            # Reshape back to original shape
            sparsified_gradients[key] = sparse_grad.view(grad_diff.shape)

        return sparsified_gradients


