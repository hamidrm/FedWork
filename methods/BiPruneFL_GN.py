"""ResNet-BinAct-aware BiPruneFL baseline adapter for FedWork.

Only change relative to the provided FedWork methods/BiPruneFL.py:
update BOTH affine GroupNorm gamma and beta on the server. The original
BiPruneFL architecture uses torch.nn.GroupNorm(2, C, affine=True).

Install beside methods/BiPruneFL.py and select type="BiPruneFL_GN".
"""
import torch
from methods.BiPruneFL import BiPruneFL


class BiPruneFL_GN(BiPruneFL):
    @staticmethod
    def _is_aux_trainable(name, tensor):
        if not (torch.is_tensor(tensor) and tensor.ndim == 1):
            return False
        name = name.lower()
        return (('norm' in name or 'bn' in name) and
                (name.endswith('.bias') or name.endswith('.weight')))

    def get_name(self):
        return 'BiPruneFL (affine GroupNorm-aware)'
