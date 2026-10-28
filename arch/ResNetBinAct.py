"""Method-aware CIFAR 32x32 ResNet-18 / ResNet-9 for FedWork.

Macro-topology/activation reference:
  ION-dgu/BipruneFL-Communication-Computation-Efficient-Federated-Learning
  src/models/resnet_BinAct.py (main blob 90cf88ab10c3088253a879a5f70f95bcc0874cd8)

ResNetMode:
  0: FedBop   -- explicit sign-initialized binary weights; Bop owns flips.
  1: FedVote  -- latent h; tanh(FedVoteK*h) in training, sign(h) for hard eval.
                   Non-affine GroupNorm: FedVote's existing wire codec does not
                   exchange normalization parameters.
  2: BiFLBiML -- auxiliary real W, sign(W) in forward with identity STE.
  3: BiPruneFL_GN -- effective masked/gained weights provided by the method;
                    never sign/quantize the effective stored tensor here.

The SAME ResNet-BinAct topology and BiReal activation STE are shared by all
four modes. Weight parameterization differs as required by each FL method.
FedVote's non-affine normalization is the one intentional norm distinction;
otherwise its parameters would silently become non-synchronized across clients.

BiPruneFL_GN is a FedWork adapter, not a byte-exact copy of authors' SubnetConv.
"""
import torch
from torch import nn
from torch.nn import functional as F


class _BiRealSign(torch.autograd.Function):
    """Official BiRealAct: forward sign(x), backward 2*(1-|clip(x,-1,1)|)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return 2.0 * (1.0 - x.clamp(-1.0, 1.0).abs()) * grad_output


class _BinaryAct(nn.Module):
    def forward(self, x):
        return _BiRealSign.apply(x)


class _BiFLSign(torch.autograd.Function):
    """BiFL's auxiliary-weight STE, not the BiReal activation STE."""

    @staticmethod
    def forward(ctx, weight):
        return torch.where(weight >= 0, torch.ones_like(weight), -torch.ones_like(weight))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _fedvote_h_init(weight, k):
    """As in FedWork/VGG7Bin FedVote: normalize reference and map to latent h."""
    w = weight.detach().clone()
    std = w.std()
    if not torch.isfinite(std) or float(std) <= 0:
        raise RuntimeError("Invalid weight standard deviation for FedVote initialization")
    w = (w / std).clamp(-1.0 + 1e-3, 1.0 - 1e-3)
    return torch.atanh(w) / float(k)


class _MethodConv(nn.Conv2d):
    """Expose a 4D `weight` parameter recognized by all existing FedWork methods."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 mode=0, fedvote_k=1.5):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride,
                         padding=padding, bias=False)
        self.mode = int(mode)
        self.fedvote_k = float(fedvote_k)
        self._hard_eval = False
        with torch.no_grad():
            if self.mode in (0, 3):
                # Bop starts at {-1,+1}; BiPruneFL overrides this effective
                # parameter with its own (0, +/-alpha) masked weights.
                self.weight.copy_(torch.where(self.weight >= 0,
                                              torch.ones_like(self.weight),
                                              -torch.ones_like(self.weight)))
            elif self.mode == 1:
                reference = torch.empty_like(self.weight)
                nn.init.kaiming_normal_(reference, mode="fan_in", nonlinearity="relu")
                self.weight.copy_(_fedvote_h_init(reference, self.fedvote_k))
            elif self.mode == 2:
                nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            else:
                raise ValueError("ResNetMode must be 0, 1, 2 or 3")

    def set_hard_eval(self, enabled=True):
        self._hard_eval = bool(enabled)

    def effective_weight(self):
        if self.mode == 1:
            if self._hard_eval:
                return torch.where(self.weight >= 0,
                                   torch.ones_like(self.weight),
                                   -torch.ones_like(self.weight))
            return torch.tanh(self.fedvote_k * self.weight)
        if self.mode == 2:
            return _BiFLSign.apply(self.weight)
        return self.weight

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), None, self.stride,
                        self.padding, self.dilation, self.groups)


def _norm(channels, mode):
    # Official resnet_BinAct uses AffineGroupNorm(2,C).
    # The current FedVote method communicates only 4D/2D weight tensors,
    # so mode=1 uses deterministic non-affine GroupNorm (no hidden client
    # parameters silently discarded during aggregation).
    return nn.GroupNorm(2, channels, affine=(mode != 1))


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, mode, fedvote_k):
        super().__init__()
        self.conv1 = _MethodConv(in_planes, planes, 3, stride, 1, mode, fedvote_k)
        self.norm1 = _norm(planes, mode)
        self.conv2 = _MethodConv(planes, planes, 3, 1, 1, mode, fedvote_k)
        self.norm2 = _norm(planes, mode)
        self.norm3 = _norm(planes, mode)
        self.act = _BinaryAct()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                _MethodConv(in_planes, planes, 1, stride, 0, mode, fedvote_k),
                _norm(planes, mode),
            )

    def forward(self, x):
        # Match official BasicBlock: act(x) -> conv1 -> GN -> act ->
        # conv2 -> GN -> residual add -> GN, no activation after last GN.
        identity = x
        out = self.act(x)
        out = self.act(self.norm1(self.conv1(out)))
        out = self.norm2(self.conv2(out))
        return self.norm3(out + self.shortcut(identity))


class _ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes, mode, fedvote_k):
        super().__init__()
        self.in_planes = 64
        self.conv1 = _MethodConv(in_channels, 64, 3, 1, 1, mode, fedvote_k)
        self.norm1 = _norm(64, mode)
        self.act = _BinaryAct()
        self.layer1 = self._make_layer(64, 2, 1, mode, fedvote_k)
        self.layer2 = self._make_layer(128, 2, 2, mode, fedvote_k)
        self.layer3 = self._make_layer(256, 2, 2, mode, fedvote_k)
        self.layer4 = self._make_layer(512, 2, 2, mode, fedvote_k)
        self.linear = _MethodConv(512, num_classes, 1, 1, 0, mode, fedvote_k)

    def _make_layer(self, planes, num_blocks, stride, mode, fedvote_k):
        blocks = []
        for s in [stride] + [1] * (num_blocks - 1):
            blocks.append(_BasicBlock(self.in_planes, planes, s, mode, fedvote_k))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.act(x)
        x = F.avg_pool2d(x, 4)
        return self.linear(x).flatten(1)


class _Res9ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, mode, fedvote_k, pool=False):
        super().__init__()
        self.conv1 = _MethodConv(in_planes, planes, 3, 1, 1, mode, fedvote_k)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.norm = _norm(planes, mode)

    def forward(self, x):
        return self.norm(self.pool(self.conv1(x)))


class _Res9BasicBlock(nn.Module):
    def __init__(self, planes, mode, fedvote_k):
        super().__init__()
        self.act = _BinaryAct()
        self.conv1 = _MethodConv(planes, planes, 3, 1, 1, mode, fedvote_k)
        self.norm1 = _norm(planes, mode)
        self.conv2 = _MethodConv(planes, planes, 3, 1, 1, mode, fedvote_k)
        self.norm2 = _norm(planes, mode)
        self.norm3 = _norm(planes, mode)

    def forward(self, x):
        out = self.act(x)
        out = self.act(self.norm1(self.conv1(out)))
        out = self.norm2(self.conv2(out))
        return self.act(self.norm3(out + x))


class _ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, mode, fedvote_k):
        super().__init__()
        self.conv1 = _MethodConv(in_channels, 64, 3, 1, 1, mode, fedvote_k)
        self.norm1 = _norm(64, mode)
        self.act = _BinaryAct()
        self.layer1 = nn.Sequential(
            _Res9ConvBlock(64, 128, mode, fedvote_k, pool=True),
            _Res9BasicBlock(128, mode, fedvote_k))
        self.layer2 = nn.Sequential(_Res9ConvBlock(128, 256, mode, fedvote_k, pool=True))
        self.layer3 = nn.Sequential(
            _Res9ConvBlock(256, 512, mode, fedvote_k, pool=True),
            _Res9BasicBlock(512, mode, fedvote_k))
        self.fc = _MethodConv(512, num_classes, 1, 1, 0, mode, fedvote_k)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 4)
        return self.fc(x).flatten(1)


class ResNetBinAct(nn.Module):
    """FedWork entrypoint; use ResNetMode per method in the XML."""

    def __init__(self,
                 num_classes=<<NumberOfOutputNodes:integer:100>>,
                 in_channels=<<NumberOfChannels:integer:3>>,
                 input_size=<<InputDataLen:integer:32>>,
                 depth=<<ResNetDepth:integer:18>>,
                 mode=<<ResNetMode:integer:0>>,
                 fedvote_k=<<FedVoteK:float:1.5>>):
        super().__init__()
        if input_size != 32:
            raise ValueError("Official CIFAR resnet_BinAct expects 32x32 inputs")
        if mode not in (0, 1, 2, 3):
            raise ValueError("ResNetMode must be 0,1,2,3")
        if fedvote_k <= 0.0:
            raise ValueError("FedVoteK must be positive")
        self.mode = int(mode)
        self.fedvote_k = float(fedvote_k)
        if depth == 18:
            self.model = _ResNet18(in_channels, num_classes, mode, fedvote_k)
        elif depth == 9:
            self.model = _ResNet9(in_channels, num_classes, mode, fedvote_k)
        else:
            raise ValueError("ResNetDepth must be 18 or 9")

    def set_hard_eval(self, enabled=True):
        """FedVote.py's hook switches between tanh(k*h) and sign(h)."""
        if self.mode == 1:
            for module in self.modules():
                if (module is not self and
                        getattr(module, "mode", None) == 1 and
                        callable(getattr(module, "set_hard_eval", None))):
                    module.set_hard_eval(enabled)

    def forward(self, x):
        return self.model(x)
