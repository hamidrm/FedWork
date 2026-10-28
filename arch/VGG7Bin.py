import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Common helpers
# ============================================================

class SignSTE(torch.autograd.Function):
    """Clipped straight-through estimator for binary activations."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x.abs() <= 1).to(dtype=grad_output.dtype)
        return grad_output * mask


def sign_ste(x):
    return SignSTE.apply(x)


class BinaryInputActivation(nn.Module):
    """Binary activation used by the FedBop/BinaryNet branch."""

    def __init__(self, binary=True):
        super().__init__()
        self.binary = bool(binary)

    def forward(self, x):
        if not self.binary:
            return x
        return sign_ste(torch.clamp(x, -1.0, 1.0))


# ============================================================
# FedBop / BinaryNet-compatible layers
# ============================================================

class CenterOnlyBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed gamma=1 and trainable beta only."""

    def __init__(self, num_features, eps=1e-3, keras_momentum=0.99):
        super().__init__()
        self.momentum = 1.0 - keras_momentum
        self.eps = eps

        self.register_buffer("weight", torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(self, x):
        if self.training:
            self.num_batches_tracked += 1
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )


class CenterOnlyBatchNorm1d(nn.Module):
    """BatchNorm1d with fixed gamma=1 and trainable beta only."""

    def __init__(self, num_features, eps=1e-3, keras_momentum=0.99):
        super().__init__()
        self.momentum = 1.0 - keras_momentum
        self.eps = eps

        self.register_buffer("weight", torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(self, x):
        if self.training:
            self.num_batches_tracked += 1
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )


class BinaryConv2d(nn.Module):
    """BinaryNet/Bop-style convolution; FedBop owns the binary weight update."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        input_quantizer=True,
    ):
        super().__init__()
        self.input_quantizer = BinaryInputActivation(binary=input_quantizer)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = 1
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.0, nonlinearity="linear")

    def forward(self, x):
        x = self.input_quantizer(x)
        return F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BinaryLinear(nn.Module):
    """BinaryNet/Bop-style dense layer."""

    def __init__(self, in_features, out_features, input_quantizer=True):
        super().__init__()
        self.input_quantizer = BinaryInputActivation(binary=input_quantizer)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.0, nonlinearity="linear")

    def forward(self, x):
        x = self.input_quantizer(x)
        return F.linear(x, self.weight, bias=None)


# ============================================================
# FedVote layers
# ============================================================

def _fedvote_latent_from_reference(reference_weight, k, eps=1e-3):
    """Initialize h from a normalized reference weight.

    Match the public FedVote repository initialization exactly:
    normalize the reference weight by its standard deviation, then clip the
    normalized weight to [-1 + 1e-3, 1 - 1e-3] before applying atanh/k.

    Note that the paper separately uses p_min=0.001 for the broadcast soft-vote
    probabilities.  These are two distinct numerical-stability safeguards.
    """

    w = reference_weight.detach().clone()
    std = w.std()
    if not torch.isfinite(std) or float(std) <= 0.0:
        raise RuntimeError("Invalid reference-weight standard deviation for FedVote")

    w.div_(std)
    w.clamp_(-1.0 + eps, 1.0 - eps)
    return (0.5 / float(k)) * torch.log((1.0 + w) / (1.0 - w))


class FedVoteConv2d(nn.Module):
    """
    FedWork-compatible FedVote convolution.

    IMPORTANT:
      self.weight is the latent variable h, not the deployed binary weight.
      This naming is intentional: methods/FedVote.py selects '*.weight' tensors.

    Soft/local-training forward:
      w_tilde = tanh(k*h)

    Hard/global-evaluation forward:
      w_bin = sign(h)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        k=1.5,
    ):
        super().__init__()
        self.k = float(k)
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = 1
        self._hard_eval = False

        # Build a temporary full-precision reference weight, then map it to h.
        reference = torch.empty(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        )
        nn.init.kaiming_normal_(reference, a=0.0, mode="fan_in")
        latent = _fedvote_latent_from_reference(reference, self.k)
        self.weight = nn.Parameter(latent)

        # In the official implementation convolution biases are frozen and
        # initialized to zero. Omitting the bias is mathematically equivalent.

    def set_hard_eval(self, enabled=True):
        self._hard_eval = bool(enabled)

    def _effective_weight(self):
        if self._hard_eval:
            return torch.where(
                self.weight >= 0,
                torch.ones_like(self.weight),
                -torch.ones_like(self.weight),
            )
        return torch.tanh(self.k * self.weight)

    def forward(self, x):
        return F.conv2d(
            x,
            self._effective_weight(),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class FedVoteLinear(nn.Module):
    """FedVote dense layer with latent h stored in self.weight."""

    def __init__(self, in_features, out_features, k=1.5):
        super().__init__()
        self.k = float(k)
        self._hard_eval = False

        reference = torch.empty(out_features, in_features)
        nn.init.kaiming_normal_(reference, a=0.0, mode="fan_in")
        latent = _fedvote_latent_from_reference(reference, self.k)
        self.weight = nn.Parameter(latent)

    def set_hard_eval(self, enabled=True):
        self._hard_eval = bool(enabled)

    def _effective_weight(self):
        if self._hard_eval:
            return torch.where(
                self.weight >= 0,
                torch.ones_like(self.weight),
                -torch.ones_like(self.weight),
            )
        return torch.tanh(self.k * self.weight)

    def forward(self, x):
        return F.linear(x, self._effective_weight(), bias=None)



# ============================================================
# BiFL-BiML layers
# ============================================================

class BiFLSignSTE(torch.autograd.Function):
    """Identity STE matching the FL-BNN repository training rule.

    The public code replaces parameter.data by sign(auxiliary_weight) for the
    forward/backward pass, then restores the auxiliary real-valued weight before
    optimizer.step(). Therefore the gradient with respect to the binary weight is
    applied directly to the auxiliary weight (identity STE).
    """

    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def bifl_sign_ste(x):
    return BiFLSignSTE.apply(x)


class BiFLBinaryConv2d(nn.Module):
    """Auxiliary real-valued weight W; forward uses Sign(W)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = 1
        self.reset_parameters()

    def reset_parameters(self):
        # Close to nn.Conv2d's standard initialization; local training will keep
        # the auxiliary weights clipped to [-1, 1].
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        return F.conv2d(
            x,
            bifl_sign_ste(self.weight),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BiFLBinaryLinear(nn.Module):
    """Auxiliary real-valued dense weight W; forward uses Sign(W)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        return F.linear(x, bifl_sign_ste(self.weight), bias=None)

# ============================================================
# Shared VGG-7 macro-architecture
# ============================================================

class VGG7Bin(nn.Module):
    """
    Matched VGG-7 macro-architecture for FedBop, FedVote and BiFL-BiML.

    VGGMode:
      0 -> FedBop/BinaryNet semantics
      1 -> FedVote semantics
      2 -> BiFL-BiML semantics

    Shared topology:
      3 -> 128 -> 128 -> pool
        -> 256 -> 256 -> pool
        -> 512 -> 512 -> pool
        -> 8192 -> 1024 -> 10

    The computational semantics intentionally remain native to each method:
      - FedBop: binary activations and Bop-controlled binary weights.
      - FedVote: real activations, tanh(k*h) during local training, static BN,
                 and sign(h) only for deployed/global hard evaluation.
      - BiFL-BiML: auxiliary real weights in [-1,1], Sign(W) forward,
                   BatchNorm + Hardtanh as in the FL-BNN implementation.
    """

    def __init__(
        self,
        num_classes=<<NumberOfOutputNodes:integer:10>>,
        in_channels=<<NumberOfChannels:integer:3>>,
        input_size=<<InputDataLen:integer:32>>,
        mode=<<VGGMode:integer:0>>,
        fedvote_k=<<FedVoteK:float:1.5>>,
        fedvote_shared_seed=<<FedVoteSharedSeed:integer:42>>,
    ):
        super().__init__()

        if mode not in (0, 1, 2):
            raise ValueError("VGGMode must be 0 (FedBop), 1 (FedVote), or 2 (BiFL-BiML).")

        self.mode = int(mode)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.input_size = int(input_size)
        self.fedvote_k = float(fedvote_k)
        self.fedvote_shared_seed = int(fedvote_shared_seed)

        flatten_features = (
            512 * (self.input_size // 8) * (self.input_size // 8)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.mode == 0:
            self.name = "FedBop-VGG7"

            self.conv1 = BinaryConv2d(
                self.in_channels, 128, 3, padding=1, input_quantizer=False
            )
            self.bn1 = CenterOnlyBatchNorm2d(128)

            self.conv2 = BinaryConv2d(
                128, 128, 3, padding=1, input_quantizer=True
            )
            self.bn2 = CenterOnlyBatchNorm2d(128)

            self.conv3 = BinaryConv2d(
                128, 256, 3, padding=1, input_quantizer=True
            )
            self.bn3 = CenterOnlyBatchNorm2d(256)

            self.conv4 = BinaryConv2d(
                256, 256, 3, padding=1, input_quantizer=True
            )
            self.bn4 = CenterOnlyBatchNorm2d(256)

            self.conv5 = BinaryConv2d(
                256, 512, 3, padding=1, input_quantizer=True
            )
            self.bn5 = CenterOnlyBatchNorm2d(512)

            self.conv6 = BinaryConv2d(
                512, 512, 3, padding=1, input_quantizer=True
            )
            self.bn6 = CenterOnlyBatchNorm2d(512)

            self.fc1 = BinaryLinear(
                flatten_features, 1024, input_quantizer=True
            )
            self.bn7 = CenterOnlyBatchNorm1d(1024)

            self.fc2 = BinaryLinear(
                1024, self.num_classes, input_quantizer=True
            )
            self.bn8 = CenterOnlyBatchNorm1d(self.num_classes)

        elif self.mode == 1:
            self.name = "FedVote-VGG7"

            self.conv1 = FedVoteConv2d(
                self.in_channels, 128, 3, padding=1, k=self.fedvote_k
            )
            self.bn1 = nn.BatchNorm2d(
                128, affine=False, track_running_stats=False
            )

            self.conv2 = FedVoteConv2d(
                128, 128, 3, padding=1, k=self.fedvote_k
            )
            self.bn2 = nn.BatchNorm2d(
                128, affine=False, track_running_stats=False
            )

            self.conv3 = FedVoteConv2d(
                128, 256, 3, padding=1, k=self.fedvote_k
            )
            self.bn3 = nn.BatchNorm2d(
                256, affine=False, track_running_stats=False
            )

            self.conv4 = FedVoteConv2d(
                256, 256, 3, padding=1, k=self.fedvote_k
            )
            self.bn4 = nn.BatchNorm2d(
                256, affine=False, track_running_stats=False
            )

            self.conv5 = FedVoteConv2d(
                256, 512, 3, padding=1, k=self.fedvote_k
            )
            self.bn5 = nn.BatchNorm2d(
                512, affine=False, track_running_stats=False
            )

            self.conv6 = FedVoteConv2d(
                512, 512, 3, padding=1, k=self.fedvote_k
            )
            self.bn6 = nn.BatchNorm2d(
                512, affine=False, track_running_stats=False
            )

            self.fc1 = FedVoteLinear(
                flatten_features, 1024, k=self.fedvote_k
            )
            self.bn7 = nn.BatchNorm1d(
                1024, affine=False, track_running_stats=False
            )

            # Keep the final classifier full precision and frozen, as in
            # FedVote. The name fc3 is deliberate because FedVote.py excludes
            # exactly 'fc3.weight' from binary voting.
            self.fc3 = nn.Linear(1024, self.num_classes)
            self._init_shared_fedvote_classifier()
            self.fc3.weight.requires_grad_(False)
            if self.fc3.bias is not None:
                self.fc3.bias.requires_grad_(False)

        else:
            # BiFL-BiML controlled-comparison branch.  The macro topology is
            # identical to FedBop's 6-conv + 2-FC VGG-7, while the BNN training
            # semantics follow FL-BNN: auxiliary real weights, binary forward,
            # ordinary BN, and Hardtanh activations.
            self.name = "BiFL-BiML-VGG7"

            self.conv1 = BiFLBinaryConv2d(self.in_channels, 128, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(128, affine=True, track_running_stats=False)

            self.conv2 = BiFLBinaryConv2d(128, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128, affine=True, track_running_stats=False)

            self.conv3 = BiFLBinaryConv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256, affine=True, track_running_stats=False)

            self.conv4 = BiFLBinaryConv2d(256, 256, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(256, affine=True, track_running_stats=False)

            self.conv5 = BiFLBinaryConv2d(256, 512, 3, padding=1)
            self.bn5 = nn.BatchNorm2d(512, affine=True, track_running_stats=False)

            self.conv6 = BiFLBinaryConv2d(512, 512, 3, padding=1)
            self.bn6 = nn.BatchNorm2d(512, affine=True, track_running_stats=False)

            self.fc1 = BiFLBinaryLinear(flatten_features, 1024)
            self.bn7 = nn.BatchNorm1d(1024, affine=True, track_running_stats=False)

            self.fc2 = BiFLBinaryLinear(1024, self.num_classes)
            self.bn8 = nn.BatchNorm1d(
                self.num_classes, affine=False, track_running_stats=False
            )

    @torch.no_grad()
    def _init_shared_fedvote_classifier(self):
        """Use an identical fixed FP classifier in every server/client model."""
        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(self.fedvote_shared_seed)
            nn.init.kaiming_normal_(
                self.fc3.weight,
                a=0.0,
                mode="fan_in",
            )
            if self.fc3.bias is not None:
                nn.init.zeros_(self.fc3.bias)
        finally:
            torch.set_rng_state(rng_state)

    def set_hard_eval(self, enabled=True):
        """Switch FedVote quantized layers between tanh(k*h) and sign(h)."""
        if self.mode != 1:
            return
        enabled = bool(enabled)
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, (FedVoteConv2d, FedVoteLinear)):
                module.set_hard_eval(enabled)

    def forward(self, x):
        if self.mode == 1:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool1(x)

            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool2(x)

            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = self.pool3(x)

            x = torch.flatten(x, 1)
            x = F.relu(self.bn7(self.fc1(x)))
            return self.fc3(x)

        if self.mode == 2:
            # Ordering follows the public FL-BNN VGG implementation:
            # conv -> BN -> Hardtanh; pooled blocks pool before BN.
            x = F.hardtanh(self.bn1(self.conv1(x)))

            x = self.conv2(x)
            x = self.pool1(x)
            x = F.hardtanh(self.bn2(x))

            x = F.hardtanh(self.bn3(self.conv3(x)))

            x = self.conv4(x)
            x = self.pool2(x)
            x = F.hardtanh(self.bn4(x))

            x = F.hardtanh(self.bn5(self.conv5(x)))

            x = self.conv6(x)
            x = self.pool3(x)
            x = F.hardtanh(self.bn6(x))

            x = torch.flatten(x, 1)
            x = F.hardtanh(self.bn7(self.fc1(x)))
            x = self.fc2(x)
            return self.bn8(x)

        # FedBop / BinaryNet ordering.
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.pool3(x)
        x = self.bn6(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn7(x)

        x = self.fc2(x)
        x = self.bn8(x)
        return x
