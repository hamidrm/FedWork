import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared BinaryNet policies
# =============================================================================
WEIGHT_DIRECT = 0       # FedBop: stored weight is used directly
WEIGHT_SIGN_STE = 1     # BiFL-BiML: forward uses sign(W), W stays real-valued
WEIGHT_TANH = 2         # FedVote: stored tensor is latent h; forward tanh(k*h)

ACT_BINARY_PRE = 0      # Existing BinaryNet behavior
ACT_RELU_POST_BN = 1    # FedVote reference behavior
ACT_IDENTITY = 2

BN_CENTER_ONLY = 0      # Existing FedWork BN
BN_STATIC = 1           # FedVote: affine=False, track_running_stats=False


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).to(grad_output.dtype)


def sign_ste(x):
    return SignSTE.apply(x)


def hard_sign(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def transform_weight(weight, mode, scale=1.5):
    mode = int(mode)
    if mode == WEIGHT_DIRECT:
        return weight
    if mode == WEIGHT_SIGN_STE:
        return sign_ste(weight)
    if mode == WEIGHT_TANH:
        return torch.tanh(float(scale) * weight)
    raise ValueError(f"Unsupported WeightMode={mode}")


class BinaryInputActivation(nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)

    def forward(self, x):
        if not self.enabled:
            return x
        return sign_ste(torch.clamp(x, -1.0, 1.0))


class CenterOnlyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-3, keras_momentum=0.99):
        super().__init__()
        self.momentum = 1.0 - keras_momentum
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            self.num_batches_tracked += 1
        return F.batch_norm(
            x, self.running_mean, self.running_var,
            weight=self.weight, bias=self.bias,
            training=self.training, momentum=self.momentum, eps=self.eps
        )


class CenterOnlyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-3, keras_momentum=0.99):
        super().__init__()
        self.momentum = 1.0 - keras_momentum
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            self.num_batches_tracked += 1
        return F.batch_norm(
            x, self.running_mean, self.running_var,
            weight=self.weight, bias=self.bias,
            training=self.training, momentum=self.momentum, eps=self.eps
        )


def make_bn2d(num_features, mode):
    if int(mode) == BN_STATIC:
        return nn.BatchNorm2d(
            num_features,
            eps=1e-5,
            affine=False,
            track_running_stats=False,
        )
    return CenterOnlyBatchNorm2d(num_features)


def make_bn1d(num_features, mode):
    if int(mode) == BN_STATIC:
        return nn.BatchNorm1d(
            num_features,
            eps=1e-5,
            affine=False,
            track_running_stats=False,
        )
    return CenterOnlyBatchNorm1d(num_features)


class _WeightPolicyMixin:
    def _init_weight_policy(self, weight_mode, weight_scale):
        self.weight_mode = int(weight_mode)
        self.weight_scale = float(weight_scale)
        self.hard_eval = False

    def set_weight_mode(self, mode, scale=None):
        self.weight_mode = int(mode)
        if scale is not None:
            self.weight_scale = float(scale)

    def set_hard_eval(self, enabled):
        self.hard_eval = bool(enabled)

    def forward_weight(self):
        # FedVote training uses soft normalized weights tanh(k*h).
        # For profiler/global evaluation we optionally evaluate the deployed
        # hard BNN, i.e. sign(h) in {-1,+1}.  Training state h is untouched.
        if self.hard_eval and self.weight_mode == WEIGHT_TANH:
            return hard_sign(self.weight)

        return transform_weight(
            self.weight,
            self.weight_mode,
            self.weight_scale
        )


class BinaryConv2d(_WeightPolicyMixin, nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        binary_input=True,
        weight_mode=WEIGHT_DIRECT,
        weight_scale=1.5,
    ):
        super().__init__()
        self.input_quantizer = BinaryInputActivation(binary_input)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = 1
        self._init_weight_policy(weight_mode, weight_scale)

        # Temporary initialization.  In FedVote mode BinaryNet later replaces
        # all latent tensors jointly using the reference-model initialization.
        nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in")

    def forward(self, x):
        x = self.input_quantizer(x)
        return F.conv2d(
            x,
            self.forward_weight(),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BinaryLinear(_WeightPolicyMixin, nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        binary_input=True,
        weight_mode=WEIGHT_DIRECT,
        weight_scale=1.5,
    ):
        super().__init__()
        self.input_quantizer = BinaryInputActivation(binary_input)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self._init_weight_policy(weight_mode, weight_scale)
        nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in")

    def forward(self, x):
        x = self.input_quantizer(x)
        return F.linear(x, self.forward_weight(), bias=None)


class BinaryNet2(nn.Module):
    """
    Shared topology.

    FedVote configuration:
        WeightMode=2
        WeightScale=1.5
        ActivationMode=1
        BNMode=1
        FreezeFinalLayer=1

    IMPORTANT:
    In FedVote mode, quantized latent parameters are NOT initialized to zero.

    The official FedVote implementation first creates a Kaiming-initialized
    full-precision reference model, then for each quantized layer:

        ref_w = ref_w / std(ref_w)
        ref_w = clamp(ref_w, -0.999, 0.999)
        h     = atanh(ref_w) / k

    This file reproduces that initialization on the shared BinaryNet topology.
    """

    def __init__(
        self,
        num_classes=<<NumberOfOutputNodes:integer:10>>,
        in_channels=<<NumberOfChannels:integer:3>>,
        filters=<<NumberOfFilters:integer:128>>,
        dense_units=<<NumberOfDenseUnits:integer:1024>>,
        weight_mode=<<WeightMode:integer:0>>,
        weight_scale=<<WeightScale:float:1.5>>,
        activation_mode=<<ActivationMode:integer:0>>,
        bn_mode=<<BNMode:integer:0>>,
        freeze_final_layer=<<FreezeFinalLayer:integer:0>>,
        final_layer_seed=<<FinalLayerSeed:integer:1234>>,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.filters = int(filters)
        self.dense_units = int(dense_units)

        self.weight_mode = int(weight_mode)
        self.weight_scale = float(weight_scale)
        self.activation_mode = int(activation_mode)
        self.bn_mode = int(bn_mode)
        self.freeze_final_layer_flag = bool(int(freeze_final_layer))
        self.final_layer_seed = int(final_layer_seed)
        self.hard_eval = False

        if self.activation_mode not in (
            ACT_BINARY_PRE,
            ACT_RELU_POST_BN,
            ACT_IDENTITY,
        ):
            raise ValueError("ActivationMode must be 0, 1, or 2")

        if self.bn_mode not in (BN_CENTER_ONLY, BN_STATIC):
            raise ValueError("BNMode must be 0 or 1")

        binary_pre = self.activation_mode == ACT_BINARY_PRE
        common = dict(
            weight_mode=self.weight_mode,
            weight_scale=self.weight_scale,
        )

        # Same FedWork topology as before.
        self.conv1 = BinaryConv2d(
            in_channels,
            filters,
            kernel_size=3,
            padding=0,
            binary_input=False,
            **common,
        )
        self.bn1 = make_bn2d(filters, self.bn_mode)

        self.conv2 = BinaryConv2d(
            filters,
            filters,
            kernel_size=3,
            padding=1,
            binary_input=binary_pre,
            **common,
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = make_bn2d(filters, self.bn_mode)

        self.conv3 = BinaryConv2d(
            filters,
            2 * filters,
            kernel_size=3,
            padding=1,
            binary_input=binary_pre,
            **common,
        )
        self.bn3 = make_bn2d(2 * filters, self.bn_mode)

        self.conv4 = BinaryConv2d(
            2 * filters,
            2 * filters,
            kernel_size=3,
            padding=1,
            binary_input=binary_pre,
            **common,
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = make_bn2d(2 * filters, self.bn_mode)

        self.conv5 = BinaryConv2d(
            2 * filters,
            4 * filters,
            kernel_size=3,
            padding=1,
            binary_input=binary_pre,
            **common,
        )
        self.bn5 = make_bn2d(4 * filters, self.bn_mode)

        self.conv6 = BinaryConv2d(
            4 * filters,
            4 * filters,
            kernel_size=3,
            padding=1,
            binary_input=binary_pre,
            **common,
        )
        self.pool6 = nn.MaxPool2d(2, 2)
        self.bn6 = make_bn2d(4 * filters, self.bn_mode)

        flatten_features = 4 * filters * 3 * 3

        self.fc1 = BinaryLinear(
            flatten_features,
            dense_units,
            binary_input=binary_pre,
            **common,
        )
        self.bn7 = make_bn1d(dense_units, self.bn_mode)

        self.fc2 = BinaryLinear(
            dense_units,
            dense_units,
            binary_input=binary_pre,
            **common,
        )
        self.bn8 = make_bn1d(dense_units, self.bn_mode)

        # Floating-point final classifier for FedVote.
        self.fc3 = nn.Linear(dense_units, num_classes, bias=False)

        # Legacy final BN is retained only for non-FedVote paths.
        self.bn9 = make_bn1d(num_classes, self.bn_mode)

        if self.weight_mode == WEIGHT_TANH:
            self._init_fedvote_from_reference()
        else:
            self._init_non_fedvote_final_layer()

        if self.freeze_final_layer_flag:
            self.freeze_final_layer()

    def iter_quantized_weight_layers(self):
        for name in (
            "conv1",
            "conv2",
            "conv3",
            "conv4",
            "conv5",
            "conv6",
            "fc1",
            "fc2",
        ):
            yield name, getattr(self, name)

    @torch.no_grad()
    def _init_fedvote_from_reference(self):
        """
        Reproduce qnn/initialize.py::init_qnn_latent_params from the official
        FedVote repository, but on our shared BinaryNet topology.

        A single deterministic RNG stream is used so all independently-created
        clients get the same initial reference model.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.final_layer_seed)

            for _, layer in self.iter_quantized_weight_layers():
                ref_w = torch.empty_like(layer.weight)
                nn.init.kaiming_normal_(ref_w, a=0, mode="fan_in")

                std = ref_w.std()
                if (not torch.isfinite(std)) or std.item() <= 0.0:
                    raise RuntimeError("Invalid reference-weight std in FedVote init")

                ref_w = ref_w / std
                ref_w.clamp_(-1.0 + 1e-3, 1.0 - 1e-3)

                latent_h = torch.atanh(ref_w) / self.weight_scale
                layer.weight.copy_(latent_h)

            # In the official code the final FP classifier comes from the same
            # reference model and is then frozen.
            nn.init.kaiming_normal_(self.fc3.weight, a=0, mode="fan_in")

    @torch.no_grad()
    def _init_non_fedvote_final_layer(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.final_layer_seed)
            nn.init.kaiming_normal_(self.fc3.weight, a=0, mode="fan_in")

    def freeze_final_layer(self):
        self.fc3.weight.requires_grad_(False)

    def set_weight_mode(self, mode, scale=None):
        self.weight_mode = int(mode)
        if scale is not None:
            self.weight_scale = float(scale)
        for _, layer in self.iter_quantized_weight_layers():
            layer.set_weight_mode(self.weight_mode, self.weight_scale)

    def set_hard_eval(self, enabled=True):
        """
        Toggle hard-BNN forward for FedVote evaluation only.

        enabled=False:
            quantized layers use tanh(k*h)  [training / soft state]

        enabled=True:
            quantized layers use sign(h)    [deployed hard BNN]

        The latent tensors h are never modified.
        """
        enabled = bool(enabled)

        for _, layer in self.iter_quantized_weight_layers():
            layer.set_hard_eval(enabled)

        self.hard_eval = enabled

    def _post_bn_activation(self, x):
        if self.activation_mode == ACT_RELU_POST_BN:
            return F.relu(x)
        return x

    def forward(self, x):
        x = self._post_bn_activation(self.bn1(self.conv1(x)))

        x = self._post_bn_activation(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self._post_bn_activation(self.bn3(self.conv3(x)))

        x = self._post_bn_activation(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self._post_bn_activation(self.bn5(self.conv5(x)))

        x = self._post_bn_activation(self.bn6(self.conv6(x)))
        x = self.pool6(x)

        x = x.flatten(1)

        x = self._post_bn_activation(self.bn7(self.fc1(x)))
        x = self._post_bn_activation(self.bn8(self.fc2(x)))

        x = self.fc3(x)

        # Official FedVote has no BN after the final classifier.
        if self.bn_mode != BN_STATIC:
            x = self.bn9(x)

        return x
