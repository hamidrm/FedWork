import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    """
    Straight-through estimator for binary activations.

    Forward:
        sign(x) in {-1, +1}

    Backward:
        clipped STE: pass gradient only for |x| <= 1.
    """

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
    """
    Larq-style input quantizer approximation:
        ste_sign(x)
    """

    def __init__(self, binary=True):
        super().__init__()
        self.binary = binary

    def forward(self, x):
        if not self.binary:
            return x

        x = torch.clamp(x, -1.0, 1.0)
        return sign_ste(x)


class CenterOnlyBatchNorm2d(nn.Module):
    """
    PyTorch equivalent of Keras BatchNormalization(scale=False).

    Trainable:
        bias / beta only

    Fixed:
        weight / gamma = 1

    This avoids passing weight=None with trainable bias to F.batch_norm,
    which can trigger invalid gradient shapes in PyTorch backward.
    """

    def __init__(self, num_features, eps=1e-3, keras_momentum=0.99):
        super().__init__()

        self.momentum = 1.0 - keras_momentum
        self.eps = eps

        # Fixed gamma = 1, not trainable.
        self.register_buffer("weight", torch.ones(num_features))

        # Trainable beta.
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
    """
    1D version of Keras BatchNormalization(scale=False).

    Trainable:
        bias / beta only

    Fixed:
        weight / gamma = 1
    """

    def __init__(self, num_features, eps=1e-3, keras_momentum=0.99):
        super().__init__()

        self.momentum = 1.0 - keras_momentum
        self.eps = eps

        # Fixed gamma = 1, not trainable.
        self.register_buffer("weight", torch.ones(num_features))

        # Trainable beta.
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
    """
    BinaryNet/Bop-style convolution.

    Architecture only:
        - optional binary input activation
        - no bias
        - no weight binarization here
    """

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
    """
    BinaryNet/Bop-style dense layer.

    Architecture only:
        - optional binary input activation
        - no bias
        - no weight binarization here
    """

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


class BinaryNet(nn.Module):
    """
    CIFAR-10 BinaryNet architecture used for Bop-style experiments.

    This file defines architecture only.

    It does not:
        - force weights to {-1,+1}
        - initialize weights as binary
        - decide which weights FedBop controls
        - apply Bop updates

    Input assumption:
        CIFAR-10 images should be scaled to [-1,+1] by the dataset pipeline.
    """

    def __init__(
        self,
        num_classes=<<NumberOfOutputNodes:integer:10>>,
        in_channels=<<NumberOfChannels:integer:3>>,
        filters=<<NumberOfFilters:integer:128>>,
        dense_units=<<NumberOfDenseUnits:integer:1024>>,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.filters = filters
        self.dense_units = dense_units

        # Input: 3 x 32 x 32
        # First conv uses valid padding:
        #   32 -> 30
        self.conv1 = BinaryConv2d(
            in_channels,
            filters,
            kernel_size=3,
            padding=0,
            input_quantizer=False,
        )
        self.bn1 = CenterOnlyBatchNorm2d(filters)

        # 30 -> 30 -> pool -> 15
        self.conv2 = BinaryConv2d(
            filters,
            filters,
            kernel_size=3,
            padding=1,
            input_quantizer=True,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = CenterOnlyBatchNorm2d(filters)

        # 15 -> 15
        self.conv3 = BinaryConv2d(
            filters,
            2 * filters,
            kernel_size=3,
            padding=1,
            input_quantizer=True,
        )
        self.bn3 = CenterOnlyBatchNorm2d(2 * filters)

        # 15 -> 15 -> pool -> 7
        self.conv4 = BinaryConv2d(
            2 * filters,
            2 * filters,
            kernel_size=3,
            padding=1,
            input_quantizer=True,
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = CenterOnlyBatchNorm2d(2 * filters)

        # 7 -> 7
        self.conv5 = BinaryConv2d(
            2 * filters,
            4 * filters,
            kernel_size=3,
            padding=1,
            input_quantizer=True,
        )
        self.bn5 = CenterOnlyBatchNorm2d(4 * filters)

        # 7 -> 7 -> pool -> 3
        self.conv6 = BinaryConv2d(
            4 * filters,
            4 * filters,
            kernel_size=3,
            padding=1,
            input_quantizer=True,
        )
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = CenterOnlyBatchNorm2d(4 * filters)

        # 512 * 3 * 3 when filters=128
        flatten_features = 4 * filters * 3 * 3

        self.fc1 = BinaryLinear(
            flatten_features,
            dense_units,
            input_quantizer=True,
        )
        self.bn7 = CenterOnlyBatchNorm1d(dense_units)

        self.fc2 = BinaryLinear(
            dense_units,
            dense_units,
            input_quantizer=True,
        )
        self.bn8 = CenterOnlyBatchNorm1d(dense_units)

        self.fc3 = BinaryLinear(
            dense_units,
            num_classes,
            input_quantizer=True,
        )
        self.bn9 = CenterOnlyBatchNorm1d(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.pool6(x)
        x = self.bn6(x)

        x = x.flatten(1)

        x = self.fc1(x)
        x = self.bn7(x)

        x = self.fc2(x)
        x = self.bn8(x)

        x = self.fc3(x)
        x = self.bn9(x)

        return x