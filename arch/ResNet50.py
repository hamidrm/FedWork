
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, fuseable=False):
        super(Bottleneck, self).__init__()
        ### - Quantization addition
        self.fuseable = fuseable
        if fuseable:
            self.skip_add = nn.quantized.FloatFunctional()
        ###
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)  # Replace BatchNorm2d with GroupNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, planes)  # Replace BatchNorm2d with GroupNorm
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, self.expansion * planes)  # Replace BatchNorm2d with GroupNorm

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, self.expansion * planes)  # Replace BatchNorm2d with GroupNorm
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.fuseable:
            out = self.skip_add.add(self.shortcut(x), out)
        else:
            out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class ResNet50(nn.Module):
    def __init__(self, num_classes=<<NumberOfOutputNodes:integer>>, fuseable=<<IsFuseable:bool:False>>):
        super(ResNet50, self).__init__()
        block = Bottleneck
        num_blocks = [3, 4, 6, 3]   # Standard ResNet50 layout
        self.in_planes = 64
        self.fuseable = fuseable

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # For 32x32 input, you could use nn.AvgPool2d(4) to match ResNet18 exactly.
        # For more general usage (e.g., 224x224 input), nn.AdaptiveAvgPool2d((1,1)) is preferred.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, fuseable=self.fuseable))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out