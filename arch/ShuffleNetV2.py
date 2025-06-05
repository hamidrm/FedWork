from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def get_valid_gn_groups(channels, max_groups=8):
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        if self.stride == 1:
            assert in_channels == out_channels
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d( mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False),
                nn.BatchNorm2d( mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d( mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d( mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d( mid_channels),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d( mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
                nn.BatchNorm2d( mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d( mid_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    

    def __init__(self, num_classes: int = <<NumberOfOutputNodes:integer>>):
        super(ShuffleNetV2, self).__init__()
        self.stage_out_channels = [24, 116, 232, 464, 1024]  # for 1.0x

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.stage_out_channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d( self.stage_out_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.in_channels = self.stage_out_channels[0]
        self.stage2 = self._make_stage(self.stage_out_channels[1], 4)
        self.stage3 = self._make_stage(self.stage_out_channels[2], 8)
        self.stage4 = self._make_stage(self.stage_out_channels[3], 4)

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stage_out_channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(  self.stage_out_channels[4]),
            nn.ReLU(inplace=True)
        )

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stage_out_channels[4], num_classes)

    def _make_stage(self, out_channels, repeats):
        layers = []
        layers.append(ShuffleUnit(self.in_channels, out_channels, 2))
        self.in_channels = out_channels
        for _ in range(repeats - 1):
            layers.append(ShuffleUnit(self.in_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
