import torch
import torch.nn as nn

def conv_bn_relu(in_c, out_c, k=3, s=1, p=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.GroupNorm(32, out_c),
        nn.ReLU(inplace=True),
    )

class CNNSimpleLite(nn.Module):

    def __init__(self, num_classes=<<NumberOfOutputNodes:integer>>, c1=<<ChannelsNumber1:integer:32>>, c2=<<ChannelsNumber2:integer:64>>, emb=<<EmbeddedSize:integer:64>>):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(1,   c1, k=3, s=1, p=1),   # 28x28
            conv_bn_relu(c1,  c2, k=3, s=2, p=1),   # 14x14 (downsample)
            conv_bn_relu(c2,  c2, k=3, s=1, p=1),   # 14x14
            conv_bn_relu(c2,  c2, k=3, s=2, p=1),   # 7x7   (downsample)
            conv_bn_relu(c2,  emb, k=3, s=1, p=0),  # 5x5   (valid 3x3)
            conv_bn_relu(emb, emb, k=1, s=1, p=0),  # 5x5
        )
        self.gap = nn.AdaptiveAvgPool2d(1)         # -> (B, emb, 1, 1)
        self.fc  = nn.Linear(emb, num_classes)     # final FC head

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)                 # (B, emb)
        x = self.fc(x)                             # (B, num_classes) logits
        return x
