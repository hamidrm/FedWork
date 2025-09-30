from torch import Tensor
import torch.nn as nn
import torch.optim as optim

class AllCNN(nn.Module):
    def __init__(self, num_classes=<<NumberOfOutputNodes:integer>>):
        super(AllCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),  # Input: 3x32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),  # Downsampling

            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),  # Downsampling

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x