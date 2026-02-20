import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG7(nn.Module):
    def __init__(self):
        super(VGG7, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "VGG7"

        #CNN
        # block 1
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)

        # block 2
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)

        # block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)

        # block 4
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        # block 5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(512)

        # block 6
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(512)

        # block 7
        self.fc1 = nn.Linear(8192, 1024)
        self.bn7 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, <<NumberOfOutputNodes:integer>>)


    def forward(self, x):

        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh(x)

        # block 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)

        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.htanh(x)

        # block 4
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.bn4(x)
        x = self.htanh(x)

        # block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.htanh(x)

        # block 6
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = self.bn6(x)
        x = self.htanh(x)

        # block 7
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn7(x)
        x = self.htanh(x)

        x = self.fc2(x)

        return x
