import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=<<NumberOfInputChannels:integer>>, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * (<<InputDataLen:integer>> // 32) * (<<InputDataLen>> // 32), 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, <<NumberOfOutputClasses:integer>>)
        
    def forward(self, x):
        # First block
        x = self.pool(F.relu(self.conv1(x)))
        # Second block
        x = self.pool(F.relu(self.conv2(x)))
        # Third block
        x = F.relu(self.conv3(x))
        # Fourth block
        x = F.relu(self.conv4(x))
        # Fifth block
        x = self.pool(F.relu(self.conv5(x)))
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
