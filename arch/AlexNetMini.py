import torch
import torch.nn as nn
import torch.nn.functional as F

'''
AlexNet model architecture from the "One weird trick..." <https://arxiv.org/abs/1404.5997> paper.
'''
class AlexNetMini(nn.Module):

    def __init__(self):
        super(AlexNetMini, self).__init__()
        droprate=<<DropeRate:float>>
        num_classes=<<NumberOfOutputNodes:integer>>
        self.features = nn.Sequential(
            nn.Conv2d(<<NumberOfInputChannels:integer>>, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=4, stride=4),
        )
        if droprate > 0.:
            self.fc = nn.Sequential(nn.Dropout(droprate),
                                    nn.Linear(256, num_classes))
            print("DROPE IS MORE THAN 0!!!")
        else:
            self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.max_pool2d(x,x.shape[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
