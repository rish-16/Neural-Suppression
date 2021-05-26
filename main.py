import torch
import torch.nn as nn
import torch.nn.functional as F

from suppress2d import Suppress2d

class SuppressConvNet(nn.Module):
    def __init__(self, in_channels, classes, flatten_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3)
        self.s1 = Suppress2d(p=0.25)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.s2 = Suppress2d(p=0.5)
        self.d1 = nn.Linear(flatten_dim, 256)
        self.d2 = nn.Linear(256, 128)
        self.d3 = nn.Linear(128, 64)
        self.d4 = nn.Linear(64, classes)

    def forward(self, x):
        N, c, h, w = x.shape
        x = F.relu(self.conv1(x))
        x = self.s1(x)
        x = F.relu(self.conv2(x))
        x = self.s2(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))
        out = F.softmax(self.d4(x), dim=-1)

        return out

net = SuppressConvNet(3, 10, 32*28*28)
x = torch.rand(100, 3, 32, 32) 
y = net(x)
print (y.shape) # [100, 10]