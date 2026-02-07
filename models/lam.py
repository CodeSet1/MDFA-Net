import torch
import torch.nn as nn
import torch.nn.functional as F

class LAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        out = self.fuse(torch.cat([f1, f2, f3], dim=1))
        return x + out
