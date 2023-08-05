import torch
import torch.nn as nn

from models.gconv import GConv2d, GMaxPool2d

"""
In this file we implement some G-equivariant CNNs
"""


class P4CNN(nn.Module):
    """
    G-equivariant CNN with G = p4 (The 4 90-degree rotations)

    6 3x3 conv layers, followed by 4x4 conv layer (10 channels each layer)
    relu activation, bn, dropout, after layer 2
    max-pool after last layer
    """

    def __init__(self, in_channels, device="cpu", n_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            GConv2d(in_channels, 10, filter_size=3, device=device),
            GConv2d(10, 10, filter_size=3, in_transformations=4, device=device),
            nn.BatchNorm2d(40, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            GConv2d(10, 10, filter_size=3, in_transformations=4, device=device),
            nn.BatchNorm2d(40, device=device),
            nn.ReLU(),
            GConv2d(10, 10, filter_size=3, in_transformations=4, device=device),
            nn.BatchNorm2d(40, device=device),
            nn.ReLU(),
            GConv2d(10, 10, filter_size=3, in_transformations=4, device=device),
            nn.BatchNorm2d(40, device=device),
            nn.ReLU(),
            GConv2d(10, 10, filter_size=3, in_transformations=4, device=device),
            nn.BatchNorm2d(40, device=device),
            nn.ReLU(),
            GConv2d(10, 10, filter_size=4, in_transformations=4, device=device),
            nn.BatchNorm2d(40, device=device),
            nn.ReLU(),
        )
        self.fc = nn.Linear(40, n_classes, device=device)

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(x)
        return x
