import torch
import torch.nn as nn

from models.gconv import GConv2d, GMaxPool2d

"""
Group Equivariant versions of All-CNN
"""

class P4AllCNN(nn.Module):
    """
    G-equivariant CNN with G = p4 (The 4 90-degree rotations)

    6 3x3 conv layers, followed by 4x4 conv layer (10 channels each layer)
    relu activation, bn, dropout, after layer 2
    max-pool after last layer
    """

    def __init__(self, in_channels, device="cpu", n_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            GConv2d(in_channels, 48, filter_size=3, stride=1, padding=1, device=device),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            GConv2d(48, 48, filter_size=3, stride=1, padding=1, in_transformations=4, device=device),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            GConv2d(48, 48, filter_size=3, stride=2, padding=1, in_transformations=4, device=device),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            GConv2d(48, 96, filter_size=3, stride=1, padding=1, in_transformations=4, device=device),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            GConv2d(96, 96, filter_size=3, stride=1, padding=1, in_transformations=4, device=device),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            GConv2d(96, 96, filter_size=3, stride=2, padding=1, in_transformations=4, device=device),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            GConv2d(96, 96, filter_size=3, stride=1, padding=1, in_transformations=4, device=device),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            GConv2d(96, 96, filter_size=1, stride=1, padding=0, in_transformations=4, device=device),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            GConv2d(96, 10, filter_size=1, stride=1, padding=0, in_transformations=4, device=device),
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )
        self.fc = nn.Linear(40, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(kernel_size=8)(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(x)
        return x
