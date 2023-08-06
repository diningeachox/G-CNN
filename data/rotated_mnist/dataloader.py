"""
    Contains methods for loading and rotating the rotated MNIST images
"""

import os
import pickle

# Timer
import time

import imageio

# Image display
import matplotlib.pylab as plt
import numpy as np

# Data processing and image reading modules
import torch
from PIL import Image
from skimage import io, transform
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import datasets, models, transforms

# Progress bar
from tqdm import tqdm


class RMNISTDataset(Dataset):
    def __init__(self, batch=1, transform=None, rot=True, ref=False):
        pass

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        img = self.images[idx]
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            img_transpose = np.transpose(
                img, (1, 2, 0)
            )  # Transform np array to the format H x W x C
            img = self.transform(img_transpose)

        return img, label
