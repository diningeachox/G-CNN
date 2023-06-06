"""
    Contains methods for loading and rotating the rotated MNIST images
"""

#Progress bar
from tqdm import tqdm

#Timer
import time
import os

import numpy as np

#Data processing and image reading modules
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, models, transforms
from skimage import io, transform
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import imageio
#Image display
import matplotlib.pylab as plt
import pickle


class RMNISTDataset(Dataset):
    def __init__(self, batch=1, transform=None, rot=True, ref=False):
        pass

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        img = self.images[idx]
        label = self.labels[idx]

        #Apply transforms
        if self.transform:
            img_transpose = np.transpose(img, (1,2,0)) #Transform np array to the format H x W x C
            img = self.transform(img_transpose)

        return img, label
