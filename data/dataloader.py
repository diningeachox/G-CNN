"""
    Contains methods for loading and rotating the CIFAR-10 images
"""

#Progress bar
from tqdm import tqdm

#Timer
import time
import os

#Data processing and image reading modules
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from skimage import io, transform
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

#Image display
import matplotlib.pylab as plt


class CIFARDataset(Dataset):
    def __init__(self, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform #Possible image transforms


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx]
        label = self.labels[idx]

        #Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label
