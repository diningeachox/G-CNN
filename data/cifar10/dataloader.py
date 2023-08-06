"""
    Contains methods for loading and rotating the CIFAR-10 images
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


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class CIFARDataset(Dataset):
    def __init__(self, batch=1, transform=None, rot=True, ref=False):
        data_file = ""
        if batch > 0:
            data_file = f"data/cifar10/data_batch_{batch}"
        else:
            data_file = "data/cifar10/test_batch"
        data_dict = unpickle(data_file)
        images = data_dict[b"data"]

        # Unpack images and store them in array

        images = np.reshape(images, (images.shape[0], 3, 32, 32))
        self.images = images.copy()
        self.labels = data_dict[b"labels"]
        self.transform = transform  # Possible image transforms

        # Add transformations for test batch
        if batch == 0:
            if rot == True:
                for i in range(1, 4):
                    rotated_images = np.rot90(images, i, (2, 3))
                    self.images = np.concatenate((self.images, rotated_images), axis=0)
            # Reflection across y axis
            if ref == True:
                flipped_images = np.flip(images, 3)
                self.images = np.concatenate((self.images, flipped_images), axis=0)
                # Reflection across x axis
                flipped_images = np.flip(images, 2)
                self.images = np.concatenate((self.images, flipped_images), axis=0)
        print(self.images.shape)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        img = self.images[idx]
        label = self.labels[idx]
        # Apply transforms
        if self.transform:
            img = img.transpose(1, 2, 0)
            img = self.transform(img)

        return img, label


def get_datasets(batch_size):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # Transforms for data augmentation
    train_transforms = transforms.Compose(
        [
            # transforms.RandomCrop((h,w)),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomAffine(degrees=90, translate=(0.0,0.0)),
            # transforms.RandomAffine(degrees=180, translate=(0.0,0.0)),
            # transforms.RandomAffine(degrees=270, translate=(0.0,0.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transforms = transforms.Compose(
        [
            # transforms.Resize((h, w)),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_data = CIFARDataset(1, train_transforms)
    test_data = CIFARDataset(0, test_transforms)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    return trainloader, testloader


# Some data exploration and visualization
if __name__ == "__main__":
    trainloader, testloader = get_datasets(4)

    it = iter(testloader)
    data = next(it)

    fig, ax = plt.subplots()
    im = ax.imshow(data[0][0].permute(1, 2, 0))
    plt.show()
