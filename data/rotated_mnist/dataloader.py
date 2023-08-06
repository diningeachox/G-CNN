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
import matplotlib.pyplot as plt
import numpy as np

# Data processing and image reading modules
import torch
import torchvision
from PIL import Image
from skimage import io, transform
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import datasets, models, transforms

# Progress bar
from tqdm import tqdm

absolute_path = os.path.dirname(__file__)

# Code from https://github.com/fillassuncao/denser-models/blob/master/load_data.py
def load_mat_mnist_rotated(path):
    f_content = None

    with open(path, "r") as f:
        f_content = f.readlines()

    x = list()
    y = list()
    if f_content is not None:
        for instance in f_content:
            instance_split = instance.rstrip().lstrip().replace("\n", "").split(" ")
            _class_ = int(float(instance_split[-1]))
            _image_ = map(float, instance_split[:-1])
            x.append(list(_image_))
            y.append(_class_)

    return np.array(x), np.array(y)


class RMNISTDataset(Dataset):
    def __init__(self, train=True, transform=None, rot=True, ref=False):
        data_file = ""
        if train:
            data_file = "data/rotated_mnist/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            data_file = (
                "data/rotated_mnist/mnist_all_rotation_normalized_float_test.amat"
            )
        x, y = load_mat_mnist_rotated(data_file)  # Load data files

        self.images = x.reshape((-1, 1, 28, 28))
        self.labels = y
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        img = self.images[idx]
        label = self.labels[idx]

        print(img.shape)
        # Apply transforms
        if self.transform:
            """
            img_transpose = np.transpose(
                img, (1, 2, 0)
            )  # Transform np array to the format H x W x C
            """
            img = (255 * img).astype(np.uint8).transpose(1, 2, 0)
            img = self.transform(img)

        return img, label


def get_datasets(batch_size):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # Transforms for data augmentation
    train_transforms = transforms.Compose(
        [
            # transforms.RandomCrop((h,w)),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomAffine(degrees=90, translate=(0.0,0.0)),
            # transforms.RandomAffine(degrees=180, translate=(0.0,0.0)),
            # transforms.RandomAffine(degrees=270, translate=(0.0,0.0)),
            torchvision.transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transforms = transforms.Compose(
        [
            # transforms.Resize((h, w)),
            transforms.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_data = RMNISTDataset(train=True, transform=train_transforms)
    test_data = RMNISTDataset(train=False, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":
    trainloader, testloader = get_datasets(4)

    it = iter(trainloader)
    data = next(it)

    fig, ax = plt.subplots()
    im = ax.imshow(data[0][0].permute(1, 2, 0))
    plt.show()
