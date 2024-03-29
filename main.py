import argparse
import pickle
import sys
from time import time

import torch
import torch.nn as nn
from torch.autograd import gradcheck

from data.cifar10.dataloader import CIFARDataset
from models.gconv import GConv2d
from models.p4allcnn import P4AllCNN
from models.p4cnn import P4CNN

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", action="store_true")

args = parser.parse_args()

if __name__ == "__main__":
    gpu = sys.argv[1:]
    # device = 'cpu'
    device = "cuda" if args.gpu else "cpu"
    print(device)
    net = P4CNN(3, device=device)
    # net = P4AllCNN(3, device=device).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # Forward
    a = torch.rand(2, 3, 28, 28).to(device)
    b = torch.rand(4, 3, 28, 28).to(device)

    start = time()
    y = net(a)
    end = time()
    print(f"Time elapsed: {end - start} s")

    # Backward
    target = torch.zeros_like(y).to(device)
    loss = nn.MSELoss()(y, target)

    start = time()
    loss.backward()
    end = time()

    # for p in net.parameters():
    # print(p.grad.norm())

    # optimizer.step()
    print(f"Backward time: {end - start} s")

    start = time()
    y = net(a)
    end = time()
    print(f"Time elapsed: {end - start} s")

    # Backward
    target = torch.zeros_like(y).to(device)
    loss = nn.MSELoss()(y, target)

    start = time()
    loss.backward()
    end = time()

    # optimizer.step()
    print(f"Backward time: {end - start} s")
