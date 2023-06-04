import pickle
import torch
import torch.nn as nn
from model import P4CNN
from time import time
import sys
import argparse

'''
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

d = unpickle('./cifar-10-batches-py/data_batch_1')
print(d)
'''
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", action="store_true")

args = parser.parse_args()

if __name__ == "__main__":
    gpu = sys.argv[1:]
    #device = 'cpu'
    device = "cuda" if args.gpu else "cpu"
    print(device)
    net = P4CNN(3, device=device).to(device)

    #Forward
    a = torch.rand(2, 3, 32, 32).to(device)
    start = time()
    y = net(a)
    end = time()
    print(f"Time elapsed: {end - start} s")

    #Backward

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    target = torch.zeros_like(y)
    loss = nn.MSELoss()(y, target)


    start = time()
    loss.backward()
    end = time()

    optimizer.step()
    print(f"Backward time: {end - start} s")
    print(y.shape)
