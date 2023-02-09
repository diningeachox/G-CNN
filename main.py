import pickle
import torch
import torch.nn as nn
from model import P4CNN
from time import time

'''
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

d = unpickle('./cifar-10-batches-py/data_batch_1')
print(d)
'''

device = 'cpu'
print(device)
net = P4CNN(3, device=device).to(device)


a = torch.rand(2, 3, 32, 32).to(device)
start = time()
b = net(a)
end = time()
print(f"Time elapsed: {end - start} s")
print(b.shape)
