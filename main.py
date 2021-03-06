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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = P4CNN(3, device=device)

start = time()
a = torch.rand(1, 3, 224, 224)
b = net(a)
end = time()
print(f"Time elapsed: {end - start} s")
print(b.shape)
