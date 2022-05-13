import pickle
import torch
import torch.nn as nn

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

d = unpickle('./cifar-10-batches-py/data_batch_1')
print(d)
