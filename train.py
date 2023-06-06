import argparse
import pickle
import sys
from time import time

import torch
import torch.nn as nn

from models.p4cnn import P4CNN
from models.gconv import GConv2d
from data.dataloader import CIFARDataset, get_datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

# get learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def train(data, model_type="p4cnn", num_epochs=10, batch_size=2, device=device):
    model = P4CNN(3, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1)
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

        #----------------Training---------------
        model.train()
        for idx, (img, label) in enumerate(data):
            img = img.to(device)
            label = label.to(device)
            y = model(img)

            loss = nn.CrossEntropyLoss()(y, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = torch.argmax(y, dim=1)

            accuracy = torch.sum((pred==y).float()).item()
            #running_metric += accuracy

            print("Step: [{}/{}] time: {:.3f}s, Batch loss:{:.6f}, Batch accuracy:{}/{} ".format(
                idx, int(len_data / batch_size), time.time() - step_time, loss.item(), int(accuracy), batch_size))



if __name__ == "__main__":
    print("Loading datasets...")
    trainloader, testloader = get_datasets(batch_size=2)
    print("Beginning training...")
    train(trainloader)
