import argparse
import pickle
import sys
from time import time

import torch
import torch.nn as nn

from data.cifar10.dataloader import CIFARDataset, get_datasets
from models.gconv import GConv2d
from models.p4allcnn import P4AllCNN
from models.p4cnn import P4CNN

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="p4cnn")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

# get learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def train(data, model_type="p4cnn", num_epochs=10, batch_size=1, device=device):
    if model_type == "p4allcnn":
        model = P4AllCNN(3, device=device).to(device)
    else:
        model = P4CNN(3, device=device).to(device)
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3).
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, verbose=1
    )
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print("Epoch {}/{}, current lr={}".format(epoch, num_epochs - 1, current_lr))

        # ----------------Training---------------
        model.train()
        for idx, (img, label) in enumerate(data):
            step_time = time()
            img = img.to(device)
            label = label.to(device)
            y = model(img)

            loss = nn.CrossEntropyLoss()(y, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # for p in model.parameters():
            # print(p.norm())

            pred = torch.argmax(y, dim=1)
            accuracy = torch.sum((pred == label).float()).item()
            # running_metric += accuracy

            print(
                "Step: [{}/{}] time: {:.3f}s, Batch loss:{:.6f}, Batch accuracy:{}/{} ".format(
                    idx,
                    int(10000 / batch_size),
                    time() - step_time,
                    loss.item(),
                    int(accuracy),
                    batch_size,
                )
            )


if __name__ == "__main__":
    print("Loading datasets...")
    if args.model == "p4allcnn":
        trainloader, testloader = get_datasets(batch_size=4)

        print("Beginning training...")
        train(trainloader, model_type=args.model, batch_size=4)
