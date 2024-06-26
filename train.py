import argparse
import pickle
import sys
from time import time

import torch
import torch.nn as nn

import data.cifar10.dataloader as CIFAR10
import data.rotated_mnist.dataloader as RMNIST
from data.cifar10.dataloader import CIFARDataset
from data.rotated_mnist.dataloader import RMNISTDataset
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


def train(
    data, test_data, model_type="p4cnn", num_epochs=100, batch_size=1, device=device
):
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

            pred = torch.argmax(y, dim=1)
            accuracy = torch.sum((pred == label).float()).item()
            # running_metric += accuracy

            print(
                "Epoch {}/{} - Step: [{}/{}] time: {:.3f}s, Batch loss:{:.6f}, Batch accuracy:{}/{} ".format(
                    epoch, 
                    num_epochs - 1,
                    idx + 1,
                    len(data.dataset) // batch_size,
                    time() - step_time,
                    loss.item(),
                    int(accuracy),
                    batch_size,
                )
            )

        #Evaluate every 10 epochs
        if epoch % 10 == 9:
            evaluate(test_data, model, device)


def evaluate(data, model, device=device):
    model.eval()
    total_accuracy = 0

    step_time = time()
    with torch.no_grad():
        for idx, (img, label) in enumerate(data):
            img = img.to(device)
            label = label.to(device)
            y = model(img)

            loss = nn.CrossEntropyLoss()(y, label)

            pred = torch.argmax(y, dim=1)
            total_accuracy += torch.sum((pred == label).float()).item()

        total_accuracy /= len(data.dataset)
    print(
        "Time: {:.3f}s, Batch loss:{:.6f}, Accuracy:{:3f}% ".format(
            time() - step_time, loss.item(), total_accuracy * 100.0
        )
    )


if __name__ == "__main__":
    print("Loading datasets...")
    if args.model == "p4allcnn":
        trainloader, testloader = CIFAR10.get_datasets(batch_size=4)
        print("Beginning training on rotated CIFAR10...")
        train(trainloader, testloader, model_type=args.model, batch_size=4)
    elif args.model == "p4cnn":
        trainloader, testloader = RMNIST.get_datasets(batch_size=16)
        print("Beginning training on rotated MNIST...")
        train(trainloader, testloader, model_type=args.model, batch_size=16)
