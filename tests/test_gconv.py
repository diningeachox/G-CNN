import pytest
import torch
from models.p4cnn import P4CNN
from models.gconv import GConv2d
from data.cifar10.dataloader import CIFARDataset, get_datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
g_conv_layer = GConv2d(3, 10, filter_size=3, device=device).to(device)

def test_group_element():
    pass

def test_group_element_inverse():
    pass

def test_equivariance():
    #Load test image
    trainloader, testloader = get_datasets(1)
    it = iter(testloader)
    data = next(it)
    img = data[0].to(device)

    # Feedforward then rotate
    y = g_conv_layer(img)
    rot_y = torch.rot90(y, 1, [2, 3])

    rot_img = torch.rot90(img, 1, [2, 3])
    y_rot_img = g_conv_layer(rot_img)

    #Check that they are the same shape
    assert rot_y.shape == y_rot_img.shape

    #Check equivariance (a low tolerance for floating point)
    assert torch.allclose(rot_y, y_rot_img, atol=1e-05, rtol=1e-5)
    assert not torch.allclose(y, torch.zeros_like(y))

def test_backward():
    pass
