import itertools

import pytest
import torch

from data.cifar10.dataloader import CIFARDataset, get_datasets
from models.gconv import GConv2d, group_element, group_element_inverse
from models.p4cnn import P4CNN

device = "cuda" if torch.cuda.is_available() else "cpu"
g_conv_layer = GConv2d(3, 10, filter_size=3, device=device).to(device)


def test_group_element():
    """
    Test transformation of group coordinates into matrices
    """
    g = group_element(1, 0, 0, 0)
    g_true = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(g, g_true, atol=1e-05, rtol=1e-5)

    g = group_element(1, 0, 1, 0)
    g_true = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(g, g_true, atol=1e-05, rtol=1e-5)

    g = group_element(3, 0, 1, 0)
    g_true = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(g, g_true, atol=1e-05, rtol=1e-5)


def test_group_element_inverse():
    """
    Test transformation of matrices into group coordinates
    """
    g_true = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    indices = group_element_inverse(g_true)
    assert indices[0] == 1 and indices[1] == 0 and indices[2] == 1

    g_true = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]])
    indices = group_element_inverse(g_true)
    assert indices[0] == 2 and indices[1] == 0 and indices[2] == 1


def test_indices():
    """
    Check that all indices of the precomuted filter bank are within range
    """
    assert torch.all(g_conv_layer.ind1 >= 0)
    assert torch.all(g_conv_layer.ind1 < g_conv_layer.filter_size)
    assert torch.all(g_conv_layer.ind2 >= 0)
    assert torch.all(g_conv_layer.ind2 < g_conv_layer.filter_size)
    assert torch.all(g_conv_layer.ind3 >= 0)
    assert torch.all(g_conv_layer.ind3 < g_conv_layer.filter_size)


def test_equivariance():
    """
    Test group equivariance of the GConv layers (entry-by-entry)
    """
    # Load test image
    trainloader, testloader = get_datasets(1)
    it = iter(testloader)
    data = next(it)
    img = data[0].to(device)

    # Feedforward then rotate
    # not simply rotating the tensor! Need to permute indices according to the transformation group
    y = g_conv_layer(img)
    rot_y = torch.rot90(y, 1, [2, 3])

    # Rotate image then feedforward
    rot_img = torch.rot90(img, 1, [2, 3])
    y_rot_img = g_conv_layer(rot_img)

    # Check that they are the same shape
    assert y.shape == y_rot_img.shape

    # Check equivariance (a low tolerance for floating point)
    # Compare the results entry-by-entry
    half_x = y.shape[2] / 2.0 - 0.5
    half_y = y.shape[3] / 2.0 - 0.5
    for i, s, u, v in itertools.product(
        range(g_conv_layer.out_channels),
        range(g_conv_layer.out_trans),
        range(y.shape[2]),
        range(y.shape[3]),
    ):
        new_coords = group_element_inverse(
            torch.mm(
                torch.inverse(group_element(1, 0, 0, m=0)),
                group_element(s, v - half_x, -1 * (u - half_y), m=0),
            ),
            rot=True,
        )
        _s = new_coords[0]
        _u = round(-1 * new_coords[2] + half_y)
        _v = round(new_coords[1] + half_x)
        y_pixel = y[0, i * g_conv_layer.out_trans + _s, _u, _v]
        y_rot_img_pixel = y_rot_img[0, i * g_conv_layer.out_trans + s, u, v]
        assert torch.allclose(y_pixel, y_rot_img_pixel, atol=1e-05, rtol=1e-5)

    assert not torch.allclose(y, torch.zeros_like(y))
    assert not torch.allclose(y_rot_img, torch.zeros_like(y))


def test_backward():
    pass
