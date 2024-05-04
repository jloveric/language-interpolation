import pytest

from language_interpolation.dual_convolutional_network import DualConvolutionalNetwork
import torch


def test_dual_convolution():
    net = DualConvolutionalNetwork(n=3, out_width=10, hidden_layers=2, hidden_width=10, in_segments=128, segments=5)
    x = torch.rand(10, 20, 15)
    res = net(x)
    print('res', res)