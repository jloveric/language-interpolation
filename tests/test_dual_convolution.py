import pytest

from language_interpolation.dual_convolutional_network import (
    DualConvolutionalLayer,
    DualConvolutionNetwork,
)
import torch


def test_dual_convolution():
    net = DualConvolutionalLayer(
        n=3,
        in_width=1,
        out_width=10,
        hidden_layers=2,
        hidden_width=10,
        in_segments=128,
        segments=5,
    )
    x = torch.rand(10, 15, 1)  # character level
    res = net(x)
    print("res", res.shape)

    net = DualConvolutionNetwork(
        n=3,
        in_width=1,
        out_width=10,
        embedding_dimension=5,
        hidden_layers=2,
        hidden_width=10,
        in_segments=128,
        segments=5,
        device="cpu",
    )
    res = net(x)
    print('res', res)
    print('res.shape',res.shape)
