import pytest

from language_interpolation.utils import reshape_apply
from high_order_layers_torch.networks import HighOrderMLP
import torch


def test_attention_network():
    net = HighOrderMLP(
        layer_type="continuous",
        n=5,
        in_width=10,
        out_width=11,
        hidden_layers=1,
        hidden_width=5,
        in_segments=2,
        out_segments=2,
        hidden_segments=2,
        device="cpu",
    )

    x = torch.rand(7, 5, 10)
    ans = reshape_apply(x, net)

    assert ans.shape == torch.Size([7, 5, 11])
