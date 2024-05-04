"""
Convolutional network that also shares
depth wise convolutions
"""

import torch
from high_order_layers_torch.networks import HighOrderMLP
from torch import Tensor


class DualConvolutionalNetwork(torch.nn.Module):
    def __init__(
        self,
        n: str,
        out_width: int,
        hidden_layers: int,
        hidden_width: int,
        in_segments: int = None,
        segments: int = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_layer = HighOrderMLP(
            layer_type="continuous",
            n=n,
            in_width=2,
            in_segments=in_segments,
            out_width=out_width,
            hidden_layers=0,
            hidden_width=1,
            device=device,
            out_segments=segments,
            hidden_segments=segments,
        )
        self.equal_layers = HighOrderMLP(
            layer_type="continuous",
            n=n,
            in_width=out_width,
            out_width=out_width,
            hidden_layers=hidden_layers,
            hidden_width=hidden_width,
            device=device,
            in_segments=segments,
            out_segments=segments,
            hidden_segments=segments,
        )

    def forward(self, x: Tensor):
        """
        x has shape [B, L (sequence length), dimension]
        """

        val = self.input_layer(x)
        while val.shape[1] > 1:
            
            if val.shape[1] % 2 == 1:
                # Add padding to the end, hope this doesn't bust anything
                val = torch.cat([val, torch.zeros(1, val.shape[1], val.shape[2])])
            
            val = self.equal_layers(val)
        return val

