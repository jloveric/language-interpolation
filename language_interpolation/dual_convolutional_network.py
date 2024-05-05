"""
Convolutional network that also shares
depth wise convolutions
"""

import torch
from high_order_layers_torch.networks import HighOrderMLP
from torch import Tensor


class DualConvolutionalLayer(torch.nn.Module):
    def __init__(
        self,
        n: str,
        in_width: int,
        out_width: int,
        hidden_layers: int,
        hidden_width: int,
        in_segments: int = None,
        segments: int = None,
        device: str = "cpu",
        normalization=None
    ):
        super().__init__()

        self._out_width = out_width
        self.device = device
        self.interior_normalization = normalization()
        self.input_layer = HighOrderMLP(
            layer_type="continuous",
            n=n,
            in_width=in_width,
            in_segments=in_segments,
            out_width=out_width,
            hidden_layers=0,
            hidden_width=1,
            device=device,
            out_segments=segments,
            hidden_segments=segments,
            normalization=normalization
        )
        self.equal_layers = HighOrderMLP(
            layer_type="continuous",
            n=n,
            in_width=2 * out_width,
            out_width=out_width,
            hidden_layers=hidden_layers,
            hidden_width=hidden_width,
            device=device,
            in_segments=segments,
            out_segments=segments,
            hidden_segments=segments,
            normalization=normalization
        )

    def forward(self, x: Tensor):
        """
        x has shape [B, L, D]
        """

        xshape = x.shape
        nx = x.reshape(xshape[0] * xshape[1], xshape[2])

        val = self.input_layer(nx)
        val = val.reshape(x.shape[0], x.shape[1], -1)
        val = self.interior_normalization(val)

        # Gradients apparently automatically accumulate, though probably want
        # some normalization here
        depth = 0
        while val.shape[1] > 1:
            depth += 1
            if val.shape[1] % 2 == 1:
                # Add padding to the end, hope this doesn't bust anything
                val = torch.cat(
                    [val, torch.zeros(val.shape[0], 1, val.shape[2], device=self.device)], dim=1
                )

            valshape = val.shape
            val = val.reshape(-1, 2 * self._out_width)
            val = self.equal_layers(val)
            val = val.reshape(valshape[0], -1, self._out_width)
        return val.squeeze(1) # There should be an extra dimension at 1


class DualConvolutionNetwork(torch.nn.Module):
    def __init__(
        self,
        n: str,
        in_width: int,
        out_width: int,
        embedding_dimension: int,
        hidden_layers: int,
        hidden_width: int,
        in_segments: int = None,
        segments: int = None,
        device: str = "cpu",
        normalization=None
    ):
        super().__init__()
        self.dual_layer = DualConvolutionalLayer(
            n=n,
            in_width=in_width,
            out_width=embedding_dimension,
            hidden_layers=hidden_layers,
            hidden_width=hidden_width,
            in_segments=in_segments,
            segments=segments,
            device=device,
            normalization=normalization
        )

        self.output_mlp = torch.nn.Linear(
            in_features=embedding_dimension,
            out_features=out_width,
            device=device,
        )

    def forward(self, x):
        x = self.dual_layer(x)
        x = self.output_mlp(x)
        return x
