from omegaconf import DictConfig
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule
import torch.optim as optim
import torch_optimizer as alt_optim
import torch
from high_order_layers_torch.networks import (
    LowOrderMLP,
    HighOrderMLP,
    HighOrderFullyConvolutionalNetwork,
    HighOrderTailFocusNetwork,
    initialize_network_polynomial_layers,
    initialize_polynomial_layer
)
from high_order_layers_torch.layers import MaxAbsNormalizationLast, high_order_fc_layers
from torchmetrics import Accuracy
from torch import Tensor
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ClassificationMixin:
    def eval_step(self, batch: Tensor, name: str):
        x, y, idx = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.flatten())

        diff = torch.argmax(y_hat, dim=1) - y.flatten()
        accuracy = torch.where(diff == 0, 1, 0).sum() / len(diff)

        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", accuracy, prog_bar=True)
        return loss


class RegressionMixin:
    def eval_step(self, batch: Tensor, name: str):
        x, y, idx = batch
        y_hat = self(x)
        loss = self.loss(y_hat.flatten(), y.flatten())

        self.log(f"{name}_loss", loss, prog_bar=True)
        return loss


class PredictionNetMixin:
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def configure_optimizers(self):
        if self.cfg.optimizer.name == "adahessian":
            return alt_optim.Adahessian(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                betas=self.cfg.optimizer.betas,
                eps=self.cfg.optimizer.eps,
                weight_decay=self.cfg.optimizer.weight_decay,
                hessian_power=self.cfg.optimizer.hessian_power,
            )
        elif self.cfg.optimizer.name == "adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.cfg.optimizer.lr,
            )

            reduce_on_plateau = False
            if self.cfg.optimizer.scheduler == "plateau":
                logger.info("Reducing lr on plateau")
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.cfg.optimizer.patience,
                    factor=self.cfg.optimizer.factor,
                    verbose=True,
                )
                reduce_on_plateau = True
            elif self.cfg.optimizer.scheduler == "exponential":
                logger.info("Reducing lr exponentially")
                lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.cfg.optimizer.gamma
                )
            else:
                return optimizer

            scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": reduce_on_plateau,
                "monitor": "train_loss",
            }
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Optimizer {self.cfg.optimizer.name} not recognized")


class HighOrderAttention(torch.nn.Module):
    """
    Basic attention. Done for no other reason than my own understanding
    and experimentation.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        normalization: None,
        query_layer: None,
        key_layer: None,
        value_layer: None,
        output_layer: None,
        heads: int = 1,
        device="cpu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.device = device
        self.heads = heads

        # Default to a linear layer
        self.query_layer = query_layer
        if query_layer is None:
            self.query_layer = torch.nn.Linear(
                in_features=self.embed_dim,
                out_features=self.out_dim * heads,
                device=device,
            )

        self.key_layer = key_layer
        if key_layer is None:
            self.key_layer = torch.nn.Linear(
                in_features=self.embed_dim,
                out_features=self.out_dim * heads,
                device=device,
            )

        self.value_layer = value_layer
        if value_layer is None:
            self.value_layer = torch.nn.Linear(
                in_features=self.embed_dim,
                out_features=self.out_dim * heads,
                device=device,
            )

        self.output_layer = output_layer
        if output_layer is None:
            self.output_layer = torch.nn.Linear(
                in_features=self.out_dim * heads,
                out_features=self.out_dim,
                device=device,
            )

        # It seems pytorch doesn't like lambdas so
        # explicitly define it.
        def noop(x):
            return x

        self.normalization = normalization or noop

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expecting tensors of the form [batch, sequence length, embedding]
        """

        # We want to run each embedding through the same network
        # 1d cnn style.  So we just make a vector with a much larger
        # batch out of the data where each batch contains 1 embedding
        # then convert it back at the end.
        q = query.view(query.shape[0] * query.shape[1], -1)
        k = key.view(key.shape[0] * key.shape[1], -1)
        v = value.view(value.shape[0] * value.shape[1], -1)

        qt = self.query_layer(q)
        kt = self.key_layer(k)
        vt = self.value_layer(v)

        # Turn into [batch,(features/encodings),(encoding size)*heads]
        qt = self.normalization(qt.view(query.shape[0], query.shape[1], qt.shape[1]))
        kt = self.normalization(kt.view(key.shape[0], key.shape[1], kt.shape[1]))
        vt = self.normalization(vt.view(value.shape[0], value.shape[1], vt.shape[1]))

        qth = qt.reshape(qt.shape[0], qt.shape[1], self.heads, -1)
        kth = kt.reshape(kt.shape[0], kt.shape[1], self.heads, -1)
        vth = vt.reshape(vt.shape[0], vt.shape[1], self.heads, -1)

        
        res = F.scaled_dot_product_attention(query=qth,key=kth,value=vth,attn_mask=None)
        # Used built in attention so I can get optimization
        #qkh = torch.nn.functional.softmax(torch.einsum('blhd,brhd->blrh',qth,kth), dim=3)
        #res = torch.einsum('blrh,brhd->blhd',qkh, vth)



        v = res.reshape(res.shape[0] * res.shape[1], -1)
        output = self.output_layer(v)
        final = output.reshape(res.shape[0], res.shape[1], -1)
        self.normalization(final)

        return final


def high_order_attention_block(
    embed_dim: int,
    out_dim: int,
    layer_type: str,
    n: int,
    segments: int,
    normalization=None,
    heads: int = 1,
    device: str = "cpu",
    input_scale: int = 2,
) -> None:
    """
    :param embed_dim: The input embedding dimension
    :param out_dim: The dimension of the output (as well as input to similarity)
    :param layer_type: continuous, discontinuous etc...
    :param segments: number of segments for high order layers
    :param normalization: normalization function
    :param heads: number of attention heads
    :param device: device to run on
    :param input_scale: 2 means input values are between [-1, 1], 20
    means they are between [-10, 10]
    """
    query = high_order_fc_layers(
        layer_type=layer_type,
        n=n,
        in_features=embed_dim,
        out_features=out_dim * heads,
        segments=segments,
        device=device,
        scale=input_scale,
    )
    

    key = high_order_fc_layers(
        layer_type=layer_type,
        n=n,
        in_features=embed_dim,
        out_features=out_dim * heads,
        segments=segments,
        device=device,
        scale=input_scale,
    )
    value = high_order_fc_layers(
        layer_type=layer_type,
        n=n,
        in_features=embed_dim,
        out_features=out_dim * heads,
        segments=segments,
        device=device,
        scale=input_scale,
    )

    # Applied to multi head attention to recombine (output)
    output = high_order_fc_layers(
        layer_type=layer_type,
        n=n,
        in_features=out_dim * heads,
        out_features=out_dim,
        segments=segments,
        device=device,
    )

    initialize_polynomial_layer(query, max_slope=0.1, max_offset=0.5)
    initialize_polynomial_layer(key, max_slope=0.1, max_offset=0.5)
    initialize_polynomial_layer(value, max_slope=0.1, max_offset=0.5)
    initialize_polynomial_layer(output, max_slope=0.1, max_offset=0.5)

    layer = HighOrderAttention(
        embed_dim=embed_dim,
        out_dim=out_dim,
        normalization=normalization,
        query_layer=query,
        key_layer=key,
        value_layer=value,
        output_layer=output,
        heads=heads,
        device=device,
    )
    return layer


def small_character_spacing(x, max_context, positional_embedding):
    xp = (
        (0.5 * (x + 1) + positional_embedding[: x.shape[1]]) * 2 - max_context
    ) / max_context
    return xp


def large_character_spacing(x, max_context, positional_embedding):
    """
    The positional embedding moves the xp slighly up or down, however, we don't
    want it to cause the original xp values to overlap (as they represent characters)
    so dposition < 1 so max_context>=128 (number of ascii values) assuming we have
    128 segments.
    """
    unit_pos_embedding = (positional_embedding[: x.shape[1]] + 0.5) / max_context - 0.5
    xp = (
        ((0.5 * (x + 1) * max_context + unit_pos_embedding)) * 2 - max_context
    ) / max_context
    return xp


class HighOrderAttentionNetwork(torch.nn.Module):
    def __init__(
        self,
        layer_type: str,
        layers: list,
        n: int,
        output_hidden_layers: int,
        output_hidden_width: int,
        output_segments: int,
        normalization: None,
        heads: int = 1,
        device: str = "cuda",
        max_context: int = 10,
    ):
        super().__init__()
        self._device = device
        self.layer = []
        self.max_context = max_context
        self.normalization = normalization

        for index, element in enumerate(layers):
            input_scale = 2.0
            # if index == 0:
            #    input_scale = max_context

            embed_dim = element[0]
            out_dim = element[1]
            segments = element[2]
            new_layer = high_order_attention_block(
                embed_dim=embed_dim,
                out_dim=out_dim,
                layer_type=layer_type,
                segments=segments,
                n=n,
                normalization=normalization,
                heads=heads,
                device=device,
                input_scale=input_scale,
            )
            self.layer.append(new_layer)

        out_dim = layers[-1][1]
        mlp_normalization = None
        if normalization is not None :
            mlp_normalization = MaxAbsNormalizationLast

        self._output_layer = HighOrderMLP(
            layer_type=layer_type,
            n=n,
            in_width=out_dim,
            in_segments=output_segments,
            out_segments=output_segments,
            hidden_segments=output_segments,
            hidden_layers=output_hidden_layers,
            hidden_width=output_hidden_width,
            out_width=128,
            device=self._device,
            normalization=mlp_normalization,
        )

        initialize_network_polynomial_layers(self._output_layer, max_slope=0.1, max_offset=0.5)

        # Make the positions 0 to max_context-1
        self.positional_embedding = (
            torch.arange(max_context, dtype=torch.get_default_dtype())
            .unsqueeze(1)
            .to(device=self._device)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Scale the input to [-1, 1] where every token is bumped by 1/(2*max_context)
        # the 0th token is -1 and the nth token is 1
        # THIS LOOKS RIGHT!

        # characters are small spacinb
        # xp = small_character_spacing(x=x, max_context=self.max_context, positional_embedding=self.positional_embedding)
        # characters are large spacing
        xp = large_character_spacing(
            x=x,
            max_context=self.max_context,
            positional_embedding=self.positional_embedding,
        )

        query = xp
        key = xp
        value = xp
        for index, layer in enumerate(self.layer):
            res = layer(query, key, value)
            query = res
            key = res
            value = res

        average = torch.sum(res, dim=1) / res.shape[1]
        
        if self.normalization is not None :
            final = self.normalization(self._output_layer(average))
        else :
            final = self._output_layer(average)
        
        #torch.cuda.empty_cache()
        return final
        # return self.model(x)


def select_network(cfg: DictConfig, device: str = None):
    normalization = None
    if cfg.net.normalize is True:
        normalization = MaxAbsNormalization  # torch.nn.LazyBatchNorm1d

    if cfg.net.model_type == "high_order_input":
        """
        Only the input layer is high order, the rest
        of the layers are standard linear+relu and normalization.
        """
        layer_list = []
        input_layer = high_order_fc_layers(
            layer_type=cfg.net.layer_type,
            n=cfg.net.n,
            in_features=cfg.net.input.width,
            out_features=cfg.net.hidden.width,
            segments=cfg.net.input.segments,
            device=cfg.accelerator,
        )
        layer_list.append(input_layer)

        if normalization is not None:
            layer_list.append(normalization())

        lower_layers = LowOrderMLP(
            in_width=cfg.net.hidden.width,
            out_width=cfg.output.width,
            hidden_width=cfg.net.hidden.width,
            hidden_layers=cfg.net.hidden.layers - 1,
            non_linearity=torch.nn.ReLU(),
            normalization=normalization,
            device=cfg.accelerator,
        )
        layer_list.append(lower_layers)

        model = torch.nn.Sequential(*layer_list)
    elif cfg.net.model_type == "high_order_transformer":

        normalizer = None
        if normalization==True :
            normalizer=MaxAbsNormalizationLast(eps=1e-6)
        

        model = HighOrderAttentionNetwork(
            cfg.net.layer_type,
            cfg.net.layers,
            cfg.net.n,
            normalization=normalizer,
            device=cfg.accelerator,
            heads=cfg.net.heads,
            max_context=cfg.data.max_features,
            output_hidden_layers=cfg.net.output_layer.hidden_layers,
            output_hidden_width=cfg.net.output_layer.hidden_width,
            output_segments=cfg.net.output_layer.segments,
        )

    elif cfg.net.model_type == "high_order":
        """
        Uniform high order model. All layers are high order.
        """
        model = HighOrderMLP(
            layer_type=cfg.net.layer_type,
            n=cfg.net.n,
            n_in=cfg.net.n_in,
            n_hidden=cfg.net.n_in,
            n_out=cfg.net.n_out,
            in_width=cfg.net.input.width,
            in_segments=cfg.net.input.segments,
            out_width=cfg.net.output.width,
            out_segments=cfg.net.output.segments,
            hidden_width=cfg.net.hidden.width,
            hidden_layers=cfg.net.hidden.layers,
            hidden_segments=cfg.net.hidden.segments,
            normalization=normalization,
            device=cfg.accelerator,
        )
    elif cfg.net.model_type == "high_order_conv":
        conv = HighOrderFullyConvolutionalNetwork(
            layer_type=cfg.net.layer_type,
            n=cfg.net.n,
            channels=cfg.net.channels,
            segments=cfg.net.segments,
            kernel_size=cfg.net.kernel_size,
            rescale_output=False,
            periodicity=cfg.net.periodicity,
            normalization=torch.nn.LazyBatchNorm1d,
            stride=cfg.net.stride,
            pooling=None,  # don't add an average pooling layer
        )

        linear = torch.nn.LazyLinear(out_features=cfg.net.out_features)
        model = torch.nn.Sequential(conv, linear)
    elif cfg.net.model_type == "high_order_tail_focus":
        tail_focus = HighOrderTailFocusNetwork(
            layer_type=cfg.net.layer_type,
            n=cfg.net.n,
            channels=cfg.net.channels,
            segments=cfg.net.segments,
            kernel_size=cfg.net.kernel_size,
            rescale_output=False,
            periodicity=cfg.net.periodicity,
            normalization=normalization,  # torch.nn.LazyBatchNorm1d,
            stride=cfg.net.stride,
            focus=cfg.net.focus,
        )

        linear = torch.nn.LazyLinear(out_features=cfg.net.out_features)
        model = torch.nn.Sequential(tail_focus, linear)

        widths, output_sizes = tail_focus.compute_sizes(cfg.net.features)
        logger.info(f"TailFocusNetwork widths {widths} output_sizes {output_sizes}")

    else:
        raise ValueError(
            f"Unrecognized model_type {cfg.model_type} should be high_order, high_order_input or high_order_conv!"
        )

    return model


class ASCIIPredictionNet(ClassificationMixin, PredictionNetMixin, LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = select_network(cfg)

        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(top_k=1)


class RegressionNet(RegressionMixin, PredictionNetMixin, LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = select_network(cfg)

        self.loss = torch.nn.functional.mse_loss
