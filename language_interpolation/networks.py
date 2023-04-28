from omegaconf import DictConfig
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule
import torch.optim as optim
import torch_optimizer as alt_optim
import torch
from high_order_layers_torch.networks import *
from high_order_layers_torch.layers import MaxAbsNormalization
from torchmetrics import Accuracy
from torch import Tensor

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
    Basic attention. Done for no other reason than may own understanding
    and experimentation.
    """

    def __init__(self, embed_dim: int, out_dim: int, normalization: None):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.query_layer = torch.nn.Linear(
            in_features=self.embed_dim, out_features=self.out_dim
        )
        self.key_layer = torch.nn.Linear(
            in_features=self.embed_dim, out_features=self.out_dim
        )
        self.value_layer = torch.nn.Linear(
            in_features=self.embed_dim, out_features=self.out_dim
        )
        if normalization is None:
            self.normalization = lambda x: x

        self.normalization = normalization

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
        q = query.reshape(query.shape[0] * query.shape[1], -1)
        k = key.reshape(key.shape[0] * key.shape[1], -1)
        v = value.reshape(value.shape[0] * value.shape[1], -1)

        qt = self.query_layer(q)
        kt = self.key_layer(k)
        vt = self.value_layer(v)

        qt = qt.reshape(query.shape[0], query.shape[1], qt.shape[1])
        kt = kt.reshape(key.shape[0], key.shape[1], kt.shape[1])
        vt = vt.reshape(value.shape[0], value.shape[1], vt.shape[1])

        # print("qt.shape", qt.shape, "kt.shape", kt.shape)
        qk = self.normalization(qt @ kt.transpose(1, 2))
        # print("qk.shape", qk.shape)

        # qkv = self.normalization(qk) * vt
        qkv = qk @ vt
        # print("qkv.shape", qkv.shape)

        return qkv




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
        )
        layer_list.append(lower_layers)

        model = nn.Sequential(*layer_list)

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
        model = nn.Sequential(conv, linear)
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
        model = nn.Sequential(tail_focus, linear)

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

        self.loss = F.mse_loss
