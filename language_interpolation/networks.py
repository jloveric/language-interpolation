from omegaconf import DictConfig
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule
import torch.optim as optim
import torch_optimizer as alt_optim
import torch
from high_order_layers_torch.networks import *
from torchmetrics import Accuracy

import logging

logger = logging.getLogger(__name__)


class ASCIIPredictionNet(LightningModule):
    def __init__(self, cfg: DictConfig, root_dir: str = None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        normalization = None
        if cfg.mlp.normalize is True:
            normalization = torch.nn.LazyBatchNorm1d

        self.model = HighOrderMLP(
            layer_type=cfg.mlp.layer_type,
            n=cfg.mlp.n,
            n_in=cfg.mlp.n_in,
            n_hidden=cfg.mlp.n_in,
            n_out=cfg.mlp.n_out,
            in_width=cfg.mlp.input.width,
            in_segments=cfg.mlp.input.segments,
            out_width=128,  # ascii has 128 characters
            out_segments=cfg.mlp.output.segments,
            hidden_width=cfg.mlp.hidden.width,
            hidden_layers=cfg.mlp.hidden.layers,
            hidden_segments=cfg.mlp.hidden.segments,
            normalization=normalization,
        )
        self.root_dir = root_dir
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(top_k=2)

    def forward(self, x):
        return self.model(x)

    def eval_step(self, batch: Tensor, name: str):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.flatten())

        diff = torch.argmax(y_hat, dim=1) - y.flatten()
        accuracy = torch.where(diff == 0, 1, 0).sum() / len(diff)

        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", accuracy, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    # def configure_optimizers(self):
    #    return optim.Adam(self.parameters(), lr=self.cfg.lr)

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
