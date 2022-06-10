from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from torchmetrics.functional import accuracy
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule, Trainer, Callback
import torch.optim as optim
import torch
from high_order_layers_torch.networks import *
from torchsummary import summary
from language_interpolation.single_text_dataset import dataset_registry
from language_interpolation.utils import generate_text
from language_interpolation.lightning_datamodule import GutenbergDataModule
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping
from language_interpolation.utils import (
    generate_text,
    TextGenerationSampler,
    create_gutenberg_cache,
)
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
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
        self.root_dir = f"{hydra.utils.get_original_cwd()}"
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

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


@hydra.main(config_path="../config", config_name="overflow_config")
def run_language_interpolation(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    create_gutenberg_cache(parent_directory=hydra.utils.get_original_cwd())

    if cfg.train is True:

        if cfg.data.type in dataset_registry:
            dataset_generator = dataset_registry[cfg.data.type]
        else:
            raise ValueError(
                f"data.type must be centered or sequence. recieved {cfg.data.type}"
            )

        datamodule = GutenbergDataModule(
            features=cfg.mlp.features,
            targets=1,
            num_workers=cfg.data.num_workers,
            pre_process_workers=cfg.data.pre_process_workers,
            gutenberg_ids_train=cfg.data.train.gutenberg_ids,
            gutenberg_ids_val=cfg.data.val.gutenberg_ids,
            gutenberg_ids_test=cfg.data.test.gutenberg_ids,
            gutenberg_range_train=cfg.data.train.gutenberg_range,
            gutenberg_range_val=cfg.data.val.gutenberg_range,
            gutenberg_range_test=cfg.data.test.gutenberg_range,
            train_filenames=cfg.data.train.filenames,
            val_filenames=cfg.data.val.filenames,
            test_filenames=cfg.data.test.filenames,
            max_size=cfg.data.max_size,
            dataset_generator=dataset_generator,
        )

        early_stopping = EarlyStopping(monitor="train_loss", patience=10)
        trainer = Trainer(
            callbacks=[early_stopping, TextGenerationSampler(cfg)],
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            gradient_clip_val=cfg.gradient_clip,
        )

        model = Net(cfg)
        trainer.fit(model, datamodule=datamodule)
        logger.info("testing")

        result = trainer.test(model)
        logger.info(f"result {result}")
        logger.info("finished testing")
        logger.info(f"best check_point {trainer.checkpoint_callback.best_model_path}")
        logger.info(f"loss {result[0]['train_loss']}")
        return result[0]["train_loss"]
    else:
        # plot some data
        logger.info("evaluating result")
        logger.info(f"cfg.checkpoint {cfg.checkpoint}")
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        logger.info(f"checkpoint_path {checkpoint_path}")
        model = Net.load_from_checkpoint(checkpoint_path)

        text_in = cfg.text
        features = cfg.mlp.input.width

        final = generate_text(
            model=model,
            features=features,
            text_list=[text_in],
            output_size=1,
            topk=cfg.topk,
        )

        logger.info(f"output: {final}")


if __name__ == "__main__":
    run_language_interpolation()
