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
from language_interpolation.single_text_dataset import (
    SingleTextDataset,
    dataset_from_file,
    encode_input_from_text,
    decode_output_to_text,
    ascii_to_float,
    generate_dataset,
    dataset_centered,
)
from language_interpolation.utils import generate_text

import random
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
            normalization = torch.nn.BatchNorm1d(num_features=cfg.mlp.hidden.width)

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

    def setup(self, stage):

        full_path = None
        if self.cfg.filenames is not None:
            full_path = [f"{self.root_dir}/{path}" for path in self.cfg.filenames]

        if self.cfg.data.type == "sequence":
            dataset_generator = generate_dataset
        elif self.cfg.data.type == "centered":
            dataset_generator = dataset_centered
        else:
            raise ValueError(
                f"data.type must be centered or sequence. recieved {self.cfg.data.type}"
            )

        self.train_dataset = SingleTextDataset(
            filenames=full_path,
            gutenberg_ids=self.cfg.gutenberg_ids,
            features=self.cfg.mlp.features,
            max_size=self.cfg.data.max_size,
            dataset_generator=dataset_generator,
        )
        self.test_dataset = SingleTextDataset(
            filenames=full_path,
            gutenberg_ids=self.cfg.gutenberg_ids,
            features=self.cfg.mlp.features,
            max_size=self.cfg.data.max_size,
            dataset_generator=dataset_generator,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.flatten())

        diff = torch.argmax(y_hat, dim=1) - y.flatten()
        accuracy = torch.where(diff == 0, 1, 0).sum() / len(diff)

        self.log(f"train_loss", loss, prog_bar=True)
        self.log(f"acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=10,
        )
        return trainloader

    def test_dataloader(self):

        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=10,
        )
        return testloader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


@hydra.main(config_path="../config", config_name="language_config")
def run_language_interpolation(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    create_gutenberg_cache(parent_directory=hydra.utils.get_original_cwd())

    if cfg.train is True:
        early_stopping = EarlyStopping(monitor="train_loss", patience=5)
        trainer = Trainer(
            callbacks=[early_stopping, TextGenerationSampler(cfg)],
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            gradient_clip_val=cfg.gradient_clip,
        )

        model = Net(cfg)
        trainer.fit(model)
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
