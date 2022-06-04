from typing import List, Union

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from torchmetrics.functional import accuracy
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule, Trainer
import torch.optim as optim
import torch
from high_order_layers_torch.networks import *
from language_interpolation.single_text_dataset import SingleTextDataset
from torchsummary import summary
from language_interpolation.single_text_dataset import (
    dataset_from_file,
    encode_input_from_text,
    decode_output_to_text,
    ascii_to_float,
    generate_dataset,
    dataset_centered,
)
import random
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn as nn
import itertools
import operator


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        normalization = None
        if cfg.normalize is True:
            normalization = torch.nn.BatchNorm1d(num_features=cfg.fcn.features)

        self.fcn = HighOrderFullyConvolutionalNetwork(
            layer_type=cfg.fcn.layer_type,
            n=cfg.fcn.n,
            channels=cfg.fcn.channels,
            segments=cfg.fcn.segments,
            kernel_size=cfg.fcn.kernel_size,
            periodicity=cfg.fcn.periodicity,
            normalization=normalization,
            rescale_output=False,
        )

        print(cfg.fcn.kernel_size)
        print(cfg.fcn.channels)
        print(cfg.fcn.segments)
        print(cfg.fcn.kernel_size)

        reduction = sum([a - 1 for a in cfg.fcn.kernel_size])

        in_features = cfg.fcn.features - reduction

        self.output_layer = high_order_fc_layers(
            n=cfg.out.n,
            in_features=in_features,
            out_features=128,  # 128 ascii characters
            layer_type=cfg.out.layer_type,
            segments=cfg.out.segments,
        )

        self.flatten = nn.Flatten()

        self.model = nn.Sequential(self.fcn, self.flatten)  # self.output_layer)

        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(top_k=2)

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):

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
            features=self.cfg.fcn.features,
            max_size=self.cfg.data.max_size,
            dataset_generator=dataset_generator,
        )
        self.test_dataset = SingleTextDataset(
            filenames=full_path,
            features=self.cfg.fcn.features,
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


@hydra.main(config_path="./config", config_name="language_config_fcn")
def run_language_interpolation(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        early_stopping = EarlyStopping(monitor="train_loss", patience=5)
        trainer = Trainer(
            callbacks=[early_stopping],
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            gradient_clip_val=cfg.gradient_clip,
        )

        model = Net(cfg)
        trainer.fit(model)
        print("testing")

        result = trainer.test(model)
        print("result", result)
        print("finished testing")
        print("best check_point", trainer.checkpoint_callback.best_model_path)
        print("loss", result[0]["train_loss"])
        return result[0]["train_loss"]
    else:
        # plot some data
        print("evaluating result")
        print("cfg.checkpoint", cfg.checkpoint)
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        print("checkpoint_path", checkpoint_path)
        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()

        text_in = cfg.text
        features = cfg.fcn.input.width

        # Make sure the prompt text is long enough.  The network is expecting a prompt
        # of size features.  It will take the last "features" characters from the
        # provided string and ignore the rest.
        text_in = text_in.rjust(features)

        for i in range(cfg.num_predict):
            encoding, text_used = encode_input_from_text(
                text_in=text_in, features=features
            )
            encoding = ascii_to_float(encoding).unsqueeze(dim=0)
            model.eval()
            output = model(encoding)
            values, indices, ascii = decode_output_to_text(
                encoding=output[0], topk=cfg.topk
            )

            # pick the next character weighted by probabilities of each character
            # prevents the same response for every query.
            actual = random.choices(ascii, values.tolist())
            text_in = text_in + actual[0]

        print("output:", text_in.replace("\n", " "))


if __name__ == "__main__":
    run_language_interpolation()
