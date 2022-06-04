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
from language_interpolation.single_text_dataset import SingleTextDataset
from torchsummary import summary
from language_interpolation.single_text_dataset import (
    encode_input_from_text,
    decode_output_to_text,
    ascii_to_float,
)
import random
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping


def generate_text(
    model: nn.Module,
    features: int,
    text_list: List[str],
    output_size: int,
    topk: int = 1,
):

    model.eval()

    features = features

    # Make sure the prompt text is long enough.  The network is expecting a prompt
    # of size features.  It will take the last "features" characters from the
    # provided string and ignore the rest.
    text_list = [text.rjust(features) for text in text_list]

    results = []
    for text_in in text_list:
        for i in range(output_size):
            encoding, text_used = encode_input_from_text(
                text_in=text_in, features=features
            )
            encoding = ascii_to_float(encoding).unsqueeze(dim=0).to(model.device)
            model.eval()
            output = model(encoding)
            values, indices, ascii = decode_output_to_text(
                encoding=output[0], topk=topk
            )

            # pick the next character weighted by probabilities of each character
            # prevents the same response for every query.
            actual = random.choices(ascii, values.tolist())
            text_in = text_in + actual[0]

        results.append(text_in.replace("\n", " "))

    return results


class TextGenerationSampler(Callback):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):

        for topk in range(1, self._cfg.topk + 1):
            predictions = generate_text(
                pl_module,
                features=self._cfg.mlp.features,
                text_list=self._cfg.prompts,
                output_size=self._cfg.num_predict,
                topk=topk,
            )

            for index, text in enumerate(predictions):
                trainer.logger.experiment.add_text(
                    f"topk={topk}_prompt={self._cfg.prompts[index]}",
                    text,
                    global_step=trainer.global_step,
                )