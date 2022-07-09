from typing import List, Union

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from high_order_layers_torch.layers import *
from pytorch_lightning import Trainer
from high_order_layers_torch.networks import *
from language_interpolation.networks import LanguageConvNet
from torchsummary import summary
from language_interpolation.single_text_dataset import (
    encode_input_from_text,
    decode_output_to_text,
    ascii_to_float,
)
import random
from pytorch_lightning.callbacks import EarlyStopping
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


@hydra.main(config_path="./config", config_name="language_config_conv")
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

        model = LanguageConvNet(cfg)
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
        model = LanguageConvNet.load_from_checkpoint(checkpoint_path)
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
