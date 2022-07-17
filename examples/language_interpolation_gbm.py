from typing import List

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
    generate_dataset_char,
    encode_input_from_text,
    decode_output_to_text,
)
import random
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping
import lightgbm as lgb


def train(cfg: DictConfig):
    root_dir = f"{hydra.utils.get_original_cwd()}"

    full_path = [f"{root_dir}/{path}" for path in cfg.filenames]
    raw_features, raw_targets = dataset_from_file(
        filename=full_path[0],
        features=cfg.features,
        targets=1,
        max_size=cfg.data.max_size,
        dataset_generator=generate_dataset_char,
    )

    features = np.array(raw_features)
    targets = np.array(raw_targets)

    feature_names = list(range(cfg.features))
    # categorical_features = f"{feature_names}"

    train_data = lgb.Dataset(
        features, label=targets, categorical_feature=list(range(cfg.features))
    )

    param = {
        "num_leaves": cfg.num_leaves,
        "objective": cfg.objective,
        "num_class": cfg.num_class,
        "max_depth": cfg.max_depth,
    }
    param["metric"] = cfg.metric

    bst = lgb.train(
        params=param,
        train_set=train_data,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[train_data],
    )
    bst.save_model("model.txt")


def predict(text: str):
    pass


@hydra.main(config_path="../config", config_name="language_config_gbm")
def run_language_interpolation(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        train(cfg)
    else:
        # plot some data
        print("evaluating result")
        print("cfg.checkpoint", cfg.checkpoint)
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        print("checkpoint_path", checkpoint_path)

        bst = lgb.Booster(model_file=checkpoint_path)

        text_in = cfg.text
        features = cfg.features

        # Make sure the prompt text is long enough.  The network is expecting a prompt
        # of size features.  It will take the last "features" characters from the
        # provided string and ignore the rest.
        text_in = text_in.rjust(features)

        for i in range(cfg.num_predict):
            encoding, text_used = encode_input_from_text(
                text_in=text_in, features=features
            )
            encoding = encoding.unsqueeze(dim=0)
            output = bst.predict(encoding.numpy())

            values, indices, ascii = decode_output_to_text(
                encoding=torch.tensor(output[0]), topk=cfg.topk
            )

            # pick the next character weighted by probabilities of each character
            # prevents the same response for every query.
            actual = random.choices(ascii, values.tolist())
            text_in = text_in + actual[0]

        print("output:", text_in.replace("\n", " "))


if __name__ == "__main__":
    run_language_interpolation()
