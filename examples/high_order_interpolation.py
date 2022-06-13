from typing import List

from language_interpolation.networks import ASCIIPredictionNet
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule, Trainer
import torch.optim as optim
import torch
from high_order_layers_torch.networks import *
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


@hydra.main(config_path="../config", config_name="language_config")
def run_language_interpolation(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    create_gutenberg_cache(parent_directory=hydra.utils.get_original_cwd())

    if cfg.train is True:

        try:  # Try is needed for multirun case
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

            early_stopping = EarlyStopping(monitor="train_loss", patience=20)
            trainer = Trainer(
                callbacks=[early_stopping, TextGenerationSampler(cfg)],
                max_epochs=cfg.max_epochs,
                gpus=cfg.gpus,
                gradient_clip_val=cfg.gradient_clip,
            )
            root_dir = f"{hydra.utils.get_original_cwd()}"

            model = ASCIIPredictionNet(cfg, root_dir=root_dir)
            trainer.fit(model, datamodule=datamodule)
            logger.info("testing")

            result = trainer.test(model, datamodule=datamodule)
            logger.info(f"result {result}")
            logger.info("finished testing")
            logger.info(
                f"best check_point {trainer.checkpoint_callback.best_model_path}"
            )
            logger.info(f"loss {result[0]['test_loss']}")
            return result[0]["test_loss"]
        except Exception as e:
            logger.error(e)
            return 1.0e9
    else:
        # plot some data
        logger.info("evaluating result")
        logger.info(f"cfg.checkpoint {cfg.checkpoint}")
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        logger.info(f"checkpoint_path {checkpoint_path}")
        model = ASCIIPredictionNet.load_from_checkpoint(checkpoint_path)

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
