from language_interpolation.networks import ASCIIPredictionNet
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from high_order_layers_torch.layers import *
from pytorch_lightning import Trainer
from high_order_layers_torch.networks import *
from language_interpolation.single_text_dataset import (
    dataset_registry,
    RandomizeCharacters,
)
from language_interpolation.utils import generate_text
from language_interpolation.lightning_datamodule import GutenbergDataModule
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from language_interpolation.utils import (
    generate_text,
    TextGenerationSampler,
    create_gutenberg_cache,
)
from language_interpolation.lightning_datamodule import TransformerDataModule

import logging
import traceback

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


@hydra.main(config_path="../config", config_name="high_order_interpolation")
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

            if cfg.net.model_type == "high_order_transformer":
                datamodule = TransformerDataModule(
                    characters_per_feature=cfg.data.characters_per_feature,
                    max_features=cfg.data.max_features,
                    batch_size=cfg.batch_size,
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
                )
            else:
                datamodule = GutenbergDataModule(
                    features=cfg.net.features,
                    targets=1,
                    batch_size=cfg.batch_size,
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
                    add_channel_dimension=cfg.data.add_channel_dimension,
                    transforms=RandomizeCharacters(
                        features=cfg.net.features,
                        symbols=128,
                        random_frac=cfg.data.random_char_frac,
                        add_channel_dimension=cfg.data.add_channel_dimension,
                    ),
                )

            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            early_stopping = EarlyStopping(monitor="train_loss", patience=20)
            trainer = Trainer(
                callbacks=[early_stopping, TextGenerationSampler(cfg), lr_monitor],
                max_epochs=cfg.max_epochs,
                accelerator=cfg.accelerator,
                gradient_clip_val=cfg.gradient_clip,
            )

            model = ASCIIPredictionNet(cfg)
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
            print(traceback.format_exc())
            logger.error(e)
            return 1.0e9
    else:
        # plot some data
        logger.info("evaluating result")
        logger.info(f"cfg.checkpoint {cfg.checkpoint}")
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        logger.info(f"checkpoint_path {checkpoint_path}")
        model = ASCIIPredictionNet.load_from_checkpoint(checkpoint_path)

        add_channel_dimension = True
        if (
            cfg.net.model_type == "high_order"
            or cfg.net.model_type == "high_order_input"
        ):
            add_channel_dimension = False

        text_in = cfg.prompts
        features = cfg.net.features

        final = generate_text(
            model=model,
            features=features,
            text_list=text_in,
            output_size=cfg.num_predict,
            topk=cfg.topk,
            add_channel_dimension=add_channel_dimension,
        )

        logger.info(f"output: {final}")


if __name__ == "__main__":
    run_language_interpolation()
