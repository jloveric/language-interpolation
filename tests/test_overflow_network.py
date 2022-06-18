import pytest
from language_interpolation.single_text_dataset import (
    dataset_sequential,
)
from language_interpolation.overflow_network import OverFlowNetwork, SequentialSamples
from language_interpolation.networks import ASCIIPredictionNet
from language_interpolation.lightning_datamodule import DataModuleFromSequentialDatasets
from omegaconf import DictConfig
from pytorch_lightning import Trainer


def test_dataset_from_gutenberg():
    num_features = 10
    num_targets = 1

    features, targets = dataset_sequential(
        filenames=None,
        gutenberg_ids=[1, 2],
        text=None,
        features=10,
        targets=1,
        max_size=1000,
    )

    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "gpus": 0,
            "lr": 1e-4,
            "batch_size": 256,
            "segments": 2,
            "mlp": {
                "model_type": "high_order",
                "layer_type": "discontinuous",
                "normalize": True,
                "features": 16,
                "n": 2,
                "n_in": 2,
                "n_out": None,
                "n_hidden": None,
                "periodicity": 2.0,
                "rescale_output": False,
                "input": {
                    "segments": 128,
                    "width": 16,
                },
                "output": {
                    "segments": 8,
                    "width": 128,
                },
                "hidden": {
                    "segments": 8,
                    "layers": 2,
                    "width": 10,
                },
            },
        }
    )

    model = ASCIIPredictionNet(cfg)

    network_list = [model]
    embedding_layer_list = ["model.model.5"]

    overflow = OverFlowNetwork(
        network_list=network_list,
        embedding_layer=embedding_layer_list,
        train=SequentialSamples(feature=features, target=targets),
        test=SequentialSamples(feature=features, target=targets),
        val=SequentialSamples(feature=features, target=targets),
        window_size=10,
        skip=10,
    )

    def train_base():
        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            gpus=0,
        )

        base = overflow.get_data_sequence(0)
        model = ASCIIPredictionNet(cfg, root_dir="/tmp")
        trainer.fit(
            model,
            datamodule=DataModuleFromSequentialDatasets(
                train_features=base[0].features,
                train_targets=base[0].targets,
                test_features=base[1].features,
                test_targets=base[1].targets,
                val_features=base[2].features,
                val_targets=base[2].targets,
                max_size=100,
            ),
        )

    def train_other():
        pass

    train_function_list = []
    overflow.train(train_function=train_function_list)
