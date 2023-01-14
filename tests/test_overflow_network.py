import pytest
from language_interpolation.single_text_dataset import (
    dataset_sequential,
)
from language_interpolation.overflow_network import OverFlowNetwork, SequentialSamples
from language_interpolation.networks import ASCIIPredictionNet, RegressionNet
from language_interpolation.lightning_datamodule import DataModuleFromSequentialDatasets
from omegaconf import DictConfig
from pytorch_lightning import Trainer


def test_overflow_network():
    num_features = 10
    num_targets = 1

    start_features = 10
    features, targets = dataset_sequential(
        filenames=None,
        gutenberg_ids=[1, 2],
        text=None,
        features=start_features,
        targets=1,
        max_size=1000,
    )

    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "gpus": 0,
            "lr": 1e-4,
            "batch_size": 16,
            "segments": 2,
            "optimizer": {
                "name": "adam",
                "lr": 1.0e-3,
                "scheduler": "plateau",
                "patience": 10,
                "factor": 0.1,
            },
            "net": {
                "model_type": "high_order",
                "layer_type": "discontinuous",
                "normalize": True,
                "features": start_features,
                "n": 2,
                "n_in": 2,
                "n_out": None,
                "n_hidden": None,
                "periodicity": 2.0,
                "rescale_output": False,
                "input": {
                    "segments": 128,
                    "width": start_features,
                },
                "output": {
                    "segments": 2,
                    "width": 128,
                },
                "hidden": {
                    "segments": 2,
                    "layers": 2,
                    "width": 10,
                },
            },
        }
    )

    parent_cfg = DictConfig(
        content={
            "max_epochs": 1,
            "gpus": 0,
            "lr": 1e-4,
            "batch_size": 16,
            "segments": 2,
            "optimizer": {
                "name": "adam",
                "lr": 1.0e-3,
                "scheduler": "plateau",
                "patience": 10,
                "factor": 0.1,
            },
            "net": {
                "model_type": "high_order",
                "layer_type": "discontinuous",
                "normalize": True,
                "features": 100,  # 10x10
                "n": 2,
                "n_in": 2,
                "n_out": None,
                "n_hidden": None,
                "periodicity": 2.0,
                "rescale_output": False,
                "input": {
                    "segments": 2,
                    "width": 100,
                },
                "output": {
                    "segments": 2,
                    "width": 10,
                },
                "hidden": {
                    "segments": 2,
                    "layers": 2,
                    "width": 10,
                },
            },
        }
    )

    model1 = ASCIIPredictionNet(cfg)
    model2 = RegressionNet(parent_cfg)
    model3 = RegressionNet(parent_cfg)

    network_list = [model1, model2, model3]
    embedding_layer_list = ["model.model.5", "model.model.5", "model.model.5"]

    overflow = OverFlowNetwork(
        network_list=network_list,
        embedding_layer=embedding_layer_list,
        train=SequentialSamples(features=features, targets=targets, index=0),
        test=SequentialSamples(features=features, targets=targets, index=0),
        val=SequentialSamples(features=features, targets=targets, index=0),
        window_size=10,
        skip=10,
    )

    def train_base():
        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            gpus=0,
        )

        base = overflow.get_data_sequence(0)
        model = network_list[0]
        trainer.fit(
            model,
            datamodule=DataModuleFromSequentialDatasets(
                train_features=base[0].features,
                train_targets=base[0].targets,
                test_features=base[1].features,
                test_targets=base[1].targets,
                val_features=base[2].features,
                val_targets=base[2].targets,
            ),
        )

    def train_other(index):
        def compute():
            trainer = Trainer(
                max_epochs=cfg.max_epochs,
                gpus=0,
            )

            base = overflow.get_data_sequence(index)
            print("base[0].shape", base[0].features[0].shape, base[0].targets[0].shape)
            model = network_list[index]
            trainer.fit(
                model,
                datamodule=DataModuleFromSequentialDatasets(
                    train_features=base[0].features,
                    train_targets=base[0].targets,
                    test_features=base[1].features,
                    test_targets=base[1].targets,
                    val_features=base[2].features,
                    val_targets=base[2].targets,
                ),
            )

        return compute

    train_function_list = [train_base, train_other(1), train_other(2)]
    overflow.train(train_function=train_function_list)
