import pytest
from language_interpolation.single_text_dataset import (
    dataset_sequential,
)
from language_interpolation.overflow_network import OverFlowNetwork
from language_interpolation.networks import ASCIIPredictionNet
from omegaconf import DictConfig


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

    network_list = []
    embedding_layer_list = []

    overflow = OverFlowNetwork(
        network_list=network_list,
        embedding_layer=embedding_layer_list,
        base_features_train=features,
        base_features_val=features,
        base_features_test=features,
        base_targets_train=features,
        base_targets_test=targets,
        base_targets_val=targets,
        window_size=10,
        skip=10,
    )

    train_function_list = []
    overflow.train(train_function=train_function_list)
