from language_interpolation.dataset_from_representation import (
    embedding_from_model,
    dataset_from_sequential_embedding,
)
from language_interpolation.single_text_dataset import dataset_from_file
from language_interpolation.networks import ASCIIPredictionNet
from omegaconf import DictConfig
import torch


def test_dataset_from_model():

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

    features, targets = dataset_from_file(
        filename="books/frankenstein.txt",
        features=16,
        targets=1,
        max_size=100,
    )

    modules = [
        module
        for module in model.model.modules()
        if not isinstance(module, torch.nn.Sequential)
    ]

    embedding = embedding_from_model(model, [features], layer_name="model.model.5")

    assert embedding[0].shape[0] == 82
    assert embedding[0].shape[1] == 10

    em_features, em_targets = dataset_from_sequential_embedding(
        embedding, window_size=1, skip=1
    )
    assert em_features[0].shape[0] == em_targets[0].shape[0]
    assert em_features[0].shape[1] == 10
    assert em_targets[0].shape[1] == 10

    em_features, em_targets = dataset_from_sequential_embedding(
        embedding, window_size=1, skip=10
    )
    assert em_features[0].shape[0] == em_targets[0].shape[0]
    assert em_features[0].shape[0] == 72
