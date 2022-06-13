from language_interpolation.dataset_from_representation import embedding_from_model
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
    print("modules", modules)

    ans = embedding_from_model(model, [features], layer_name="model.model.5")

    assert ans.shape[0] == 82
    assert ans.shape[1] == 10
