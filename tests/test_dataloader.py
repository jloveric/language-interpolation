import pytest
from language_interpolation.single_text_dataset import (
    dataset_from_gutenberg,
    dataset_from_file,
    SingleTextDataset,
    RandomizeCharacters,
    TextTransformerDataset
)
from torch.utils.data import DataLoader, Dataset
import torch

from language_interpolation.lightning_datamodule import TransformerDataModule


def test_transformer_datamodule() :
    data_module = TransformerDataModule(
        characters_per_feature=10,
        max_features=100,
        batch_size=32,
        gutenberg_ids_test=[1],
        gutenberg_ids_train=[2],
        gutenberg_ids_val=[3],
        pre_process_workers=0
    )

    data_module.setup()

    train_dataloader = data_module.train_dataloader()

    for index, element in enumerate(iter(train_dataloader)):
        print('index', index)
        print('element', element)
        break

