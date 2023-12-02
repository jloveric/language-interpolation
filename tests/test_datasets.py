import pytest
from language_interpolation.single_text_dataset import (
    dataset_from_gutenberg,
    dataset_from_file,
    SingleTextDataset,
    RandomizeCharacters,
    TextTransformerDataset,
    ascii_to_float,
    float_to_ascii
)
from torch.utils.data import DataLoader, Dataset
import torch

def test_ascii_to_float():
    sample = torch.arange(128)
    res = ascii_to_float(sample)
    assert torch.all(torch.max(res)<1)
    assert torch.all(torch.min(res)>-1)
    
    reverse = float_to_ascii(res)
    assert torch.all(sample==reverse)

def test_dataset_from_gutenberg():
    num_features = 10
    num_targets = 1

    features, targets = dataset_from_gutenberg(
        gutenberg_id=1000, features=num_features, targets=num_targets, max_size=-1
    )
    print("got gutenberg data")
    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == num_features
    assert targets.shape[1] == num_targets


def test_dataset_from_file():
    num_features = 10
    num_targets = 1

    features, targets = dataset_from_file(
        filename="books/frankenstein.txt",
        features=num_features,
        targets=num_targets,
        max_size=-1,
    )

    print("read dataset from file")

    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == num_features
    assert targets.shape[1] == num_targets


def test_single_text_dataset():
    num_features = 10
    num_targets = 1

    dataset = SingleTextDataset(
        gutenberg_ids=[1, 2], features=num_features, targets=num_targets, num_workers=0
    )
    print("got dataset from ids")
    assert dataset.inputs.shape[0] == dataset.output.shape[0]
    assert dataset.inputs.shape[1] == num_features
    assert dataset.output.shape[1] == num_targets


def test_single_text_dataset_for_conv():
    num_features = 10
    num_targets = 1

    dataset = SingleTextDataset(
        gutenberg_ids=[1, 2],
        features=num_features,
        targets=num_targets,
        num_workers=0,
        add_channel_dimension=True,
    )
    print("got dataset from ids")
    assert dataset.inputs.shape[0] == dataset.output.shape[0]
    assert dataset.inputs.shape[1] == 1
    assert dataset.inputs.shape[2] == num_features
    assert dataset.output.shape[1] == num_targets


# Currently not really they way I want to do this
# I think!
def test_text_transformer_dataset():
    characters_per_feature = 10
    num_targets = 1
    embedding_size=100

    dataset =TextTransformerDataset(
        gutenberg_ids=[1, 2],
        characters_per_feature=characters_per_feature,
        targets=num_targets,
        num_workers=0,
        add_channel_dimension=True,
        embedding_size=embedding_size,
    )
    data = dataset(0)

    print("data size", data[0].shape)
    #assert dataset.inputs.shape[0] == dataset.output.shape[0]
    #assert dataset.inputs.shape[1] == 1
    #assert dataset.inputs.shape[2] == num_features
    #assert dataset.output.shape[1] == num_targets


@pytest.mark.parametrize("add_channel_dimension", [True, False])
def test_randomize_characters(add_channel_dimension: bool):
    num_features = 10
    num_targets = 1

    dataset = SingleTextDataset(
        gutenberg_ids=[1, 2],
        features=num_features,
        targets=num_targets,
        num_workers=0,
        add_channel_dimension=add_channel_dimension,
        transforms=RandomizeCharacters(
            features=num_features,
            symbols=128,
            random_frac=0.25,
            add_channel_dimension=add_channel_dimension,
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        drop_last=True,  # Needed for batchnorm
    )

    it = iter(dataloader)
    vals = next(it)

    if add_channel_dimension is True:
        assert vals[0].shape == torch.Size([2, 1, 10])
    else:
        assert vals[0].shape == torch.Size([2, 10])

    assert vals[1].shape == torch.Size([2, 1])
