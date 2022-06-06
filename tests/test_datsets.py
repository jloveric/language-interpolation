import pytest
from language_interpolation.single_text_dataset import (
    dataset_from_gutenberg,
    dataset_from_file,
)


def test_dataset_from_gutenberg():
    num_features = 10
    num_targets = 1

    features, targets = dataset_from_gutenberg(
        gutenberg_id=1000, features=num_features, targets=num_targets, max_size=-1
    )

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

    print("features.shape", features.shape, targets.shape)

    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == num_features
    assert targets.shape[1] == num_targets
