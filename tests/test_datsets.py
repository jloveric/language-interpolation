import pytest
from language_interpolation.single_text_dataset import dataset_from_gutenberg


def test_dataset_from_gutenberg():
    features = 10
    targets = 1

    features, targets = dataset_from_gutenberg(
        gutenberg_id=1000, features=features, targets=targets, max_size=-1
    )

    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == features
    assert targets.shape[1] == targets
