import pytest

from language_interpolation.networks import HighOrderAttentionNetwork
from language_interpolation.lightning_datamodule import TransformerDataModule

from omegaconf import DictConfig

def test_attention_network() :
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
    input_data, output = next(iter(train_dataloader))
    print('input shape', input_data.shape, "output.shape", output.shape)

    network = HighOrderAttentionNetwork(
        layers = [[10, 5],[5, 5]],
        n=3,
        segments=2,
        normalization=lambda x: x,
        layer_type="continuous"
    )
    result = network(input_data)
    print('result', result.shape)
    assert result.shape[0] == 32
    assert result.shape[1] == 1