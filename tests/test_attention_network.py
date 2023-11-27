import pytest

from language_interpolation.networks import HighOrderAttentionNetwork
from language_interpolation.lightning_datamodule import TransformerDataModule

from omegaconf import DictConfig
from language_interpolation.utils import generate_transformer_text


def test_attention_network():
    characters_per_feature = 10

    data_module = TransformerDataModule(
        characters_per_feature=10,
        max_features=100,
        batch_size=32,
        gutenberg_ids_test=[1],
        gutenberg_ids_train=[2],
        gutenberg_ids_val=[3],
        pre_process_workers=0,
    )

    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    input_data, output, indexes = next(iter(train_dataloader))
    print("indexes", indexes)
    print("input shape", input_data.shape, "output.shape", output.shape)

    assert len(indexes) == 32
    assert input_data.shape[0] == 32
    assert input_data.shape[2] == 10

    network = HighOrderAttentionNetwork(
        layers=[[10, 5, 64], [5, 5, 2]],
        n=3,
        normalization=None,
        layer_type="continuous",
        device="cpu",
        heads =2
    )
    result = network(input_data)
    print('result', result)
    print("result", result.shape)
    assert result.shape[0] == 32
    assert result.shape[1] == 128

    text_list = ["hello sir", "Test this now"]
    ans = generate_transformer_text(
        model=network,
        text_list=text_list,
        characters_per_feature=characters_per_feature,
        max_characters=1000,
        output_size=10
    )
    print("ans", ans)
