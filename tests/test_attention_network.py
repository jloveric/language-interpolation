import pytest

from language_interpolation.networks import (
    HighOrderAttentionNetwork,
    large_character_spacing,
    small_character_spacing,
)
from language_interpolation.lightning_datamodule import TransformerDataModule
from high_order_layers_torch.layers import MaxAbsNormalizationLast
from high_order_layers_torch.networks import initialize_network_polynomial_layers
from omegaconf import DictConfig
from language_interpolation.utils import generate_transformer_text
import torch

@pytest.mark.parametrize("max_features", [3,4,100])
@pytest.mark.parametrize("characters_per_feature", [1, 2, 10])
def test_attention_network(max_features, characters_per_feature):

    data_module = TransformerDataModule(
        characters_per_feature=characters_per_feature,
        max_features=max_features,
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
    assert input_data.shape[2] == characters_per_feature

    normalization = MaxAbsNormalizationLast(eps=1e-6)

    network = HighOrderAttentionNetwork(
        layers=[
            {
                "input": characters_per_feature,
                "output": 10,
                "hidden": 10,
                "layers": 1,
                "segments": 3,
                "input_segments": 3,
            },
            {"input": 10, "output": 5, "segments": 3},
            {"input": 5, "output": 5, "segments": 2},
            {"input": 5, "hidden": 5, "segments": 2, "layers": 1},
        ],
        n=3,
        normalization=normalization,
        layer_type="continuous",
        device="cpu",
        heads=2,
        max_context=max_features,
    )

    initialize_network_polynomial_layers(network, max_slope=1.0, max_offset=0.0)

    result = network(input_data)
    
    assert result.shape[0] == 32
    assert result.shape[1] == 128

    new_sample = torch.rand(1, max_features, characters_per_feature) * 2 - 1

    print('new_sample.shape', new_sample.shape, 'positional_embedding', network.positional_embedding.shape )
    output = large_character_spacing(
        x=new_sample,
        max_context=network.max_context,
        positional_embedding=network.positional_embedding,
    )
    print("output", output)
    
    text_list = ["hello sir", "Test this now"]

    output_size = 10
    ans = generate_transformer_text(
        model=network,
        text_list=text_list,
        characters_per_feature=characters_per_feature,
        max_characters=max_features*characters_per_feature,
        output_size=output_size,
    )

    for index, text in enumerate(text_list):
        assert len(text) + output_size == len(ans[index])
    