from typing import List, Any

from high_order_layers_torch.layers import *
from pytorch_lightning import Callback
from high_order_layers_torch.networks import *
from language_interpolation.single_text_dataset import (
    encode_input_from_text,
    decode_output_to_text,
    ascii_to_float,
)
from torch import nn
import random
from gutenbergpy.gutenbergcache import GutenbergCache, GutenbergCacheSettings
import logging
from pathlib import Path
import copy

logger = logging.getLogger(__name__)

def reshape_apply(x : Tensor, layer: Any) :
    """
    TODO: Move this to high order layers torch
    Linear layer works on arbitrary shaped tensors, but the
    High Order Layers do not, so this just solves it for the second
    case, but also works for the first
    Args:
        x : The tensor that needs to be reshaped
        layer: The layer to apply to
    Returns:
        output
    """
    shape = x.shape
    last = shape[-1]
    first = torch.prod(torch.tensor(shape[:-1]))
    flatout : Tensor = layer(x.view(first, last))
    return flatout.view(*shape[:-1],-1)

def create_gutenberg_cache(parent_directory: str):
    """
    Gutenberg will delete your directory contents so make sure that the CacheUnpackDirectory
    is set properly. This happened to me once!
    Args :
        parent_directory : The directory that the gutenberg directory sits in.
    """
    directory = Path(parent_directory) / "gutenberg"
    directory.mkdir(parents=True, exist_ok=True)
    filename = (directory / "gutenbergindex.db").as_posix()
    cache_archive_name = (directory / "rdf-files.tar.bz2").as_posix()
    logger.info(
        f"guttenberg filename {filename}, directory {directory.as_posix()}, cache_archive_name {cache_archive_name}"
    )

    GutenbergCacheSettings.set(
        CacheFilename=filename,
        CacheUnpackDir=(
            directory / "unpack"
        ).as_posix(),  # This needs to be in it's own directory.
        CacheArchiveName=cache_archive_name,
    )

    logger.info(f"Cachesettings {GutenbergCacheSettings.CACHE_RDF_UNPACK_DIRECTORY}")

    if not GutenbergCache.exists():
        logger.info(f"Creating Gutenberg cache {filename}")
        GutenbergCache.create()
    else:
        logger.info("Gutenberg cache exists. Skipping.")

    cache = GutenbergCache.get_cache()

    return cache


def generate_text(
    model: nn.Module,
    features: int,
    text_list: List[str],
    output_size: int,
    topk: int = 1,
    add_channel_dimension: bool = False,
):
    model.eval()

    features = features

    # Make sure the prompt text is long enough.  The network is expecting a prompt
    # of size features.  It will take the last "features" characters from the
    # provided string and ignore the rest.
    text_list = [text.rjust(features) for text in text_list]

    results = []
    for text_in in text_list:
        for i in range(output_size):
            encoding, text_used = encode_input_from_text(
                text_in=text_in, features=features
            )
            encoding = ascii_to_float(encoding).unsqueeze(dim=0).to(model.device)
            if add_channel_dimension is True:
                encoding = encoding.unsqueeze(1)

            model.eval()
            output = model(encoding)
            values, indices, ascii = decode_output_to_text(
                encoding=output[0], topk=topk
            )

            # pick the next character weighted by probabilities of each character
            # prevents the same response for every query.
            actual = random.choices(ascii, values.tolist())
            text_in = text_in + actual[0]

        results.append(text_in.replace("\n", " "))

    return results


def generate_transformer_text(
    model: nn.Module,
    characters_per_feature: int,
    max_characters: int,
    text_list: List[str],
    output_size: int,
    topk: int = 1,
):
    """
    :param characters_per_feature: typically 1, the number of characters that make up
    a feature.
    :param max_characters: The maximum number of characters to use, acts like a moving
    window. Set to 0 if all characters should be used.
    :param text_list: List of prompts
    :param output_size: The number of characters to generate
    :param topk: weighted random selection of the topk next characters
    :returns: the continuation of the prompts, the original text + the next output_size
    characters
    """
    model.eval()

    # We want to pad the left with spaces if the text does not
    # fit exactly into feature size blocks.
    def justify_sample(sample):
        just = max(
            characters_per_feature,
            (((len(sample) - 1) // characters_per_feature) + 1)
            * characters_per_feature,
        )
        return sample.rjust(just)

    results = []
    for text_raw in text_list:
        plain_copy = text_raw
        for i in range(output_size):
            text_in = justify_sample(plain_copy)
            encoding, text_used = encode_input_from_text(
                text_in=text_in, features=max_characters
            )
            encoding = (
                ascii_to_float(encoding)
                .to(model._device)
                .reshape(1, -1, characters_per_feature)
            )
            model.eval()

            output = model(encoding)
            values, indices, ascii = decode_output_to_text(
                encoding=output[0], topk=topk
            )

            # pick the next character weighted by probabilities of each character
            # prevents the same response for every query.
            values = values.nan_to_num(nan=1.0)
            actual = random.choices(ascii, values.tolist())
            plain_copy = plain_copy + actual[0]

        results.append(plain_copy.replace("\n", " "))

    return results

def generate_mamba_text(
    model: nn.Module,
    characters_per_feature: int,
    max_characters: int,
    text_list: List[str],
    output_size: int,
    topk: int = 1,
):
    """
    TODO: This can be done much more efficiently because right now I'm re-computing
    the output from the 0th input every time, while those may be identical (I think)
    and so can be re-used. I think I have quadratic generation here when it should
    be linear.

    :param characters_per_feature: typically 1, the number of characters that make up
    a feature.
    :param max_characters: The maximum number of characters to use, acts like a moving
    window. Set to 0 if all characters should be used.
    :param text_list: List of prompts
    :param output_size: The number of characters to generate
    :param topk: weighted random selection of the topk next characters
    :returns: the continuation of the prompts, the original text + the next output_size
    characters
    """
    model.eval()


    results = []
    for text_raw in text_list:
        text_in = text_raw
        for i in range(output_size):
            encoding, text_used = encode_input_from_text(
                text_in=text_in, features=max_characters
            )
            encoding = (
                encoding
                .to(model._device)
                .reshape(1, -1, characters_per_feature)
            )
            model.eval()

            output = model(encoding)
            values, indices, ascii = decode_output_to_text(
                encoding=output[0,-1,:], topk=topk
            )

            # pick the next character weighted by probabilities of each character
            # prevents the same response for every query.
            values = values.nan_to_num(nan=1.0)
            actual = random.choices(ascii, values.tolist())
            text_in = text_in + actual[0]

        results.append(text_in.replace("\n", " "))

    return results



class TextGenerationSampler(Callback):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        with torch.no_grad():
            for topk in range(1, self._cfg.topk + 1):
                if self._cfg.net.model_type in [
                    "high_order_transformer",
                    "high_order_input_transformer",
                ]:
                    predictions = generate_transformer_text(
                        pl_module,
                        characters_per_feature=self._cfg.data.characters_per_feature,
                        max_characters=self._cfg.data.characters_per_feature
                        * self._cfg.data.max_features,
                        text_list=self._cfg.prompts,
                        output_size=self._cfg.num_predict,
                        topk=topk,
                    )
                elif self._cfg.net.model_type in ["mamba"]:
                    predictions = generate_mamba_text(
                        pl_module,
                        characters_per_feature=self._cfg.data.characters_per_feature,
                        max_characters=self._cfg.data.characters_per_feature
                        * self._cfg.data.max_features,
                        text_list=self._cfg.prompts,
                        output_size=self._cfg.num_predict,
                        topk=topk,
                    )
                else:
                    predictions = generate_text(
                        pl_module,
                        features=self._cfg.net.features,
                        text_list=self._cfg.prompts,
                        output_size=self._cfg.num_predict,
                        topk=topk,
                        add_channel_dimension=self._cfg.data.add_channel_dimension,
                    )

                for index, text in enumerate(predictions):
                    trainer.logger.experiment.add_text(
                        f"topk={topk}_prompt={self._cfg.prompts[index]}",
                        text.strip(),
                        global_step=trainer.global_step,
                    )
