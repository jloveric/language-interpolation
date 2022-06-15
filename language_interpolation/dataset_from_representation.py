from torch.nn import Module
from torch import Tensor
import logging
from typing import List, Tuple
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import torch

logger = logging.getLogger(__name__)


def embedding_from_model(
    model: Module,
    model_input: List[Tensor],
    layer_name: str = None,
    batch_size: int = 32,
) -> List[Tensor]:
    """
    Given a model, pass the input through the model and compute
    the features from the designated layer.  Store these features so
    they can be used to store a different model.  Since this is designed
    for sequence data model_input data should be in order
    Args :
      model : The imported torch model
      model_input: Model input as a List.  Each sequence is an element
      of the list.  The tensor has 2 dimensions (batch, values)
      layer_name: Access the layer with this name as determined by "torch_intermediate_layer_getter"
      batch_size: Size of batch to process!
    Return :
      features, targets based on the representation from model
    """

    model.eval()
    out = []
    for sequence in model_input:
        shape = sequence.shape

        # ceil operation
        batches = (shape[0] + batch_size - 1) // batch_size

        batch_list = [
            sequence[k * batch_size : (k + 1) * batch_size, :] for k in range(batches)
        ]

        for features in batch_list:

            model.eval()
            return_layers = {
                layer_name: "embedding",
            }

            mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
            mid_outputs, model_output = mid_getter(features)

            out.append(mid_outputs["embedding"])

    result = torch.cat(out)
    logger.info(f"Embeddings of size {result.shape}")
    return result


def dataset_from_sequential_embedding(
    feature_sequence: List[Tensor], window_size: int = 1, skip: int = 1
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Create an ordered dataset from a list of sequences.  The datasets are sequence sets
    where a window_size number of features is used to predict the next feature.
    Args :
      feature_sequence : A list of tensor representing sequences.  A list is used, where each
      element in the list could be the features from a single book.  We don't want to append
      the lists as the sequence makes no sense at the join.
      window_size : number of features to collect in the new feature list
      skip : The number of features to skip to grab the target value.  If the embedding represents
      10 characters [0:10], then we want the target to be the [10] variable (probably), so skip
      would be 10.
    Returns :
      a tuple of Lists of features and targets where the features and targets are sequential
      for each of the "books" and every element of the list represents a different "book".
    """

    if window_size > skip:
        raise ValueError(f"window_size {window_size} must be greater than {skip}.")

    features_list = []
    targets_list = []
    for sequence in feature_sequence:
        features = []
        targets = []
        for j in range(len(sequence) - skip):
            features.append(sequence[j : (j + window_size), :])
            targets.append(sequence[j + skip, :].reshape(1, -1))

        features_list.append(torch.cat(features))
        targets_list.append(torch.cat(targets))

    return features_list, targets_list
