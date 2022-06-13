from torch.nn import Module
from torch import Tensor
import logging
from typing import List
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import torch

logger = logging.getLogger(__name__)


def dataset_from_model(
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
