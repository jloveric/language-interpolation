from omegaconf import DictConfig
from high_order_layers_torch.layers import *
import torch
from high_order_layers_torch.networks import *
from language_interpolation.dataset_from_representation import (
    embedding_from_model,
    dataset_from_sequential_embedding,
    DatasetFromRepresentation,
)
from language_interpolation.lightning_datamodule import DataModuleFromSequentialDatasets
import logging
import torch.nn
from torch import Tensor
from typing import NamedTuple, List

logger = logging.getLogger(__name__)


class SequentialSamples(NamedTuple):
    features: list[Tensor]
    targets: list[Tensor]
    index: int


class OverFlowNetwork:
    def __init__(
        self,
        network_list: List[nn.Module],
        embedding_layer: List[str],
        train: SequentialSamples,
        test: SequentialSamples,
        val: SequentialSamples,
        window_size: int = 10,
        skip: int = 10,
    ):
        self._network_list = network_list
        self._embedding_layer = embedding_layer
        self._window_size = window_size
        self._skip = skip

        self._data_sequence = {0: [train, test, val]}

    def compute_dataset_from_network(self, index) -> list[SequentialSamples]:
        """
        Compute the dataset at index+1 from the dataset at index and
        store in data_sequence[index+1]
        Args :
            index : Index of the network and dataset to use
        Returns :
            The dataset in a list of SequentialSamples for train, test, val
        """

        datasets = []
        all_data = self.get_data_sequence(index)
        for i in range(len(all_data)):

            data = all_data[i]

            embeddings: list[Tensor] = embedding_from_model(
                self._network_list[index],
                model_input=data.features,
                layer_name=self._embedding_layer[index],
            )
            features, targets = dataset_from_sequential_embedding(
                embeddings, self._window_size, self._skip
            )
            # To use in computing next datasets
            datasets.append(
                SequentialSamples(features=features, targets=targets, index=i)
            )

        self._data_sequence[index + 1] = datasets
        return datasets

    def get_data_sequence(self, index: int):
        if index not in self._data_sequence:
            raise ValueError(f"No data sequence computed for index {index} yet.")

        return self._data_sequence[index]

    def train(self, train_function: List):

        for index in range(len(train_function)):

            logger.info(f"Training function {index}")
            train_function[index]()

            logger.info(f"Computing dataset from network {index}")
            self.compute_dataset_from_network(index)
