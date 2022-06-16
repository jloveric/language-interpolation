from omegaconf import DictConfig
from high_order_layers_torch.layers import *
import torch
from high_order_layers_torch.networks import *
from language_interpolation.dataset_from_representation import (
    embedding_from_model,
    dataset_from_sequential_embedding,
    DatasetFromRepresentation,
)
from lightning_datamodule import GutenbergDataModule
import logging
import torch.nn

logger = logging.getLogger(__name__)


class OverFlowNetwork:
    def __init__(
        self,
        network_list: List[nn.Module],
        embedding_layer: List[str],
        base_features_train: List[List[Tensor]],
        base_targets_train: List[List[Tensor]],
        base_features_val: List[List[Tensor]],
        base_targets_val: List[List[Tensor]],
        base_features_test: List[List[Tensor]],
        base_targets_test: List[List[Tensor]],
        window_size: int = 10,
        skip: int = 10,
    ):
        self._network_list = network_list
        self._embedding_layer = embedding_layer
        self._window_size = window_size
        self._skip = skip
        self._base_features_train = base_features_train
        self._base_targets_train = base_targets_train
        self._base_features_val = base_features_val
        self._base_targets_val = base_targets_val
        self._base_features_test = base_features_test
        self._base_targets_test = base_targets_test

    def compute_dataset_from_network(self, index):
        """
        Args :
            The arguments
        Returns :
            The dataset
        """
        data = self.get_dataset(index)
        embeddings: List[Tensor] = embedding_from_model(
            self._network_list[index],
            model_input=data,
            layer_name=self._embedding_layer[index],
        )
        features, targets = dataset_from_sequential_embedding(
            embeddings, self._window_size, self._skip
        )
        dataset = DatasetFromRepresentation(features=features, targets=targets)
        return dataset

    def get_dataset(self, index: int):
        """
        Get the dataset to feed into the later
        """

        if index == 0:
            return (
                self._base_features_train,
                self._base_targets_train,
                self._base_features_val,
                self._base_targets_val,
                self._base_features_test,
                self._base_targets_test,
            )
        else:
            pass

    def train_network(self, index: int):
        pass
