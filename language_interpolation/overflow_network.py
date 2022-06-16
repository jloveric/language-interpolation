from omegaconf import DictConfig
from high_order_layers_torch.layers import *
import torch
from high_order_layers_torch.networks import *
from language_interpolation.dataset_from_representation import (
    embedding_from_model,
    dataset_from_sequential_embedding,
)
import logging

logger = logging.getLogger(__name__)


class OverFlowNetwork:
    def __init__(self, num_networks: int):
        self._network_list = []
        self._embedding_layer = []
        pass

    def compute_dataset_from_network(self, index):
        data = self.get_dataset(self, index)
        embedding_from_model(
            self._network_list[index],
            model_input=data[i],
            layer_name=self._embedding_layer[index],
        )

    def get_dataset(self, index: int):
        pass

    def train_network(self, index: int):
        pass
