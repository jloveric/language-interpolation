from typing import Optional, Callable, Tuple
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from typing import List
from language_interpolation.single_text_dataset import (
    SingleTextDataset,
    TextTransformerDataset,
    generate_dataset,
    generate_flat_dataset,
    unify_ids,
    create_full_paths,
)
from language_interpolation.dataset_from_representation import DatasetFromRepresentation
import logging
import random
import torch

logger = logging.getLogger(__name__)


class GutenbergDataModule(pl.LightningDataModule):
    def __init__(
        self,
        features: int,
        targets: int = 1,
        batch_size: int = 32,
        num_workers: int = 10,
        shuffle: bool = True,
        pin_memory: bool = True,
        gutenberg_ids_train: List[int] = None,
        gutenberg_ids_val: List[int] = None,
        gutenberg_ids_test: List[int] = None,
        gutenberg_range_train: List[int] = None,
        gutenberg_range_val: List[int] = None,
        gutenberg_range_test: List[int] = None,
        train_filenames: List[str] = None,
        val_filenames: List[str] = None,
        test_filenames: List[str] = None,
        pre_process_workers: int = 10,
        max_size: int = -1,
        dataset_generator: Callable[
            [str, int, int], Tuple[Tensor, Tensor]
        ] = generate_dataset,
        root_dir: str = ".",
        add_channel_dimension: bool = False,
        transforms: Callable[[Tensor], Tensor] = None,
    ):
        """
        Data module for project gutenberg
        """
        super().__init__()
        self._features = features
        self._targets = targets
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        self._gutenberg_ids_train = gutenberg_ids_train
        self._gutenberg_ids_val = gutenberg_ids_val
        self._gutenberg_ids_test = gutenberg_ids_test
        self._gutenberg_range_train = gutenberg_range_train
        self._gutenberg_range_val = gutenberg_range_val
        self._gutenberg_range_test = gutenberg_range_test
        self._train_filenames = train_filenames
        self._val_filenames = val_filenames
        self._test_filenames = test_filenames
        self._pre_process_workers = pre_process_workers
        self._max_size = max_size
        self._dataset_generator = dataset_generator
        self._root_dir = root_dir
        self._add_channel_dimension = add_channel_dimension
        self._transforms = transforms

    def setup(self, stage: Optional[str] = None):
        train_files = create_full_paths(self._root_dir, self._train_filenames)
        test_files = create_full_paths(self._root_dir, self._test_filenames)
        val_files = create_full_paths(self._root_dir, self._val_filenames)

        train_ids = unify_ids(self._gutenberg_ids_train, self._gutenberg_range_train)
        val_ids = unify_ids(self._gutenberg_ids_val, self._gutenberg_range_val)
        test_ids = unify_ids(self._gutenberg_ids_test, self._gutenberg_range_test)

        self._train_dataset = SingleTextDataset(
            filenames=train_files,
            gutenberg_ids=train_ids,
            features=self._features,
            targets=self._targets,
            max_size=self._max_size,
            dataset_generator=self._dataset_generator,
            num_workers=self._pre_process_workers,
            add_channel_dimension=self._add_channel_dimension,
            transforms=self._transforms,
        )
        self._val_dataset = SingleTextDataset(
            filenames=val_files,
            gutenberg_ids=val_ids,
            features=self._features,
            targets=self._targets,
            max_size=self._max_size,
            dataset_generator=self._dataset_generator,
            num_workers=self._pre_process_workers,
            add_channel_dimension=self._add_channel_dimension,
            transforms=self._transforms,
        )
        self._test_dataset = SingleTextDataset(
            filenames=test_files,
            gutenberg_ids=test_ids,
            features=self._features,
            targets=self._targets,
            max_size=self._max_size,
            dataset_generator=self._dataset_generator,
            num_workers=self._pre_process_workers,
            add_channel_dimension=self._add_channel_dimension,
            transforms=self._transforms,
        )

        logger.info(f"Training dataset has {len(self.train_dataset)} samples.")
        logger.info(f"Validation dataset has {len(self.val_dataset)} samples.")
        logger.info(f"Test dataset has {len(self.test_dataset)} samples.")

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,  # Needed for batchnorm
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
        )


class DataModuleFromSequentialDatasets(pl.LightningDataModule):
    def __init__(
        self,
        train_features: List[Tensor],
        train_targets: List[Tensor],
        test_features: List[Tensor] = None,
        test_targets: List[Tensor] = None,
        val_features: List[Tensor] = None,
        val_targets: List[Tensor] = None,
        batch_size: int = 32,
        num_workers: int = 10,
        shuffle: bool = True,
        pin_memory: bool = True,
        pre_process_workers: int = 10,
        max_size: int = -1,
    ):
        """
        Data module for project gutenberg
        """
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        self._pre_process_workers = pre_process_workers
        self._max_size = max_size

        self._train = [train_features, train_targets]
        self._test = [test_features, test_targets]
        self._val = [val_features, val_targets]

    def setup(self, stage: Optional[str] = None):
        self._train_dataset = DatasetFromRepresentation(self._train[0], self._train[1])
        self._test_dataset = DatasetFromRepresentation(self._test[0], self._test[1])
        self._val_dataset = DatasetFromRepresentation(self._val[0], self._val[1])

        logger.info(f"Training dataset has {len(self._train_dataset)} samples.")
        logger.info(f"Validation dataset has {len(self._val_dataset)} samples.")
        logger.info(f"Test dataset has {len(self._test_dataset)} samples.")

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,  # Needed for batchnorm
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
        )


class TransformerMixin:
    def collate_fn(self, batch) -> tuple[Tensor, Tensor, list[int]]:
        # TODO: this does not make sense to me
        # The max size includes the output

        max_size = max(self._max_size, batch[0][0].shape[0])
        this_size = random.randint(1, max_size - 1)
        final_features = torch.stack([sample[0][:this_size] for sample in batch])

        # grab the first letter of the next token
        final_targets = torch.stack([sample[0][this_size][0] for sample in batch])

        final_indexes = [sample[1] for sample in batch]
        if self._as_index is True:
            return (
                final_features,
                final_targets,
                final_indexes,
            )

        return self.normalize(final_features), final_targets, final_indexes


class MambaMixin:
    def collate_fn(self, batch) -> tuple[Tensor, Tensor, list[int]]:
        # The targets are just the features shifted by 1
        # The max size includes the output
        final_features = torch.stack([sample[0][:-1] for sample in batch])

        # grab the first letter of the next token
        final_targets = torch.stack([sample[0][1:] for sample in batch])

        final_indexes = [sample[1] for sample in batch]
        if self._as_index is True:
            return (
                final_features,
                final_targets,
                final_indexes,
            )

        return final_features, final_targets, final_indexes


class SequenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        characters_per_feature: int,
        max_features: int,
        targets: int = 1,
        batch_size: int = 32,
        num_workers: int = 10,
        shuffle: bool = True,
        pin_memory: bool = False,
        gutenberg_ids_train: List[int] = None,
        gutenberg_ids_val: List[int] = None,
        gutenberg_ids_test: List[int] = None,
        gutenberg_range_train: List[int] = None,
        gutenberg_range_val: List[int] = None,
        gutenberg_range_test: List[int] = None,
        train_filenames: List[str] = None,
        val_filenames: List[str] = None,
        test_filenames: List[str] = None,
        pre_process_workers: int = 10,
        max_size: int = -1,
        dataset_generator: Callable[
            [str, int, int], Tuple[Tensor, Tensor]
        ] = generate_flat_dataset,
        root_dir: str = ".",
        add_channel_dimension: bool = False,
        transforms: Callable[[Tensor], Tensor] = None,
        repeats: int = 1,
        as_index: bool = False,
    ):
        """
        Data module for this type of transformer
        :param max_size: Truncate the loaded data to "max_size", when -1 is
        used the entire text is used.
        """
        super().__init__()
        self._characters_per_feature = characters_per_feature

        self._max_features = max_features

        self._targets = targets
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        self._gutenberg_ids_train = gutenberg_ids_train
        self._gutenberg_ids_val = gutenberg_ids_val
        self._gutenberg_ids_test = gutenberg_ids_test
        self._gutenberg_range_train = gutenberg_range_train
        self._gutenberg_range_val = gutenberg_range_val
        self._gutenberg_range_test = gutenberg_range_test
        self._train_filenames = train_filenames
        self._val_filenames = val_filenames
        self._test_filenames = test_filenames
        self._pre_process_workers = pre_process_workers
        self._max_size = max_size
        self._dataset_generator = dataset_generator
        self._root_dir = root_dir
        self._add_channel_dimension = add_channel_dimension
        self._transforms = transforms
        self._repeats = repeats
        self._as_index = as_index

    def normalize(self, data):
        return (data - 64 + 0.5) / 64.0

    def setup(self, stage: Optional[str] = None):
        train_files = create_full_paths(self._root_dir, self._train_filenames)
        test_files = create_full_paths(self._root_dir, self._test_filenames)
        val_files = create_full_paths(self._root_dir, self._val_filenames)

        train_ids = unify_ids(self._gutenberg_ids_train, self._gutenberg_range_train)
        val_ids = unify_ids(self._gutenberg_ids_val, self._gutenberg_range_val)
        test_ids = unify_ids(self._gutenberg_ids_test, self._gutenberg_range_test)

        self._train_dataset = TextTransformerDataset(
            filenames=train_files,
            gutenberg_ids=train_ids,
            characters_per_feature=self._characters_per_feature,
            max_features=self._max_features,
            targets=self._targets,
            max_size=self._max_size,
            dataset_generator=self._dataset_generator,
            num_workers=self._pre_process_workers,
            add_channel_dimension=self._add_channel_dimension,
            transforms=self._transforms,
            repeats=self._repeats,
        )
        self._val_dataset = TextTransformerDataset(
            filenames=val_files,
            gutenberg_ids=val_ids,
            characters_per_feature=self._characters_per_feature,
            max_features=self._max_features,
            targets=self._targets,
            max_size=self._max_size,
            dataset_generator=self._dataset_generator,
            num_workers=self._pre_process_workers,
            add_channel_dimension=self._add_channel_dimension,
            transforms=self._transforms,
            repeats=self._repeats,
        )
        self._test_dataset = TextTransformerDataset(
            filenames=test_files,
            gutenberg_ids=test_ids,
            characters_per_feature=self._characters_per_feature,
            max_features=self._max_features,
            targets=self._targets,
            max_size=self._max_size,
            dataset_generator=self._dataset_generator,
            num_workers=self._pre_process_workers,
            add_channel_dimension=self._add_channel_dimension,
            transforms=self._transforms,
            repeats=self._repeats,
        )

        logger.info(f"Training dataset has {len(self.train_dataset)} samples.")
        logger.info(f"Validation dataset has {len(self.val_dataset)} samples.")
        logger.info(f"Test dataset has {len(self.test_dataset)} samples.")

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,  # Needed for batchnorm
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )


class TransformerDataModule(TransformerMixin, SequenceDataModule):
    pass


class MambaDataModule(MambaMixin, SequenceDataModule):
    pass
