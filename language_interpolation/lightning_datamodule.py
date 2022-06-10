from typing import Optional, Callable, Tuple
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from typing import List
from language_interpolation.single_text_dataset import (
    SingleTextDataset,
    generate_dataset,
    unify_ids,
    create_full_paths,
)
import logging

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
        )
        self._val_dataset = SingleTextDataset(
            filenames=val_files,
            gutenberg_ids=val_ids,
            features=self._features,
            targets=self._targets,
            max_size=self._max_size,
            dataset_generator=self._dataset_generator,
            num_workers=self._pre_process_workers,
        )
        self._test_dataset = SingleTextDataset(
            filenames=test_files,
            gutenberg_ids=test_ids,
            features=self._features,
            targets=self._targets,
            max_size=self._max_size,
            dataset_generator=self._dataset_generator,
            num_workers=self._pre_process_workers,
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
