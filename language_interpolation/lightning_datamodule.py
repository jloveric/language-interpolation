from typing import Optional
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from typing import List
from language_interpolation.single_text_dataset import (
    SingleTextDataset,
    generate_dataset,
    dataset_centered,
    unify_ids,
)


class GutenbergDataModule(pl.LightningDataModule):
    def __init__(
        self,
        features: int,
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
        pre_process_workers: int = 10,
        max_size: int = -1,
    ):
        """
        Data module for project gutenberg
        """
        super().__init__()
        self._features = features
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
        self._pre_process_workers = pre_process_workers
        self._max_size = max_size

    def setup(self, stage: Optional[str] = None):

        full_path = None
        if self.cfg.filenames is not None:
            full_path = [f"{self.root_dir}/{path}" for path in self.cfg.filenames]

        if self.cfg.data.type == "sequence":
            dataset_generator = generate_dataset
        elif self.cfg.data.type == "centered":
            dataset_generator = dataset_centered
        else:
            raise ValueError(
                f"data.type must be centered or sequence. recieved {self.cfg.data.type}"
            )

        train_ids = unify_ids(self._gutenberg_ids_train, self._gutenberg_range_train)
        val_ids = unify_ids(self._gutenberg_ids_val, self._gutenberg_range_val)
        test_ids = unify_ids(self._gutenberg_ids_test, self._gutenberg_range_test)

        self._train_dataset = SingleTextDataset(
            filenames=full_path,
            gutenberg_ids=train_ids,
            features=self._features,
            max_size=self._max_size,
            dataset_generator=dataset_generator,
            num_workers=self._pre_process_workers,
        )
        self._val_dataset = SingleTextDataset(
            filenames=full_path,
            gutenberg_ids=val_ids,
            features=self._features,
            max_size=self._max_size,
            dataset_generator=dataset_generator,
            num_workers=self._pre_process_workers,
        )
        self._test_dataset = SingleTextDataset(
            filenames=full_path,
            gutenberg_ids=test_ids,
            features=self._features,
            max_size=self._max_size,
            dataset_generator=dataset_generator,
            num_workers=self._pre_process_workers,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
        )
