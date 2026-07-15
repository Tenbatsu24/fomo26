import logging

from typing import Literal, Optional

import lightning as pl
import torch.distributed as dist

from torchvision.transforms import Compose
from torch.utils.data import DataLoader, RandomSampler
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper

from fomo26.modules.datasets.train_dataset import (
    ClsRegDataset,
    ClsRegTestDataset,
    SegDataset,
    SegTestDataset,
    SingleSubjectPredictDataset,
)


class SegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        test_samples=None,
        predict_samples=None,
        predict_transforms: Optional[Compose] = None,
        train_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
    ):
        super().__init__()
        if predict_samples is None:
            predict_samples = []
        if test_samples is None:
            test_samples = []
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.test_samples = test_samples
        self.val_split = val_split
        self.predict_samples = predict_samples
        self.predict_transforms = predict_transforms

        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
            self.setup_predict()

    def setup_fit(self):
        self.train_dataset = SegDataset(
            self.train_split,
            transforms=self.train_transforms,
        )

        self.val_dataset = SegDataset(
            self.val_split,
            transforms=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = SegTestDataset(
            self.test_samples,
            transforms=self.test_transforms,
        )

    def setup_predict(self):
        self.predict_dataset = SingleSubjectPredictDataset(
            self.predict_samples,
            transforms=self.predict_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=1,
        )


class ClsRegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        predict_transforms: Optional[Compose] = None,
        test_samples=None,
        predict_samples=None,
        use_random_datasampler: Optional[bool] = True,
    ):
        super().__init__()
        if predict_samples is None:
            predict_samples = []
        if test_samples is None:
            test_samples = []
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_samples = test_samples
        self.use_random_datasampler = use_random_datasampler
        self.predict_samples = predict_samples
        self.predict_transforms = predict_transforms
        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
            self.setup_predict()

    def setup_fit(self):
        self.train_dataset = ClsRegDataset(
            self.train_split,
            transforms=self.train_transforms,
        )

        self.val_dataset = ClsRegDataset(
            self.val_split,
            transforms=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = ClsRegTestDataset(
            self.test_samples,
            transforms=self.test_transforms,
        )

    def setup_predict(self):
        self.predict_dataset = SingleSubjectPredictDataset(
            self.predict_samples,
            transforms=self.predict_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=1,
            batch_size=1,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=1,
        )
