import logging

from typing import Literal, Optional

import lightning as pl

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from fomo26.dataset import MedicalTaskDataset

LOGGER = logging.getLogger(__name__)


class SegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        dataset_class: MedicalTaskDataset,
        root: str,
        fold: int,
        seed: int,
        n_splits: int = 5,
        predict_files: Optional[list] = None,
        test_files: Optional[list] = None,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        predict_transforms: Optional[Compose] = None,
    ):
        super().__init__()
        if predict_files is None:
            predict_files = []
        if test_files is None:
            test_files = []

        self.batch_size = batch_size
        self.dataset_class = dataset_class
        self.root = root
        self.fold = fold
        self.seed = seed
        self.n_splits = n_splits
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.predict_transforms = predict_transforms
        self.num_workers = num_workers
        self.predict_files = predict_files
        self.test_files = test_files

        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
            self.setup_predict()

    def setup_fit(self):
        self.train_dataset = self.dataset_class(
            root=self.root,
            split="train",
            fold=self.fold,
            seed=self.seed,
            n_splits=self.n_splits,
            transform=self.train_transforms,
        )
        self.val_dataset = self.dataset_class(
            root=self.root,
            split="val",
            fold=self.fold,
            seed=self.seed,
            n_splits=self.n_splits,
            transform=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = self.dataset_class(
            root=self.root,
            split="val",
            fold=self.fold,
            seed=self.seed,
            n_splits=self.n_splits,
            transform=self.test_transforms,
        )

    def setup_predict(self):
        self.predict_dataset = self.dataset_class(
            root=self.root,
            split="val",
            fold=self.fold,
            seed=self.seed,
            n_splits=self.n_splits,
            transform=self.predict_transforms,
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
        dataset_class,
        root: str,
        fold: int,
        seed: int,
        n_splits: int = 5,
        predict_files: Optional[list] = None,
        test_files: Optional[list] = None,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        predict_transforms: Optional[Compose] = None,
        use_random_datasampler: Optional[bool] = True,
    ):
        super().__init__()
        if predict_files is None:
            predict_files = []
        if test_files is None:
            test_files = []

        self.batch_size = batch_size
        self.dataset_class = dataset_class
        self.root = root
        self.fold = fold
        self.seed = seed
        self.n_splits = n_splits
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.predict_transforms = predict_transforms
        self.num_workers = num_workers
        self.predict_files = predict_files
        self.test_files = test_files
        self.use_random_datasampler = use_random_datasampler
        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
            self.setup_predict()

    def setup_fit(self):
        self.train_dataset = self.dataset_class(
            root=self.root,
            split="train",
            fold=self.fold,
            seed=self.seed,
            n_splits=self.n_splits,
            transform=self.train_transforms,
        )
        self.val_dataset = self.dataset_class(
            root=self.root,
            split="val",
            fold=self.fold,
            seed=self.seed,
            n_splits=self.n_splits,
            transform=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = self.dataset_class(
            root=self.root,
            split="val",
            fold=self.fold,
            seed=self.seed,
            n_splits=self.n_splits,
            transform=self.test_transforms,
        )

    def setup_predict(self):
        self.predict_dataset = self.dataset_class(
            root=self.root,
            split="val",
            fold=self.fold,
            seed=self.seed,
            n_splits=self.n_splits,
            transform=self.predict_transforms,
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
            drop_last=False,
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
