"""Tests for med_adapt.datasets.utils."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from med_adapt.datasets.utils import build_dataloaders
from med_adapt.datasets.data import MedicalTaskDataset


class FakeDataset(MedicalTaskDataset):
    FOLDER_NAME = "Task_fake"
    TASK_NAME = "Fake Task"
    TASK_TYPE = "classification"
    MODALITIES = ("t1",)
    NUM_MODALITIES = 1
    NUM_CLASSES = 2
    LABEL_FILENAME = "label.txt"
    MASK_FILENAME = None

    def __init__(self, *args, **kwargs):
        # Bypass parent __init__ to avoid file system checks
        self.transform = kwargs.pop("transform", None)
        self.samples = [{"subject": "sub-01", "image_paths": [], "label": 0}] * 4

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = torch.zeros(1, 8, 8, 8, dtype=torch.float32)
        label = torch.tensor(int(sample["label"]), dtype=torch.long)
        result = {"image": image, "label": label, "subject": sample["subject"]}
        if self.transform:
            result = self.transform(result)
        return result


class TestBuildDataloaders:
    def test_returns_three_dataloaders(self):
        train_dl, val_dl, test_dl = build_dataloaders(
            dataset_class=FakeDataset,
            root="/tmp/fake",
            fold=0,
            seed=42,
            batch_size=2,
            num_workers=0,
        )
        assert train_dl is not None
        assert val_dl is not None
        assert test_dl is not None

    def test_train_batch_size(self):
        train_dl, val_dl, test_dl = build_dataloaders(
            dataset_class=FakeDataset,
            root="/tmp/fake",
            fold=0,
            seed=42,
            batch_size=4,
            num_workers=0,
        )
        assert train_dl.batch_size == 4
        assert val_dl.batch_size == 1
        assert test_dl.batch_size == 1

    def test_train_shuffle(self):
        train_dl, val_dl, test_dl = build_dataloaders(
            dataset_class=FakeDataset,
            root="/tmp/fake",
            fold=0,
            seed=42,
            batch_size=2,
            num_workers=0,
        )
        # DataLoader in newer PyTorch doesn't expose .shuffle; check sampler instead
        from torch.utils.data import SequentialSampler, RandomSampler

        assert isinstance(train_dl.sampler, RandomSampler)
        assert isinstance(val_dl.sampler, SequentialSampler)

    def test_dataloader_iteration(self):
        train_dl, val_dl, test_dl = build_dataloaders(
            dataset_class=FakeDataset,
            root="/tmp/fake",
            fold=0,
            seed=42,
            batch_size=2,
            num_workers=0,
        )
        batch = next(iter(train_dl))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape[0] == 2
