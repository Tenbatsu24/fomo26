from __future__ import annotations

from typing import Union

from torch.utils.data import DataLoader

from med_adapt.datasets.data import MedicalTaskDataset


def build_dataloaders(
    dataset_class: type[MedicalTaskDataset],
    root: str,
    fold: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    num_val_workers: Union[int, None] = None,
    train_transforms=None,
    val_transforms=None,
    test_transforms=None,
    n_splits: int = 5,
    val_drop_last: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    if num_val_workers is None:
        num_val_workers = min(num_workers, 2)

    train_ds = dataset_class(
        root=root,
        split="train",
        fold=fold,
        seed=seed,
        n_splits=n_splits,
        transform=train_transforms,
    )
    val_ds = dataset_class(
        root=root,
        split="val",
        fold=fold,
        seed=seed,
        n_splits=n_splits,
        transform=val_transforms,
    )
    test_ds = dataset_class(
        root=root,
        split="all",
        transform=test_transforms,
    )

    train_dl = DataLoader(
        train_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        num_workers=num_val_workers,
        batch_size=1,
        pin_memory=False,
        persistent_workers=(num_val_workers > 0),
        drop_last=val_drop_last,
    )
    test_dl = DataLoader(
        test_ds,
        num_workers=num_val_workers,
        batch_size=1,
        pin_memory=False,
        persistent_workers=False,
    )

    return train_dl, val_dl, test_dl


def build_pretrain_dataloaders(
    dataset_class,
    root,
    batch_size,
    num_workers,
    split_seed=42,
    train_transforms=None,
    val_transforms=None,
):
    from torch.utils.data import DataLoader

    num_workers = max(2, num_workers)

    tr_ds = dataset_class(
        root=root,
        split="train",
        seed=split_seed,
        transform=train_transforms,
    )

    tr_dl = DataLoader(
        tr_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_ds = dataset_class(
        root=root, split="val", seed=split_seed, transform=val_transforms
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    return tr_dl, val_dl
