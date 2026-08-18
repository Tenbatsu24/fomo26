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
    resample_spacing=None,
    resize_to=None,
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
        resample_spacing=resample_spacing,
        resize_to=resize_to,
    )
    val_ds = dataset_class(
        root=root,
        split="val",
        fold=fold,
        seed=seed,
        n_splits=n_splits,
        transform=val_transforms,
        resample_spacing=resample_spacing,
        resize_to=resize_to,
    )
    test_ds = dataset_class(
        root=root,
        split="val",
        fold=fold,
        seed=seed,
        n_splits=n_splits,
        transform=test_transforms,
        resample_spacing=resample_spacing,
        resize_to=resize_to,
    )

    train_dl = DataLoader(
        train_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
        shuffle=True,
    )
    val_dl = DataLoader(
        val_ds,
        num_workers=num_val_workers,
        batch_size=1,
        pin_memory=False,
        shuffle=False,
        persistent_workers=(num_val_workers > 0),
        drop_last=val_drop_last,
    )
    test_dl = DataLoader(
        test_ds,
        num_workers=num_val_workers,
        batch_size=1,
        pin_memory=False,
        shuffle=False,
        persistent_workers=False,
    )

    return train_dl, val_dl, test_dl


def build_pretrain_dataloaders(
    dataset_class,
    root,
    batch_size,
    num_workers,
    split_seed=42,
    sampler_seed=42,
    num_train_samples=None,
    train_transforms=None,
    val_transforms=None,
):
    from torch.utils.data import DataLoader

    from med_adapt.datasets.sampler import RandomSampler

    tr_ds = dataset_class(
        root=root,
        split="train",
        seed=split_seed,
        transform=train_transforms,
    )

    sampler = RandomSampler(tr_ds, seed=sampler_seed, num_samples=num_train_samples)

    tr_dl = DataLoader(
        tr_ds,
        batch_size=batch_size,
        sampler=sampler,
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
        num_workers=2,
        pin_memory=False,
        drop_last=False,
        persistent_workers=True,
    )

    return tr_dl, val_dl
