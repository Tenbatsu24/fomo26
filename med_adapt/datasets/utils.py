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
    """Build train, validation, and test DataLoaders.

    Args:
        dataset_class: A subclass of :class:`MedicalTaskDataset`.
        root: Path to the data root directory.
        fold: Fold index for the split (0-based).
        seed: Random seed for reproducibility.
        batch_size: Batch size for the training loader.  Val/test use 1.
        num_workers: Number of DataLoader workers.
        num_val_workers: Number of validation / test Dataloader workers.
        train_transforms: Optional transform pipeline for training.
        val_transforms: Optional transform pipeline for validation.
        test_transforms: Optional transform pipeline for testing.
        n_splits: Total number of cross-validation folds.
        val_drop_last: Whether to drop the last incomplete validation batch.
        resample_spacing: Optional target spacing for resampling (tuple or "median").
        resize_to: Optional target volume shape (H, W, D) for resizing.

    Returns:
        ``(train_dl, val_dl, test_dl)``
    """

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
