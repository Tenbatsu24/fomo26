from __future__ import annotations

from pathlib import Path
from typing import Optional

import blosc2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from med_adapt.registry import register_dataset


@register_dataset("OpenMind")
class OpenNeuroDataset(Dataset):
    FOLDER_NAME: str = "Dataset745_OpenMind"

    TASK_NAME: str = "OpenNeuro"
    TASK_TYPE: str = "pretrain"

    NUM_MODALITIES: int = 1
    NUM_CLASSES: Optional[int] = 1

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        seed: int = 42,
        train_fraction: float = 0.95,
        transform=None,
        inventory_file: str = "inventory.csv",
    ):
        super().__init__()

        self.root = Path(root) / self.FOLDER_NAME
        self.split = split
        self.seed = seed
        self.transform = transform

        inventory_path = self.root / inventory_file

        if not inventory_path.exists():
            raise FileNotFoundError(f"Inventory file not found: {inventory_path}")

        df = pd.read_csv(inventory_path)

        # --------------------------------------------------
        # Stratified split by modality
        # --------------------------------------------------

        train_df, val_df = train_test_split(
            df,
            train_size=train_fraction,
            random_state=seed,
            shuffle=True,
            stratify=df["modality"],
        )

        if split == "train":
            self.df = train_df.reset_index(drop=True)

        elif split in {"val", "valid", "validation"}:
            self.df = val_df.reset_index(drop=True)

        else:
            raise ValueError(
                f"Unknown split '{split}'. " f"Expected one of ['train', 'val']."
            )

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(
        self,
        image_path: str,
    ) -> Path:

        prefix = "$nnssl_preprocessed"

        if image_path.startswith(prefix):

            relative = image_path[len(prefix) :].lstrip("/")

            return self.root / relative

        return Path(image_path)

    @staticmethod
    def _percentile_zscore(
        image: np.ndarray,
        lower: float = 0.5,
        upper: float = 99.5,
    ):
        """
        Percentile clipping followed by z-score normalization.

        Uses foreground voxels if available
        (voxels > 0), otherwise falls back to all voxels.
        """

        image = image.astype(
            np.float32,
            copy=False,
        )

        values = image.reshape(-1)

        lo = np.percentile(
            values,
            lower,
        )

        hi = np.percentile(
            values,
            upper,
        )

        image = np.clip(
            image,
            lo,
            hi,
        )

        values = image.reshape(-1)

        mean = values.mean()
        std = values.std()

        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean

        return image

    def _load_image(
        self,
        path: Path,
    ):
        """
        Loads a .b2nd image.

        Expected shape:
            (C, X, Y, Z)

        Returns:
            float32 normalized image
        """

        array = blosc2.open(str(path))

        image = np.asarray(
            array,
            dtype=np.float32,
        )

        image = self._percentile_zscore(
            image,
            lower=0.5,
            upper=99.5,
        )

        return torch.from_numpy(image.transpose(0, 2, 3, 1))

    def __getitem__(
        self,
        index: int,
    ):
        row = self.df.iloc[index]
        image_path = self._resolve_image_path(row["image_path"])
        image = self._load_image(image_path)
        # image = torch.rand((1, 224, 224, 196), dtype=torch.float32)
        sample = {
            "image": image,
            "label": 0,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def save_gallery(loader, filename="gallery.png", n_examples=16):
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend

    import matplotlib.pyplot as plt

    images = []

    for batch in loader:
        x = batch["image"]  # [B, C, H, W, D]

        for i in range(x.shape[0]):
            images.append(x[i])

            if len(images) >= n_examples:
                break

        if len(images) >= n_examples:
            break

    if len(images) == 0:
        raise RuntimeError("No images found.")

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(12, 12),
    )

    for ax, img in zip(axes.flat, images):
        # img: [C, H, W, D]

        c = img.shape[0] // 2
        d = img.shape[-1] // 2

        # choose middle channel if multi-channel
        slice_2d = img[c, :, :, d].cpu().numpy()

        ax.imshow(
            slice_2d,
            cmap="gray",
            interpolation="nearest",
        )

        ax.axis("off")

    # hide unused axes
    for ax in axes.flat[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved gallery to {filename}")
