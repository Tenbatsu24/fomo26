from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import blosc2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from sklearn.model_selection import train_test_split

from med_adapt.registry import register_dataset
from med_adapt.datasets.io import _percentile_zscore


@register_dataset("OpenMind")
class OpenNeuroDataset(IterableDataset):

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

        image = _percentile_zscore(
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
        image = self._load_image(image_path)  # to be used when actually training
        # image = torch.randn((1, 224, 224, 196), dtype=torch.float32)
        sample = {"image": image, "label": 0, "index": index}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __iter__(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        rng = random.Random(self.seed)

        while True:
            indices = list(range(len(self)))

            if self.split == "train":
                rng.shuffle(indices)

            for idx in indices[global_worker_id::global_workers]:
                yield self[idx]


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
