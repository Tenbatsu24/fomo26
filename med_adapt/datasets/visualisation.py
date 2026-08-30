"""Dataset gallery visualisation.

All functions in this module are pure: they take explicit parameters
instead of relying on ``self``. This makes them easy to test and reuse
outside the dataset class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from med_adapt.utils.config import get_logger
from med_adapt.datasets.io import ensure_3d, load_nifti

logger = get_logger(__name__)


def create_gallery(
    samples: List[Dict[str, Any]],
    num_modalities: int,
    modalities: tuple[str, ...],
    task_type: str,
    gallery_path: Path,
    gallery_size: int = 8,
) -> None:
    """Create and save an example gallery image.

    Args:
        samples: List of sample dicts as produced by
            :meth:`MedicalTaskDataset._build_samples`.
        num_modalities: Number of image modalities.
        modalities: Modality name strings.
        task_type: One of ``"classification"``, ``"regression"``,
            ``"segmentation"``.
        gallery_path: Where to save the PNG.
        gallery_size: Maximum number of examples to display.
    """
    if len(samples) == 0:
        return

    logger.info(f"creating example gallery at {gallery_path}")

    n_examples = min(gallery_size, len(samples))

    indices = np.linspace(0, len(samples) - 1, n_examples, dtype=int)

    ncols = num_modalities
    nrows = n_examples

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, index in enumerate(indices):
        sample = samples[index]
        loaded_images = []

        for image_path in sample["image_paths"]:
            image, _, _ = load_nifti(image_path)
            image = ensure_3d(image, image_path)
            loaded_images.append(image)

        mask = None
        if task_type == "segmentation":
            mask, _, _ = load_nifti(sample["label"], is_mask=True)
            mask = ensure_3d(mask, sample["label"])

            mask_sum_per_depth = np.sum(mask > 0, axis=(0, 1))
            slice_index = (
                int(np.argmax(mask_sum_per_depth))
                if np.max(mask_sum_per_depth) > 0
                else mask.shape[-1] // 2
            )
        else:
            slice_index = loaded_images[0].shape[-1] // 2

        for col, (modality, image) in enumerate(zip(modalities, loaded_images)):
            ax = axes[row, col]
            slice_image = image[..., slice_index]
            ax.imshow(slice_image.T, cmap="gray", origin="lower")

            if mask is not None:
                slice_mask = mask[..., slice_index]
                masked = np.ma.masked_where(slice_mask == 0, slice_mask)
                ax.imshow(masked.T, alpha=0.45, origin="lower")

            ax.set_title(f"{sample['subject']}\n{modality}")
            ax.axis("off")

        if task_type != "segmentation":
            target = sample["label"]
            if task_type == "classification":
                target_text = f"class={int(target)}"
            else:
                target_text = f"value={float(target):.2f}"
            axes[row, 0].set_title(f"{sample['subject']}\n{target_text}")

    fig.suptitle("Example Gallery", fontsize=18, fontweight="bold")
    fig.savefig(gallery_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
