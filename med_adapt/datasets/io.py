"""NIfTI I/O and spatial resampling utilities.

Functions in this module operate on raw numpy arrays and 4×4 affine matrices.
They are intentionally pure — no torch, no dataset logic — so they can be
tested and reused independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib

from scipy.ndimage import zoom, binary_fill_holes


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
    nonzero_mask = image != 0
    mask = binary_fill_holes(nonzero_mask)

    values = image[mask].reshape(-1)

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

    values = image[mask].reshape(-1)

    mean = values.mean()
    std = values.std()

    if std > 1e-4:
        image = (image - mean) / std
    else:
        image = image - mean

    return image


def load_nifti(
    path: Path, is_mask=False
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    image = nib.load(str(path))
    image = nib.as_closest_canonical(image)

    data = np.asarray(image.get_fdata(dtype=np.float32))
    affine = image.affine
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])

    if not is_mask:
        data = _percentile_zscore(data)

    return data, affine, spacing


# =============================================================================
# Resampling (spacing → spacing)
# =============================================================================


def resample_nifti(
    path: Path,
    target_spacing: Tuple[float, float, float],
    is_mask=False,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    data, affine, spacing = load_nifti(path, is_mask=is_mask)

    if spacing == target_spacing:
        return data, affine, spacing

    resampled, new_affine = resample_volume(
        data, affine, target_spacing, spacing, 0 if is_mask else 3
    )
    return resampled, new_affine, target_spacing


def resample_volume(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: Tuple[float, float, float],
    current_spacing: Tuple[float, float, float],
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = tuple(old / new for old, new in zip(current_spacing, target_spacing))
    resampled = zoom(data, scale, order=order)

    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = affine[i, i] * (target_spacing[i] / current_spacing[i])

    return resampled, new_affine


def resize_volume(
    data: np.ndarray, target_shape: Tuple[int, int, int], is_mask: bool = False
) -> np.ndarray:
    scale = tuple(t / c for t, c in zip(target_shape, data.shape))
    return zoom(data, scale, order=0 if is_mask else 3)


def ensure_3d(array: np.ndarray, path: Path) -> np.ndarray:
    """Ensure an image is 3D."""
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {array.shape}")
    return array


def read_labels(path: Path) -> list[float]:
    """Read labels from a text file.

    Supports one value per line, whitespace-separated, or comma-separated.
    """
    text = path.read_text().replace(",", " ")
    return [float(token) for token in text.split()]


def normalize_subject_name(name: str) -> str:
    """Normalize subject naming for matching (handles ``_`` and ``-``)."""
    return name.replace("_", "-")
