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
from scipy.ndimage import zoom


# =============================================================================
# Loading
# =============================================================================


def load_nifti(
    path: Path,
    preprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """Load a NIfTI file in canonical RAS+ orientation.

    Args:
        path: Path to the .nii.gz file.
        preprocess: If True, clip to the 0.5–99.5 percentile range and
            z-score normalize.

    Returns:
        (data, affine, spacing) where *data* is a 3D float32 array,
        *affine* is the 4×4 affine matrix, and *spacing* is
        (pixdim1, pixdim2, pixdim3) in mm.
    """
    image = nib.load(str(path))
    image = nib.as_closest_canonical(image)

    data = np.asarray(image.get_fdata(dtype=np.float32))
    affine = image.affine
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])

    if preprocess:
        lower = np.percentile(data, 0.5)
        upper = np.percentile(data, 99.5)
        data = np.clip(data, lower, upper)

        mean = data.mean()
        std = data.std()

        if std > 0:
            data = (data - mean) / std
        else:
            data = np.zeros_like(data)

    return data, affine, spacing


# =============================================================================
# Resampling (spacing → spacing)
# =============================================================================


def resample_nifti(
    path: Path,
    target_spacing: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """Load a NIfTI and resample it to *target_spacing*.

    Intensity preprocessing is NOT applied — the caller decides when to
    clip/normalize.
    """
    img = nib.load(str(path))
    img_canon = nib.as_closest_canonical(img)
    data = np.asarray(img_canon.get_fdata(dtype=np.float32))
    affine = img_canon.affine
    spacing = tuple(float(x) for x in img_canon.header.get_zooms()[:3])

    if spacing == target_spacing:
        return data, affine, spacing

    resampled, new_affine = _resample_volume(
        data, affine, target_spacing, spacing
    )
    return resampled, new_affine, target_spacing


def resample_volume(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: Tuple[float, float, float],
    current_spacing: Tuple[float, float, float],
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a 3D volume to *target_spacing* via affine remapping.

    Args:
        data: 3D array (H, W, D).
        affine: 4×4 NIfTI affine matrix.
        target_spacing: (pixdim1, pixdim2, pixdim3) in mm.
        current_spacing: current (pixdim1, pixdim2, pixdim3) in mm.
        order: interpolation order — use ``0`` for segmentation masks
            (nearest-neighbor) and ``1`` for images (linear).

    Returns:
        (resampled_data, new_affine)
    """
    scale = tuple(old / new for old, new in zip(current_spacing, target_spacing))
    resampled = zoom(data, scale, order=order)

    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = affine[i, i] * (target_spacing[i] / current_spacing[i])

    return resampled, new_affine


# =============================================================================
# Resizing (shape → shape)
# =============================================================================


def resize_volume(
    data: np.ndarray,
    target_shape: Tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """Resize a 3D volume to *target_shape* via interpolation.

    Args:
        data: 3D array (H, W, D).
        target_shape: (H, W, D) target voxel dimensions.
        order: interpolation order — use ``0`` for segmentation masks
            (nearest-neighbor) and ``1`` for images (linear).

    Returns:
        Resized array with shape *target_shape*.
    """
    scale = tuple(t / c for t, c in zip(target_shape, data.shape))
    return zoom(data, scale, order=order)


# =============================================================================
# Helpers
# =============================================================================


def ensure_3d(array: np.ndarray, path: Path) -> np.ndarray:
    """Ensure an image is 3D."""
    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume at {path}, got shape {array.shape}"
        )
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
