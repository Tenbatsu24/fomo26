"""Dataset statistics computation and caching.

All functions in this module are pure: they take explicit parameters
instead of relying on ``self``. This makes them easy to test and reuse
outside the dataset class.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from .io import ensure_3d, load_nifti

from med_adapt.utils.config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Loading / caching
# =============================================================================


def load_or_compute_statistics(
    samples: List[Dict[str, Any]],
    task_name: str,
    statistics_path: Path,
    cases_path: Path,
) -> Dict[str, Any]:
    """Load cached statistics or compute them from *samples*.

    If both *statistics_path* and *cases_path* already exist the cached
    versions are returned. Otherwise statistics are computed, written to
    disk, and returned.
    """
    if statistics_path.exists() and cases_path.exists():
        logger.info(
            f"{task_name} | loading cached statistics from {statistics_path}",
        )
        with open(statistics_path, "r") as f:
            return json.load(f)

    logger.info(f"{task_name} | computing dataset statistics")

    statistics, per_case_rows = compute_statistics(samples)

    with open(statistics_path, "w") as f:
        json.dump(statistics, f, indent=2)

    write_cases_csv(per_case_rows, cases_path)

    return statistics


# =============================================================================
# Computation
# =============================================================================


def compute_statistics(
    samples: List[Dict[str, Any]],
    num_modalities: int = 0,
    modalities: tuple[str, ...] = (),
    num_classes: int | None = None,
    task_name: str = "",
    folder_name: str = "",
    task_type: str = "",
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute dataset statistics in a single pass.

    Args:
        samples: List of sample dicts as produced by
            :meth:`MedicalTaskDataset._build_samples`.
        num_modalities: Number of image modalities.
        modalities: Modality name strings.
        num_classes: Number of segmentation classes, or ``None``.
        task_name: Human-readable task name (for the stats dict).
        folder_name: Dataset folder name (for the stats dict).
        task_type: One of ``"classification"``, ``"regression"``,
            ``"segmentation"``.

    Returns:
        ``(statistics, per_case_rows)`` where *per_case_rows* is a list of
        dicts suitable for writing to ``dataset_cases.csv``.
    """
    per_channel_mean = [[] for _ in range(num_modalities)]
    per_channel_std = [[] for _ in range(num_modalities)]

    resolutions = []
    spacing_per_modality = []
    per_case_rows = []

    for sample in samples:
        sample_shapes = []
        sample_spacings = []

        for channel_index, image_path in enumerate(sample["image_paths"]):
            image, _, spacing = load_nifti(image_path, preprocess=False)
            image = ensure_3d(image, image_path)

            per_channel_mean[channel_index].append(float(np.mean(image)))
            per_channel_std[channel_index].append(float(np.std(image)))

            sample_shapes.append(list(image.shape))
            sample_spacings.append(list(spacing))

            per_case_rows.append(
                {
                    "subject": sample["subject"],
                    "modality": modalities[channel_index],
                    "shape_h": image.shape[0],
                    "shape_w": image.shape[1],
                    "shape_d": image.shape[2],
                    "spacing_h": round(spacing[0], 4),
                    "spacing_w": round(spacing[1], 4),
                    "spacing_d": round(spacing[2], 4),
                    "mean_intensity": round(float(np.mean(image)), 6),
                    "std_intensity": round(float(np.std(image)), 6),
                }
            )

        resolutions.append(sample_shapes)
        spacing_per_modality.append(sample_spacings)

    mean_per_channel = [float(np.mean(v)) for v in per_channel_mean]
    std_per_channel = [float(np.mean(v)) for v in per_channel_std]

    resolution_array = np.asarray(resolutions)
    spacing_array = np.asarray(spacing_per_modality)

    statistics = {
        "task": task_name,
        "folder": folder_name,
        "task_type": task_type,
        "num_samples": len(samples),
        "num_modalities": num_modalities,
        "modalities": list(modalities),
        "num_classes": num_classes,
        "mean_per_channel": mean_per_channel,
        "std_per_channel": std_per_channel,
        "resolution": {
            "mean": np.mean(resolution_array, axis=0).tolist(),
            "std": np.std(resolution_array, axis=0).tolist(),
            "min": np.min(resolution_array, axis=0).tolist(),
            "max": np.max(resolution_array, axis=0).tolist(),
        },
        "spacing": {
            "mean": np.mean(spacing_array, axis=0).tolist(),
            "std": np.std(spacing_array, axis=0).tolist(),
            "min": np.min(spacing_array, axis=0).tolist(),
            "max": np.max(spacing_array, axis=0).tolist(),
            "median": np.median(spacing_array, axis=0).tolist(),
        },
        "spacing_per_modality": spacing_array.tolist(),
    }

    return statistics, per_case_rows


# =============================================================================
# CSV output
# =============================================================================


def write_cases_csv(
    rows: List[Dict[str, Any]],
    path: Path,
    task_name: str = "",
) -> None:
    """Write per-case metadata to *path* as CSV."""
    logger.info(f"{task_name} | writing per-case metadata to {path}")

    fieldnames = [
        "subject",
        "modality",
        "shape_h",
        "shape_w",
        "shape_d",
        "spacing_h",
        "spacing_w",
        "spacing_d",
        "mean_intensity",
        "std_intensity",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Histograms
# =============================================================================


def save_histograms(
    samples: List[Dict[str, Any]],
    num_modalities: int,
    modalities: tuple[str, ...],
    statistics: Dict[str, Any],
    task_name: str,
    histogram_path: Path,
) -> None:
    """Compute and save intensity / resolution histograms to *histogram_path*."""
    logger.info(f"{task_name} | saving histogram plot to {histogram_path}")

    channel_values = [[] for _ in range(num_modalities)]
    resolutions = []
    spacings = []

    for sample in samples:
        sample_shapes = []
        sample_spacings = []

        for channel_index, image_path in enumerate(sample["image_paths"]):
            image, _, spacing = load_nifti(image_path, preprocess=False)
            image = ensure_3d(image, image_path)

            flat = image.ravel()
            if len(flat) > 100_000:
                flat = np.random.choice(flat, size=100_000, replace=False)

            channel_values[channel_index].extend(flat.tolist())
            sample_shapes.append(image.shape)
            sample_spacings.append(spacing)

        resolutions.extend(np.asarray(sample_shapes).reshape(-1, 3))
        spacings.extend(np.asarray(sample_spacings).reshape(-1, 3))

    nrows = 2
    ncols = max(num_modalities, 3)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 8),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for channel_index, modality in enumerate(modalities):
        axes[0, channel_index].hist(channel_values[channel_index], bins=100)
        axes[0, channel_index].set_title(
            f"{modality}\n"
            f"mean={statistics['mean_per_channel'][channel_index]:.3f}, "
            f"std={statistics['std_per_channel'][channel_index]:.3f}"
        )
        axes[0, channel_index].set_xlabel("Intensity")
        axes[0, channel_index].set_ylabel("Frequency")

    for axis_index, axis_name in enumerate(["X", "Y", "Z"]):
        axes[1, axis_index].hist(np.asarray(resolutions)[:, axis_index], bins=30)
        axes[1, axis_index].set_title(f"Resolution {axis_name}")
        axes[1, axis_index].set_xlabel("Voxels")
        axes[1, axis_index].set_ylabel("Frequency")

    for row in range(nrows):
        for col in range(ncols):
            if row == 0 and col >= num_modalities:
                axes[row, col].axis("off")
            if row == 1 and col >= 3:
                axes[row, col].axis("off")

    fig.suptitle(f"{task_name} — Dataset Statistics", fontsize=18, fontweight="bold")
    fig.savefig(histogram_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Logging
# =============================================================================


def log_statistics(
    statistics: Dict[str, Any],
    task_name: str,
    task_type: str,
    modalities: tuple[str, ...],
    num_classes: int | None,
) -> None:
    """Print a formatted summary of *statistics* to the logger."""

    def _fmt(values: List[float]) -> str:
        return "[" + " ".join(f"{v:.3f}" for v in values) + "]"

    logger.info("=" * 80)
    logger.info(task_name)
    logger.info(f"Task type: {task_type}")
    logger.info(f"Samples: {statistics['num_samples']}")
    logger.info(f"Modalities: {modalities}")

    if num_classes is not None:
        logger.info(f"Number of classes: {num_classes}")

    logger.info(f"Mean per channel: {_fmt(statistics['mean_per_channel'])}")
    logger.info(f"Std per channel: {_fmt(statistics['std_per_channel'])}")
    logger.info(
        f"Mean resolution: {[
            _fmt(ch) for ch in statistics['resolution']['mean']
        ]}"
    )
    logger.info(
        f"Mean spacing: {[
            _fmt(ch) for ch in statistics['spacing']['mean']
        ]}"
    )
    logger.info("=" * 80)
