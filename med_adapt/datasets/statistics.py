from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from med_adapt.utils.config import get_logger
from med_adapt.datasets.io import ensure_3d, load_nifti

logger = get_logger(__name__)


# resolved path -> (mtime, parsed rows). Self-invalidates if the CSV is
# ever rewritten during a run.
_csv_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

# resolved path -> (mtime, statistics dict). Same idea one level up --
# avoids re-deriving the full statistics dict (per-modality medians,
# preprocessing geometry, ...) from already-parsed rows on every single
# dataset instantiation.
_statistics_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def load_or_compute_statistics(
    samples: List[Dict[str, Any]],
    task_name: str,
    folder_name: str,
    task_type: str,
    num_classes: int | None,
    cases_path: Path,
    modalities: tuple[str, ...],
) -> Dict[str, Any]:
    cases_path = Path(cases_path)

    if cases_path.exists():
        mtime = cases_path.stat().st_mtime
        resolved = str(cases_path.resolve())

        cached = _statistics_cache.get(resolved)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        logger.info(
            f"{task_name} | loading source statistics " f"from {cases_path}",
        )

        per_case_rows = read_cases_csv(cases_path)

    else:
        logger.info(f"{task_name} | computing dataset statistics")

        _, per_case_rows = compute_statistics(
            samples,
            num_modalities=len(modalities),
            modalities=modalities,
            num_classes=num_classes,
            task_name=task_name,
            folder_name=folder_name,
            task_type=task_type,
        )

        write_cases_csv(
            per_case_rows,
            cases_path,
        )

        resolved = str(cases_path.resolve())
        mtime = cases_path.stat().st_mtime

    statistics = statistics_from_case_rows(per_case_rows, modalities)
    _statistics_cache[resolved] = (mtime, statistics)

    return statistics


def read_cases_csv(
    cases_path: Path,
) -> List[Dict[str, Any]]:
    cases_path = Path(cases_path)
    resolved = str(cases_path.resolve())
    mtime = cases_path.stat().st_mtime

    cached = _csv_cache.get(resolved)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with open(cases_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Dataset cases CSV is empty: {cases_path}")

    numeric_columns = (
        "shape_h",
        "shape_w",
        "shape_d",
        "spacing_h",
        "spacing_w",
        "spacing_d",
        "mean_intensity",
        "std_intensity",
    )

    for row in rows:
        for column in numeric_columns:
            row[column] = float(row[column])

    _csv_cache[resolved] = (mtime, rows)
    return rows


def transpose_from_resolution(median_resolution) -> Tuple[int, int, int]:
    """Axis order (H, W, D) that puts the smallest dimension last (depth)."""
    depth_axis = int(np.argmin(np.asarray(median_resolution)))
    in_plane_axes = [axis for axis in range(3) if axis != depth_axis]
    return (in_plane_axes[0], in_plane_axes[1], depth_axis)


def compute_preprocessing_geometry(
    per_case_rows: List[Dict[str, Any]],
    modalities: tuple[str, ...],
    median_spacing: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """Median spacing/resolution/transpose derived from per-case metadata.

    This is now the single implementation of this math: both
    ``statistics_from_case_rows`` and ``MedicalTaskDataset``'s
    ``find_median_spacing`` / ``find_median_resolution`` / ``find_transpose``
    / ``median_resolution()`` call this rather than each keeping its own
    copy -- previously they were two independent implementations that
    could (and did) silently drift apart, e.g. one of them being derived
    from a partial/stale set of cases while the other wasn't.
    """
    if modalities:
        rows = [row for row in per_case_rows if row["modality"] in set(modalities)]
    else:
        rows = list(per_case_rows)

    if not rows:
        raise ValueError(f"No rows found for modalities {modalities!r}")

    resolutions = np.asarray(
        [
            [float(row["shape_h"]), float(row["shape_w"]), float(row["shape_d"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    spacings = np.asarray(
        [
            [
                float(row["spacing_h"]),
                float(row["spacing_w"]),
                float(row["spacing_d"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    if median_spacing is None:
        target_spacing = np.median(spacings, axis=0)
    else:
        target_spacing = np.asarray(median_spacing, dtype=np.float64)

    # Preserve physical field of view while changing spacing.
    resolutions_at_target_spacing = resolutions * spacings / target_spacing
    median_resolution_native = np.median(resolutions_at_target_spacing, axis=0)
    median_resolution_native = np.asarray(
        [int(round(v)) for v in median_resolution_native],
        dtype=np.int64,
    )

    transpose = transpose_from_resolution(median_resolution_native)

    median_resolution = tuple(int(median_resolution_native[axis]) for axis in transpose)

    return {
        "median_spacing": tuple(float(v) for v in target_spacing),
        "median_resolution_at_median_spacing": tuple(
            int(v) for v in median_resolution_native
        ),
        "transpose": transpose,
        "median_resolution": median_resolution,
    }


def statistics_from_case_rows(
    per_case_rows: List[Dict[str, Any]],
    modalities: tuple[str, ...],
) -> Dict[str, Any]:

    rows_by_modality = {
        modality: [row for row in per_case_rows if row["modality"] == modality]
        for modality in modalities
    }

    for modality, rows in rows_by_modality.items():
        if not rows:
            raise ValueError(f"No statistics rows found for modality {modality!r}")

    resolution_medians = []
    spacing_medians = []
    intensity_medians = []
    std_medians = []

    for modality in modalities:
        rows = rows_by_modality[modality]

        resolutions = np.asarray(
            [
                [
                    row["shape_h"],
                    row["shape_w"],
                    row["shape_d"],
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        spacings = np.asarray(
            [
                [
                    row["spacing_h"],
                    row["spacing_w"],
                    row["spacing_d"],
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        means = np.asarray(
            [row["mean_intensity"] for row in rows],
            dtype=np.float64,
        )

        stds = np.asarray(
            [row["std_intensity"] for row in rows],
            dtype=np.float64,
        )

        resolution_medians.append(np.median(resolutions, axis=0).tolist())
        spacing_medians.append(np.median(spacings, axis=0).tolist())
        intensity_medians.append(float(np.median(means)))
        std_medians.append(float(np.median(stds)))

    # Geometry used by preprocessing (median spacing/resolution/transpose)
    # -- computed once here via the shared implementation, and reused by
    # MedicalTaskDataset instead of being re-derived independently from
    # dataset_cases.csv a second time.
    preprocessing = compute_preprocessing_geometry(per_case_rows, modalities)

    num_samples = len({row["subject"] for row in per_case_rows})

    return {
        "num_samples": num_samples,
        "median_per_channel": intensity_medians,
        "std_per_channel": std_medians,
        # Original source-file geometry, per modality.
        "resolution": {
            "median": resolution_medians,
        },
        "spacing": {
            "median": spacing_medians,
        },
        # Geometry used by preprocessing.
        "preprocessing": preprocessing,
    }


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
            image, _, spacing = load_nifti(image_path)
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

    # Aggregated across cases via median (robust to outlier scans), not mean.
    median_per_channel = [float(np.median(v)) for v in per_channel_mean]
    std_per_channel = [float(np.median(v)) for v in per_channel_std]

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
        "median_per_channel": median_per_channel,
        "std_per_channel": std_per_channel,
        "resolution": {
            "median": np.median(resolution_array, axis=0).tolist(),
            "std": np.std(resolution_array, axis=0).tolist(),
            "min": np.min(resolution_array, axis=0).tolist(),
            "max": np.max(resolution_array, axis=0).tolist(),
        },
        "spacing": {
            "median": np.median(spacing_array, axis=0).tolist(),
            "std": np.std(spacing_array, axis=0).tolist(),
            "min": np.min(spacing_array, axis=0).tolist(),
            "max": np.max(spacing_array, axis=0).tolist(),
        },
        "spacing_per_modality": spacing_array.tolist(),
        "resolution_per_modality": resolution_array.tolist(),
    }

    return statistics, per_case_rows


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


def save_histograms(
    samples: List[Dict[str, Any]],
    num_modalities: int,
    modalities: tuple[str, ...],
    statistics: Dict[str, Any],
    task_name: str,
    histogram_path: Path,
    per_case_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Compute and save intensity / resolution histograms to *histogram_path*.

    If *per_case_rows* (e.g. already loaded from ``dataset_cases.csv``) is
    supplied, the resolution histogram is built from it directly instead
    of reloading every volume a second time just to read its shape --
    volumes then only need to be loaded once, for the intensity values.
    Passing it is optional and purely an optimization; omitting it
    reproduces the previous behavior exactly.
    """
    logger.info(f"{task_name} | saving histogram plot to {histogram_path}")

    channel_values = [[] for _ in range(num_modalities)]

    if per_case_rows is not None:
        modality_set = set(modalities)
        resolutions = np.asarray(
            [
                [row["shape_h"], row["shape_w"], row["shape_d"]]
                for row in per_case_rows
                if row["modality"] in modality_set
            ],
            dtype=np.float64,
        )
        collect_shapes = False
    else:
        resolutions = []
        collect_shapes = True

    for sample in samples:
        for channel_index, image_path in enumerate(sample["image_paths"]):
            image, _, _ = load_nifti(image_path)
            image = ensure_3d(image, image_path)

            flat = image.ravel()
            if len(flat) > 100_000:
                flat = np.random.choice(flat, size=100_000, replace=False)

            channel_values[channel_index].extend(flat.tolist())

            if collect_shapes:
                resolutions.append(image.shape)

    if collect_shapes:
        resolutions = np.asarray(resolutions).reshape(-1, 3)

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
            f"median={statistics['median_per_channel'][channel_index]:.3f}, "
            f"std={statistics['std_per_channel'][channel_index]:.3f}"
        )
        axes[0, channel_index].set_xlabel("Intensity")
        axes[0, channel_index].set_ylabel("Frequency")

    for axis_index, axis_name in enumerate(["X", "Y", "Z"]):
        axes[1, axis_index].hist(resolutions[:, axis_index], bins=30)
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


def log_statistics(
    statistics: Dict[str, Any],
    task_name: str,
    task_type: str,
    modalities: tuple[str, ...],
    num_classes: int | None,
) -> None:
    """Log source-NIfTI statistics derived from dataset_cases.csv."""

    def _fmt_vector(
        values: List[float],
    ) -> str:
        return "[" + " ".join(f"{v:.3f}" for v in values) + "]"

    def _fmt_per_modality(
        vectors: List[List[float]],
    ) -> str:
        return ", ".join(_fmt_vector(vector) for vector in vectors)

    logger.info("=" * 80)
    logger.info(task_name)
    logger.info(f"Task type: {task_type}")
    logger.info(f"Samples: {statistics['num_samples']}")
    logger.info(f"Modalities: {modalities}")

    if num_classes is not None:
        logger.info(f"Number of classes: {num_classes}")

    logger.info(
        "Median intensity per channel: "
        f"{_fmt_vector(statistics['median_per_channel'])}"
    )

    logger.info(
        "Median std per channel: " f"{_fmt_vector(statistics['std_per_channel'])}"
    )

    logger.info(
        "Median source resolution per modality: "
        f"{_fmt_per_modality(statistics['resolution']['median'])}"
    )

    logger.info(
        "Median source spacing per modality: "
        f"{_fmt_per_modality(statistics['spacing']['median'])}"
    )

    logger.info("=" * 80)

    preprocessing = statistics["preprocessing"]

    logger.info(
        "Median dataset spacing: " f"{_fmt_vector(preprocessing['median_spacing'])}"
    )

    logger.info(
        "Median resolution at median spacing "
        "(native axes): "
        f"{preprocessing['median_resolution_at_median_spacing']}"
    )

    logger.info("Transpose native -> canonical HWD: " f"{preprocessing['transpose']}")

    logger.info(
        "Median resolution after transpose: " f"{preprocessing['median_resolution']}"
    )
