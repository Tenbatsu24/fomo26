import csv
import json
import random

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold

from med_adapt.registry import register_dataset
from med_adapt.utils.config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible dataset splitting and visualization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_nifti(
    path: Path,
    preprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:

    image = nib.load(str(path))

    # Canonical RAS+ orientation.
    image = nib.as_closest_canonical(image)

    data = np.asarray(image.get_fdata(dtype=np.float32))

    affine = image.affine

    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])

    # ------------------------------------------------------------------
    # Image preprocessing only
    # ------------------------------------------------------------------

    if preprocess:

        # Compute robust intensity limits.
        lower = np.percentile(data, 0.5)
        upper = np.percentile(data, 99.5)

        # Clip intensities to the 0.5th-99.5th percentile range.
        data = np.clip(
            data,
            lower,
            upper,
        )

        # Z-score normalize after clipping.
        mean = data.mean()
        std = data.std()

        if std > 0:

            data = (data - mean) / std

        else:

            # Constant image.
            data = np.zeros_like(data)

    return data, affine, spacing


def resample_to_spacing(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a 3D volume to a target voxel spacing via affine remapping.

    Args:
        data: 3D array (H, W, D).
        affine: 4×4 NIfTI affine matrix.
        target_spacing: (pixdim1, pixdim2, pixdim3) in mm.

    Returns:
        (resampled_data, new_affine)
    """
    current_spacing = tuple(float(x) for x in affine[:3, 3] if False) or tuple(
        float(x)
        for x in nib.load.__class__.header.fget.__get__(
            None, type(nib.Nifti1Image())
        ).get_zooms()[:3]
    )  # fallback — we already have spacing from load_nifti

    # Compute scale factors: old_spacing / new_spacing
    scale = tuple(old / new for old, new in zip(current_spacing, target_spacing))

    resampled = zoom(data, scale, order=1)

    # Update affine: scale the pixel-size rows
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = affine[i, i] * (target_spacing[i] / current_spacing[i])

    return resampled, new_affine


def resample_nifti(
    path: Path,
    target_spacing: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """Load a NIfTI and resample it to *target_spacing* before returning.

    Intensity preprocessing is NOT applied — the caller decides when to clip/normalize.
    """
    img = nib.load(str(path))
    img_canon = nib.as_closest_canonical(img)
    data = np.asarray(img_canon.get_fdata(dtype=np.float32))
    affine = img_canon.affine
    spacing = tuple(float(x) for x in img_canon.header.get_zooms()[:3])

    if spacing == target_spacing:
        return data, affine, spacing

    resampled, new_affine = _resample_volume(data, affine, target_spacing, spacing)
    return resampled, new_affine, target_spacing


def _resample_volume(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: Tuple[float, float, float],
    current_spacing: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    scale = tuple(old / new for old, new in zip(current_spacing, target_spacing))
    resampled = zoom(data, scale, order=1)
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = affine[i, i] * (target_spacing[i] / current_spacing[i])
    return resampled, new_affine


def ensure_3d(array: np.ndarray, path: Path) -> np.ndarray:
    """Ensure an image is 3D."""
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {array.shape}")

    return array


def read_labels(path: Path) -> List[float]:
    """
    Read labels from a text file.

    Supports:
        one value per line
        whitespace-separated values
        comma-separated values
    """
    text = path.read_text().replace(",", " ")

    values = []
    for token in text.split():
        values.append(float(token))

    return values


def normalize_subject_name(name: str) -> str:
    """
    Normalize subject naming for matching.

    Handles:
        sub-01
        sub_01
    """
    return name.replace("_", "-")


# =============================================================================
# Base Dataset
# =============================================================================


class MedicalTaskDataset(Dataset):
    """
    Base class for all task datasets.

    Subclasses define:
        FOLDER_NAME
        TASK_NAME
        TASK_TYPE
        MODALITIES
        NUM_MODALITIES
        NUM_CLASSES
        LABEL_FILENAME
        MASK_FILENAME
    """

    FOLDER_NAME: str = ""
    TASK_NAME: str = ""
    TASK_TYPE: str = ""

    MODALITIES: Tuple[str, ...] = ()
    NUM_MODALITIES: int = 0
    NUM_CLASSES: Optional[int] = None

    LABEL_FILENAME: Optional[str] = None
    MASK_FILENAME: Optional[str] = None

    # Number of examples displayed in the gallery
    GALLERY_SIZE: int = 8

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        fold: Optional[int] = None,
        seed: Optional[int] = None,
        n_splits: int = 5,
        transform=None,
        return_paths: bool = False,
        resample_spacing: Optional[
            Union[Tuple[float, float, float], Literal["median"]]
        ] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.task_dir = self.root / self.FOLDER_NAME

        self.transform = transform
        self.return_paths = return_paths

        self.fold = fold
        self.seed = seed
        self.n_splits = n_splits

        if not self.task_dir.exists():
            raise FileNotFoundError(f"Task directory does not exist: {self.task_dir}")

        if self.NUM_MODALITIES != len(self.MODALITIES):
            raise ValueError(
                f"{self.__class__.__name__}: NUM_MODALITIES="
                f"{self.NUM_MODALITIES}, but "
                f"{len(self.MODALITIES)} modalities are defined."
            )

        if seed is not None:
            set_seed(seed)

        self.samples = self._build_samples()

        logger.info(
            "%s | total samples: %d",
            self.TASK_NAME,
            len(self.samples),
        )

        self.samples = self._apply_split(self.samples)

        logger.info(
            "%s | selected samples: %d",
            self.TASK_NAME,
            len(self.samples),
        )

        self.statistics_path = self.task_dir / "dataset_statistics.json"

        self.cases_path = self.task_dir / "dataset_cases.csv"

        self.histogram_path = self.task_dir / "dataset_histograms.png"

        self.gallery_path = self.task_dir / "dataset_gallery.png"

        self.statistics = self._load_or_compute_statistics()

        self._log_statistics()

        if resample_spacing == "median":
            spacing_array = np.asarray(self.statistics["spacing_per_modality"])
            median_spacing = tuple(
                float(np.median(spacing_array[:, dim])) for dim in range(3)
            )
            self.resample_spacing = median_spacing
        else:
            self.resample_spacing = resample_spacing

    # -------------------------------------------------------------------------
    # Sample discovery
    # -------------------------------------------------------------------------

    def _get_subject_directories(self) -> List[Path]:
        """
        Discover subject directories under labels/ or preprocessed/.

        Example:

            Task_1/
            ├── labels/
            │   ├── sub-01/
            │   └── sub-02/
            └── preprocessed/
                ├── sub-01/
                └── sub-02/
        """
        base_dir = self.task_dir / "preprocessed"

        if not base_dir.exists():
            raise FileNotFoundError(f"Missing preprocessed directory: {base_dir}")

        subjects = sorted(p for p in base_dir.iterdir() if p.is_dir())

        if not subjects:
            raise RuntimeError(f"No subject directories found in {base_dir}")

        return subjects

    def _find_session_directory(self, subject_dir: Path) -> Path:
        sessions = sorted(p for p in subject_dir.iterdir() if p.is_dir())

        if len(sessions) == 0:
            raise RuntimeError(f"No session directory found in {subject_dir}")

        if len(sessions) > 1:
            logger.warning(
                "Multiple sessions found in %s. Using %s.",
                subject_dir,
                sessions[0],
            )

        return sessions[0]

    def _build_samples(self) -> List[Dict[str, Any]]:
        """
        Build samples from the directory structure.

        Expected structure:

            Task_X/
            ├── labels/
            │   └── sub-01/
            │       └── ses-01/
            │           └── label.txt
            └── preprocessed/
                └── sub-01/
                    └── ses-01/
                        └── adc.nii.gz
                        └── ...

        For classification/regression:
            label.txt must contain one numeric value.

        For segmentation:
            the target is MASK_FILENAME, e.g. seg.nii.gz.
        """

        subject_dirs = self._get_subject_directories()

        samples = []

        for subject_dir in subject_dirs:

            session_dir = self._find_session_directory(subject_dir)

            image_paths = []

            for modality in self.MODALITIES:

                path = session_dir / f"{modality}.nii.gz"

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing modality '{modality}' for "
                        f"{subject_dir.name}: {path}"
                    )

                image_paths.append(path)

            # ------------------------------------------------------------------
            # Classification / Segmentation / Regression
            # ------------------------------------------------------------------

            else:

                # The labels directory mirrors the preprocessed directory.
                #
                # Example:
                #
                # Task_1/
                # ├── labels/sub-01/ses-01/label.txt
                # └── preprocessed/sub-01/ses-01/*.nii.gz
                labels_root = self.task_dir / "labels"

                matching_dirs = [
                    p
                    for p in labels_root.iterdir()
                    if p.is_dir()
                    and normalize_subject_name(p.name)
                    == normalize_subject_name(subject_dir.name)
                ]

                if len(matching_dirs) != 1:
                    raise RuntimeError(
                        f"Could not uniquely match label directory for "
                        f"{subject_dir.name}. Matches: {matching_dirs}"
                    )

                labels_subject_dir = matching_dirs[0]

                labels_session_dir = self._find_session_directory(labels_subject_dir)

                if self.LABEL_FILENAME is not None:
                    label_path = labels_session_dir / self.LABEL_FILENAME
                    label_values = read_labels(label_path)
                    if len(label_values) != 1:
                        raise ValueError(
                            f"Expected exactly one label in {label_path}, "
                            f"got {len(label_values)}"
                        )
                    target = label_values[0]
                elif self.MASK_FILENAME is not None:
                    mask_path = labels_session_dir / self.MASK_FILENAME
                    target = mask_path
                else:
                    raise FileNotFoundError(
                        f"Missing label file for {subject_dir.name}"
                    )

            samples.append(
                {
                    "subject": subject_dir.name,
                    "image_paths": image_paths,
                    "label": target,
                }
            )

        return samples

    # -------------------------------------------------------------------------
    # Splitting
    # -------------------------------------------------------------------------

    def _apply_split(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Apply a fold-based split.

        If either fold or seed is None:
            return all samples.

        fold:
            Test fold index.

        seed:
            Random state for reproducibility.
        """

        if self.fold is None or self.seed is None:
            logger.info(
                "%s | no fold split requested; using all samples",
                self.TASK_NAME,
            )
            return samples

        if not 0 <= self.fold < self.n_splits:
            raise ValueError(
                f"fold must be in [0, {self.n_splits - 1}], " f"got {self.fold}"
            )

        indices = np.arange(len(samples))

        if self.TASK_TYPE == "classification":

            labels = np.asarray([int(sample["label"]) for sample in samples])

            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            )

            splits = splitter.split(indices, labels)

        else:

            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            )

            splits = splitter.split(indices)

        for current_fold, (train_indices, test_indices) in enumerate(splits):

            if current_fold == self.fold:

                if self.split == "train":
                    selected_indices = train_indices
                else:
                    selected_indices = test_indices

                logger.info(
                    "%s | fold=%d | seed=%d | selected=%d",
                    self.TASK_NAME,
                    self.fold,
                    self.seed,
                    len(selected_indices),
                )

                return [samples[int(index)] for index in selected_indices]

        raise RuntimeError("Unable to create requested fold.")

    # -------------------------------------------------------------------------
    # Dataset API
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        sample = self.samples[index]

        images = []

        for image_path in sample["image_paths"]:

            if self.resample_spacing is not None:
                image, _, _ = resample_nifti(image_path, self.resample_spacing)
            else:
                image, _, _ = load_nifti(
                    image_path,
                    preprocess=True,
                )

            image = ensure_3d(image, image_path)

            images.append(image)

        # Shape:
        #   [C, H, W, D]
        image = np.stack(images, axis=0)

        target = sample["label"]

        if self.TASK_TYPE == "segmentation":

            target, _, _ = load_nifti(target)
            target = ensure_3d(target, sample["label"])

            target = torch.from_numpy(target.astype(np.int64))

        elif self.TASK_TYPE == "classification":

            target = torch.tensor(
                int(target),
                dtype=torch.long,
            )

        elif self.TASK_TYPE == "regression":

            target = torch.tensor(
                float(target),
                dtype=torch.float32,
            )

        else:
            raise ValueError(f"Unknown task type: {self.TASK_TYPE}")

        image = torch.from_numpy(image.astype(np.float32))

        sample_dict = {
            "image": image,
            "label": target,
            "subject": sample["subject"],
        }

        if self.return_paths:
            sample_dict["image_paths"] = sample["image_paths"]

            if self.TASK_TYPE == "segmentation":
                sample_dict["mask_path"] = sample["label"]

        if self.transform is not None:
            # print(f"transform: {self.transform}")
            # print({k: v.shape if isinstance(v, torch.Tensor) else [] for k, v in sample_dict.items()})
            sample_dict = self.transform(sample_dict)
            # print({k: v.shape if isinstance(v, torch.Tensor) else [] for k, v in sample_dict.items()})
        # else:
        #     print(f"No transform: {self.transform}")

        return sample_dict

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def _load_or_compute_statistics(self) -> Dict[str, Any]:

        if self.statistics_path.exists() and self.cases_path.exists():

            logger.info(
                "%s | loading cached statistics from %s",
                self.TASK_NAME,
                self.statistics_path,
            )

            with open(self.statistics_path, "r") as f:
                return json.load(f)

        logger.info(
            "%s | computing dataset statistics",
            self.TASK_NAME,
        )

        statistics, per_case_rows = self._compute_statistics()

        with open(self.statistics_path, "w") as f:
            json.dump(
                statistics,
                f,
                indent=2,
            )

        self._write_cases_csv(per_case_rows)

        self._save_histograms(statistics)

        return statistics

    def _compute_statistics(
        self,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Compute dataset statistics in a single pass.

        Returns:
            (statistics, per_case_rows) where per_case_rows is a list of
            dicts suitable for writing to dataset_cases.csv.
        """
        per_channel_mean = [[] for _ in range(self.NUM_MODALITIES)]
        per_channel_std = [[] for _ in range(self.NUM_MODALITIES)]

        resolutions = []
        spacing_per_modality = []

        per_case_rows = []

        for sample in self.samples:

            sample_shapes = []
            sample_spacings = []

            for channel_index, image_path in enumerate(sample["image_paths"]):

                image, _, spacing = load_nifti(
                    image_path,
                    preprocess=True,
                )

                image = ensure_3d(image, image_path)

                per_channel_mean[channel_index].append(float(np.mean(image)))

                per_channel_std[channel_index].append(float(np.std(image)))

                sample_shapes.append(list(image.shape))
                sample_spacings.append(list(spacing))

                per_case_rows.append(
                    {
                        "subject": sample["subject"],
                        "modality": self.MODALITIES[channel_index],
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

        # Statistics are per-sample because instance normalization is used.
        mean_per_channel = [float(np.mean(values)) for values in per_channel_mean]

        std_per_channel = [float(np.mean(values)) for values in per_channel_std]

        resolution_array = np.asarray(resolutions)

        spacing_array = np.asarray(spacing_per_modality)

        statistics = {
            "task": self.TASK_NAME,
            "folder": self.FOLDER_NAME,
            "task_type": self.TASK_TYPE,
            "num_samples": len(self.samples),
            "num_modalities": self.NUM_MODALITIES,
            "modalities": list(self.MODALITIES),
            "num_classes": self.NUM_CLASSES,
            "mean_per_channel": mean_per_channel,
            "std_per_channel": std_per_channel,
            "resolution": {
                "mean": np.mean(
                    resolution_array,
                    axis=0,
                ).tolist(),
                "std": np.std(
                    resolution_array,
                    axis=0,
                ).tolist(),
                "min": np.min(
                    resolution_array,
                    axis=0,
                ).tolist(),
                "max": np.max(
                    resolution_array,
                    axis=0,
                ).tolist(),
            },
            "spacing": {
                "mean": np.mean(
                    spacing_array,
                    axis=0,
                ).tolist(),
                "std": np.std(
                    spacing_array,
                    axis=0,
                ).tolist(),
                "min": np.min(
                    spacing_array,
                    axis=0,
                ).tolist(),
                "max": np.max(
                    spacing_array,
                    axis=0,
                ).tolist(),
                "median": np.median(
                    spacing_array,
                    axis=0,
                ).tolist(),
            },
            "spacing_per_modality": spacing_array.tolist(),
        }

        return statistics, per_case_rows

    def _write_cases_csv(self, rows: List[Dict[str, Any]]) -> None:
        """Write per-case metadata to dataset_cases.csv."""
        logger.info(
            "%s | writing per-case metadata to %s",
            self.TASK_NAME,
            self.cases_path,
        )

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

        with open(self.cases_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _log_statistics(self) -> None:

        logger.info("=" * 80)
        logger.info("%s", self.TASK_NAME)
        logger.info("Task type: %s", self.TASK_TYPE)
        logger.info("Samples: %d", self.statistics["num_samples"])
        logger.info("Modalities: %s", self.MODALITIES)

        if self.NUM_CLASSES is not None:
            logger.info("Number of classes: %d", self.NUM_CLASSES)

        logger.info(
            "Mean per channel: %s",
            self.statistics["mean_per_channel"],
        )

        logger.info(
            "Std per channel: %s",
            self.statistics["std_per_channel"],
        )

        logger.info(
            "Mean resolution: %s",
            self.statistics["resolution"]["mean"],
        )

        logger.info(
            "Mean spacing: %s",
            self.statistics["spacing"]["mean"],
        )

        logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------------------

    def _save_histograms(
        self,
        statistics: Dict[str, Any],
    ) -> None:

        logger.info(
            "%s | saving histogram plot to %s",
            self.TASK_NAME,
            self.histogram_path,
        )

        # We recompute values for plotting.
        channel_values = [[] for _ in range(self.NUM_MODALITIES)]

        resolutions = []
        spacings = []

        for sample in self.samples:

            sample_shapes = []
            sample_spacings = []

            for channel_index, image_path in enumerate(sample["image_paths"]):
                image, _, spacing = load_nifti(
                    image_path,
                    preprocess=True,
                )

                image = ensure_3d(image, image_path)

                # Subsample for memory efficiency.
                flat = image.ravel()

                if len(flat) > 100_000:
                    flat = np.random.choice(
                        flat,
                        size=100_000,
                        replace=False,
                    )

                channel_values[channel_index].extend(flat.tolist())

                sample_shapes.append(image.shape)
                sample_spacings.append(spacing)

            resolutions.extend(np.asarray(sample_shapes).reshape(-1, 3))

            spacings.extend(np.asarray(sample_spacings).reshape(-1, 3))

        nrows = 2
        ncols = max(
            self.NUM_MODALITIES,
            3,
        )

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5 * ncols, 8),
            constrained_layout=True,
        )

        axes = np.atleast_2d(axes)

        for channel_index, modality in enumerate(self.MODALITIES):

            axes[0, channel_index].hist(
                channel_values[channel_index],
                bins=100,
            )

            axes[0, channel_index].set_title(
                f"{modality}\n"
                f"mean={statistics['mean_per_channel'][channel_index]:.3f}, "
                f"std={statistics['std_per_channel'][channel_index]:.3f}"
            )

            axes[0, channel_index].set_xlabel("Intensity")
            axes[0, channel_index].set_ylabel("Frequency")

        for axis_index, axis_name in enumerate(["X", "Y", "Z"]):

            axes[1, axis_index].hist(
                np.asarray(resolutions)[:, axis_index],
                bins=30,
            )

            axes[1, axis_index].set_title(f"Resolution {axis_name}")

            axes[1, axis_index].set_xlabel("Voxels")
            axes[1, axis_index].set_ylabel("Frequency")

        # If there are spare axes, hide them.
        for row in range(nrows):
            for col in range(ncols):

                if row == 0 and col >= self.NUM_MODALITIES:
                    axes[row, col].axis("off")

                if row == 1 and col >= 3:
                    axes[row, col].axis("off")

        fig.suptitle(
            f"{self.TASK_NAME} — Dataset Statistics",
            fontsize=18,
            fontweight="bold",
        )

        fig.savefig(
            self.histogram_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    # -------------------------------------------------------------------------
    # Gallery
    # -------------------------------------------------------------------------

    def create_gallery(self) -> None:
        """Create and save an example gallery image.

        This is intentionally *not* called automatically in ``__init__``
        because loading and rendering every sample is computationally heavy.
        Call this method explicitly when you need the gallery.
        """
        if len(self.samples) == 0:
            return

        logger.info(
            "%s | creating example gallery at %s",
            self.TASK_NAME,
            self.gallery_path,
        )

        n_examples = min(
            self.GALLERY_SIZE,
            len(self.samples),
        )

        indices = np.linspace(
            0,
            len(self.samples) - 1,
            n_examples,
            dtype=int,
        )

        ncols = self.NUM_MODALITIES
        nrows = n_examples

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                4 * ncols,
                3.5 * nrows,
            ),
            squeeze=False,
            constrained_layout=True,
        )

        for row, index in enumerate(indices):

            sample = self.samples[index]
            loaded_images = []

            for image_path in sample["image_paths"]:

                image, _, _ = load_nifti(
                    image_path,
                    preprocess=True,
                )

                image = ensure_3d(
                    image,
                    image_path,
                )

                loaded_images.append(image)

            mask = None

            if self.TASK_TYPE == "segmentation":

                mask, _, _ = load_nifti(
                    sample["label"],
                    preprocess=False,
                )

                mask = ensure_3d(
                    mask,
                    sample["label"],
                )

                mask_sum_per_depth = np.sum(
                    mask > 0,
                    axis=(0, 1),
                )

                if np.max(mask_sum_per_depth) > 0:
                    slice_index = int(np.argmax(mask_sum_per_depth))
                else:
                    slice_index = mask.shape[-1] // 2

            else:
                slice_index = loaded_images[0].shape[-1] // 2

            for col, (modality, image) in enumerate(
                zip(
                    self.MODALITIES,
                    loaded_images,
                )
            ):

                ax = axes[row, col]

                slice_image = image[..., slice_index]

                ax.imshow(
                    slice_image.T,
                    cmap="gray",
                    origin="lower",
                )

                if mask is not None:

                    slice_mask = mask[..., slice_index]

                    masked = np.ma.masked_where(
                        slice_mask == 0,
                        slice_mask,
                    )

                    ax.imshow(
                        masked.T,
                        alpha=0.45,
                        origin="lower",
                    )

                ax.set_title(f"{sample['subject']}\n" f"{modality}")

                ax.axis("off")

            if self.TASK_TYPE != "segmentation":

                target = sample["label"]

                target_text = (
                    f"class={int(target)}"
                    if self.TASK_TYPE == "classification"
                    else f"value={float(target):.2f}"
                )

                axes[row, 0].set_title(f"{sample['subject']}\n" f"{target_text}")

        fig.suptitle(
            f"{self.TASK_NAME} — Example Gallery",
            fontsize=18,
            fontweight="bold",
        )

        fig.savefig(
            self.gallery_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)


@register_dataset("CLS002_FOMO26_Infarct")
class Task1InfarctClassification(MedicalTaskDataset):

    FOLDER_NAME = "Task_1"
    TASK_NAME = "Infarct Classification"
    TASK_TYPE = "classification"

    MODALITIES = (
        "adc",
        "dwi_b1000",
        "flair",
    )

    NUM_MODALITIES = 3

    NUM_CLASSES = 2

    LABEL_FILENAME = "label.txt"

    MASK_FILENAME = None


@register_dataset("SEG009_FOMO26_Meningioma")
class Task2MeningiomaSegmentation(MedicalTaskDataset):

    FOLDER_NAME = "Task_2"
    TASK_NAME = "Meningioma Segmentation"
    TASK_TYPE = "segmentation"

    MODALITIES = (
        "dwi_b1000",
        "flair",
    )

    NUM_MODALITIES = 2

    NUM_CLASSES = 2

    LABEL_FILENAME = None

    MASK_FILENAME = "seg.nii.gz"


@register_dataset("REGR002_FOMO26_BrainAge")
class Task3BrainAgeRegression(MedicalTaskDataset):

    FOLDER_NAME = "Task_3"
    TASK_NAME = "Brain Age Regression"
    TASK_TYPE = "regression"

    MODALITIES = ("t1w",)

    NUM_MODALITIES = 1

    NUM_CLASSES = None

    LABEL_FILENAME = "labels.txt"

    MASK_FILENAME = None


@register_dataset("SEG010_FOMO26_TrigeminalNeuralgia")
class Task4TrigeminalNeuralgiaSegmentation(MedicalTaskDataset):

    FOLDER_NAME = "Task_4"
    TASK_NAME = "Trigeminal Neuralgia Segmentation"
    TASK_TYPE = "segmentation"

    MODALITIES = ("t2w",)

    NUM_MODALITIES = 1

    NUM_CLASSES = 3

    LABEL_FILENAME = None

    MASK_FILENAME = "seg.nii.gz"


@register_dataset("CLS003_FOMO26_Polymicrogyria")
class Task5PolymicrogyriaClassification(MedicalTaskDataset):

    FOLDER_NAME = "Task_5"
    TASK_NAME = "Polymicrogyria Classification"
    TASK_TYPE = "classification"

    MODALITIES = ("t1",)

    NUM_MODALITIES = 1

    NUM_CLASSES = 2

    LABEL_FILENAME = "labels.txt"

    MASK_FILENAME = None


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Inspect medical imaging datasets by computing/logging "
            "statistics and generating example galleries."
        )
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Fold index. If both --fold and --seed are specified, "
            "only that fold is used."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for fold splitting.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds.",
    )

    args = parser.parse_args()

    root = Path("../../data")

    datasets = [
        Task1InfarctClassification,
        # Task2MeningiomaSegmentation,
        # Task3BrainAgeRegression,
        # Task4TrigeminalNeuralgiaSegmentation,
        # Task5PolymicrogyriaClassification,
    ]

    logger.info("=" * 100)
    logger.info("DATASET INSPECTION")
    logger.info("Root: %s", root)
    logger.info("=" * 100)

    for dataset_cls in datasets:

        logger.info("")
        logger.info("=" * 100)
        logger.info(
            "Inspecting %s",
            dataset_cls.TASK_NAME,
        )
        logger.info("=" * 100)

        try:

            dataset = dataset_cls(
                root=root,
                fold=args.fold,
                seed=args.seed,
                n_splits=args.n_splits,
            )

            logger.info(
                "Finished inspection of %s",
                dataset_cls.TASK_NAME,
            )

            logger.info(
                "Statistics saved to: %s",
                dataset.statistics_path,
            )

            logger.info(
                "Per-case metadata saved to: %s",
                dataset.cases_path,
            )

            logger.info(
                "Histogram saved to: %s",
                dataset.histogram_path,
            )

        except Exception:

            logger.exception(
                "Failed to inspect %s",
                dataset_cls.TASK_NAME,
            )

    logger.info("")
    logger.info("=" * 100)
    logger.info("DATASET INSPECTION COMPLETE")
    logger.info("=" * 100)
