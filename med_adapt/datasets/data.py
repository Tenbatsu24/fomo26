import json
import random

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import torch
import numpy as np
import torch.distributed as dist

from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from sklearn.model_selection import KFold, StratifiedKFold

from med_adapt.utils.config import get_logger

from .io import (
    ensure_3d,
    load_nifti,
    normalize_subject_name,
    read_labels,
    resample_nifti,
    resize_volume,
)
from .statistics import (
    load_or_compute_statistics,
    log_statistics,
)
from .visualisation import create_gallery
from ..utils import get_data_path

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_AGE_BUCKET_EDGES: Tuple[int, ...] = (20, 30, 40, 50, 60, 70, 80, 90)


def _classification_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(sample["label"]) for sample in samples])


def _age_bucket_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Bucket brain-age regression targets for stratified splitting.

    Buckets: <20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, >=90.
    """
    ages = np.asarray([float(sample["label"]) for sample in samples])
    return np.digitize(ages, _AGE_BUCKET_EDGES)


def _segmentation_positivity_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Label each sample by whether its mask contains any foreground voxel.

    Used so the train/val split keeps a roughly similar ratio of
    positive/negative scans in both folds, instead of letting a random
    split concentrate positives in one side.
    """
    labels = []
    for sample in samples:
        mask, _, _ = load_nifti(sample["label"], is_mask=True)
        mask = ensure_3d(mask, sample["label"])
        labels.append(int(np.any(mask > 0)))
    return np.asarray(labels)


# =============================================================================
# Base Dataset
# =============================================================================


class MedicalTaskDataset(IterableDataset):
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
        resize_to: Optional[Union[Tuple[int, int, int], Literal["median"]]] = None,
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

        samples = self._build_samples()

        logger.info(
            f"{self.TASK_NAME} | total samples: {len(samples)}",
        )

        self.samples = self._apply_split(samples)

        logger.info(
            f"{self.TASK_NAME} | selected samples: {len(self.samples)}",
        )

        self.statistics_path = self.task_dir / "dataset_statistics.json"
        self.cases_path = self.task_dir / "dataset_cases.csv"
        self.histogram_path = self.task_dir / "dataset_histograms.png"
        self.gallery_path = self.task_dir / "dataset_gallery.png"

        self.statistics = load_or_compute_statistics(
            self.samples,
            self.TASK_NAME,
            self.FOLDER_NAME,
            self.TASK_TYPE,
            self.NUM_CLASSES,
            self.statistics_path,
            self.cases_path,
            self.MODALITIES,
        )

        log_statistics(
            self.statistics,
            self.TASK_NAME,
            self.TASK_TYPE,
            self.MODALITIES,
            self.NUM_CLASSES,
        )

        if resample_spacing == "median":
            # Median spacing per modality is already cached in the
            # statistics dict; collapse across modalities to a single
            # (H, W, D) spacing.
            spacing_median = np.median(
                np.asarray(self.statistics["spacing"]["median"]), axis=0
            )
            self.resample_spacing = tuple(float(v) for v in spacing_median)
        else:
            self.resample_spacing = resample_spacing

        if resize_to == "median":
            resolution_median = np.median(
                np.asarray(self.statistics["resolution"]["median"]), axis=0
            )
            self.resize_to = tuple(int(round(v)) for v in resolution_median)
        else:
            self.resize_to = resize_to

    @classmethod
    def median_resolution(cls) -> Tuple[int, ...]:
        data_root = get_data_path()
        path_to_stats = data_root / cls.FOLDER_NAME / "dataset_statistics.json"
        with open(path_to_stats, "r") as f:
            statistics = json.load(f)
        resolution_median = np.median(
            np.asarray(statistics["resolution"]["median"]), axis=0
        )
        return tuple(int(round(v)) for v in resolution_median)

    def _get_subject_directories(self) -> List[Path]:
        base_dir = self.task_dir / "preprocessed"

        if not base_dir.exists():
            raise FileNotFoundError(f"Missing preprocessed directory: {base_dir}")

        label_file = (
            self.MASK_FILENAME
            if self.TASK_TYPE == "segmentation"
            else self.LABEL_FILENAME
        )

        subjects = sorted(
            p
            for p in base_dir.iterdir()
            if p.is_dir()
            and (self.task_dir / "labels" / p.name / "ses-01" / label_file).exists()
        )

        if not subjects:
            raise RuntimeError(f"No subject directories found in {base_dir}")

        return subjects

    def _find_session_directory(self, subject_dir: Path) -> Path:
        sessions = sorted(p for p in subject_dir.iterdir() if p.is_dir())

        if len(sessions) == 0:
            raise RuntimeError(f"No session directory found in {subject_dir}")

        if len(sessions) > 1:
            logger.warning(
                f"Multiple sessions found in {subject_dir}. Using {sessions[0]}.",
            )

        return sessions[0]

    def _build_samples(self) -> List[Dict[str, Any]]:
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
            else:
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

    def _apply_split(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.fold is None or self.seed is None:
            logger.info(
                f"{self.TASK_NAME} | no fold split requested; using all samples",
            )
            return samples

        if not 0 <= self.fold < self.n_splits:
            raise ValueError(
                f"fold must be in [0, {self.n_splits - 1}], " f"got {self.fold}"
            )

        indices = np.arange(len(samples))

        if self.split in ["train", "val"]:
            if self.TASK_TYPE == "classification":
                strat_labels = _classification_labels(samples)
            elif self.TASK_TYPE == "regression":
                strat_labels = _age_bucket_labels(samples)
            elif self.TASK_TYPE == "segmentation":
                strat_labels = _segmentation_positivity_labels(samples)
            else:
                strat_labels = np.ones(len(samples))

            try:
                splitter = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.seed,
                )
                splits = list(splitter.split(indices, strat_labels))
            except ValueError as exc:
                # e.g. a bucket/class has fewer members than n_splits.
                logger.warning(
                    f"{self.TASK_NAME} | stratified split failed ({exc}); "
                    f"falling back to a plain KFold split",
                )
                splitter = KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.seed,
                )
                splits = list(splitter.split(indices))

            for current_fold, (train_indices, test_indices) in enumerate(splits):
                if current_fold == self.fold:
                    if self.split == "train":
                        selected_indices = train_indices
                    else:
                        selected_indices = test_indices

                    logger.info(
                        f"{self.TASK_NAME} | fold={self.fold} | seed={self.seed} | selected={len(selected_indices)}",
                    )

                    return [samples[int(index)] for index in selected_indices]
        else:
            return [samples[int(index)] for index in indices]

        raise RuntimeError("Unable to create requested fold.")

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

        if self.split == "train":
            # Training: infinite loop with shuffling each epoch
            while True:
                indices = list(range(len(self)))
                rng.shuffle(indices)
                for idx in indices[global_worker_id::global_workers]:
                    yield self[idx]
        else:
            # Validation/Test: single epoch without shuffling
            indices = list(range(len(self)))
            # Optionally, you could shuffle validation set too, but usually not needed
            # if self.split == "val" and self.shuffle_val: rng.shuffle(indices)
            for idx in indices[global_worker_id::global_workers]:
                yield self[idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        images = []

        for image_path in sample["image_paths"]:

            if self.resample_spacing is not None:
                image, _, _ = resample_nifti(image_path, self.resample_spacing)
            else:
                image, _, _ = load_nifti(image_path)

            image = ensure_3d(image, image_path)

            if self.resize_to is not None:
                image = resize_volume(image, self.resize_to)

            images.append(image)

        image = np.stack(images, axis=0)

        target = sample["label"]

        if self.TASK_TYPE == "segmentation":

            if self.resample_spacing is not None:
                # Resample mask with nearest-neighbor to preserve label
                # integrity (nnU-Net convention).
                target, *_ = resample_nifti(target, self.resample_spacing, is_mask=True)
            else:
                target, *_ = load_nifti(target, is_mask=True)

            target = ensure_3d(target, sample["label"])

            if self.resize_to is not None:
                target = resize_volume(target, self.resize_to, is_mask=True)

            target = torch.from_numpy(target.astype(np.int64)).unsqueeze(0)

        elif self.TASK_TYPE == "classification":

            target = torch.tensor(
                int(target),
                dtype=torch.long,
            )

        elif self.TASK_TYPE == "regression":

            target = torch.tensor(
                float(target),
                dtype=torch.float32,
            ).unsqueeze(0)

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
            sample_dict = self.transform(sample_dict)

        return sample_dict

    def create_gallery(self) -> None:
        create_gallery(
            self.samples,
            self.NUM_MODALITIES,
            self.MODALITIES,
            self.TASK_TYPE,
            self.gallery_path,
            self.GALLERY_SIZE,
        )
