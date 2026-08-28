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

from med_adapt.utils import get_data_path
from med_adapt.datasets.io import (
    ensure_3d,
    load_nifti,
    normalize_subject_name,
    read_labels,
    resample_nifti,
    resize_volume,
)
from med_adapt.datasets.statistics import (
    compute_preprocessing_geometry,
    load_or_compute_statistics,
    log_statistics,
    read_cases_csv,
    transpose_from_resolution,
)
from med_adapt.datasets.visualisation import create_gallery

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_AGE_BUCKET_EDGES: Tuple[int, ...] = (40, 60, 80)


def _classification_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(sample["label"]) for sample in samples])


def _age_bucket_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Bucket brain-age regression targets for stratified splitting.

    Buckets: <20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, >=90.
    """
    ages = np.asarray([float(sample["label"]) for sample in samples])
    return np.digitize(ages, _AGE_BUCKET_EDGES)


# mask path -> "contains any foreground voxel" (0/1).
#
# Stratified splitting needs this once per subject, but every fold/split
# instantiation used to reload and re-scan every mask from scratch just to
# recompute the same flag (e.g. in 5-fold CV each mask is used in ~4 train
# sets and 1 val set -> ~5 full reloads of the same volume). Cached here
# instead; label files aren't expected to change mid-run.
_mask_positivity_cache: Dict[str, int] = {}


def _mask_has_foreground(mask_path: Path) -> int:
    key = str(mask_path)
    cached = _mask_positivity_cache.get(key)
    if cached is None:
        mask, _, _ = load_nifti(mask_path, is_mask=True)
        mask = ensure_3d(mask, mask_path)
        cached = int(np.any(mask > 0))
        _mask_positivity_cache[key] = cached
    return cached


def _segmentation_positivity_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Label each sample by whether its mask contains any foreground voxel.

    Used so the train/val split keeps a roughly similar ratio of
    positive/negative scans in both folds, instead of letting a random
    split concentrate positives in one side.
    """
    return np.asarray([_mask_has_foreground(sample["label"]) for sample in samples])


class MedicalTaskDataset(IterableDataset):
    FOLDER_NAME: str = ""
    TASK_NAME: str = ""
    TASK_TYPE: str = ""

    MODALITIES: Tuple[str, ...] = ()
    NUM_MODALITIES: int = 0
    NUM_CLASSES: Optional[int] = None

    LABEL_FILENAME: Optional[str] = None
    MASK_FILENAME: Optional[str] = None

    GALLERY_SIZE: int = 8

    # Process-wide caches, shared by every fold/split instance of a given
    # (dataset class, data root) -- keying on both means different
    # subclasses or roots never collide. Both assume the underlying files
    # don't change mid-run; restart the process if they do.
    _sample_cache: Dict[Tuple[type, str], List[Dict[str, Any]]] = {}
    _geometry_cache: Dict[Tuple[type, str, float], Dict[str, Any]] = {}

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

        self.cases_path = self.task_dir / "dataset_cases.csv"
        self.histogram_path = self.task_dir / "dataset_histograms.png"
        self.gallery_path = self.task_dir / "dataset_gallery.png"

        # BUGFIX: statistics must be computed from the *full*, pre-split
        # `samples` list, not `self.samples`. Passing the split subset here
        # would make the cached dataset_cases.csv -- and therefore every
        # median spacing/resolution/transpose derived from it, for every
        # future instance of this class regardless of split -- silently
        # reflect whichever single fold/split happened to construct it
        # first, instead of the whole dataset.
        self.statistics = load_or_compute_statistics(
            samples=samples,
            task_name=self.TASK_NAME,
            folder_name=self.FOLDER_NAME,
            task_type=self.TASK_TYPE,
            num_classes=self.NUM_CLASSES,
            cases_path=self.cases_path,
            modalities=self.MODALITIES,
        )

        log_statistics(
            self.statistics,
            self.TASK_NAME,
            self.TASK_TYPE,
            self.MODALITIES,
            self.NUM_CLASSES,
        )

        preprocessing = self.statistics["preprocessing"]
        median_spacing = preprocessing["median_spacing"]

        self.transpose = preprocessing["transpose"]
        median_resolution = preprocessing["median_resolution"]

        if resize_to == "median":
            self.resample_spacing = median_spacing
            self.resize_to = median_resolution

        else:
            if resample_spacing == "median":
                self.resample_spacing = median_spacing
            else:
                self.resample_spacing = resample_spacing

            self.resize_to = resize_to

    @classmethod
    def _read_case_rows(
        cls,
        root: str | Path | None = None,
    ) -> List[Dict[str, Any]]:
        if root is None:
            root = get_data_path()

        cases_path = Path(root) / cls.FOLDER_NAME / "dataset_cases.csv"

        if not cases_path.exists():
            raise FileNotFoundError(f"Dataset cases CSV does not exist: {cases_path}")

        rows = read_cases_csv(cases_path)

        if cls.MODALITIES:
            modalities = set(cls.MODALITIES)
            rows = [row for row in rows if row["modality"] in modalities]

        if not rows:
            raise ValueError(
                f"No rows for modalities {cls.MODALITIES} "
                f"were found in {cases_path}"
            )

        return rows

    @classmethod
    def _geometry(cls, root: str | Path | None = None) -> Dict[str, Any]:
        """Median spacing/resolution/transpose, cached per (class, root).

        This describes the whole dataset and is independent of any
        train/val/test split, so it's wasteful to recompute it from
        scratch for every fold/split instance. Cached here and
        auto-invalidated if dataset_cases.csv is ever rewritten.
        """
        resolved_root = Path(root) if root is not None else get_data_path()
        cases_path = resolved_root / cls.FOLDER_NAME / "dataset_cases.csv"

        if not cases_path.exists():
            raise FileNotFoundError(f"Dataset cases CSV does not exist: {cases_path}")

        cache_key = (cls, str(cases_path.resolve()), cases_path.stat().st_mtime)

        cached = cls._geometry_cache.get(cache_key)
        if cached is not None:
            return cached

        rows = cls._read_case_rows(resolved_root)
        geometry = compute_preprocessing_geometry(rows, cls.MODALITIES)

        cls._geometry_cache[cache_key] = geometry
        return geometry

    @classmethod
    def find_median_spacing(
        cls,
        root: str | Path | None = None,
    ) -> Tuple[float, float, float]:
        return cls._geometry(root)["median_spacing"]

    @classmethod
    def find_median_resolution(
        cls,
        root: str | Path | None = None,
        median_spacing: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[int, int, int]:
        if median_spacing is not None:
            # An explicit target spacing overrides the dataset's own
            # median -- compute directly for this one-off request rather
            # than caching it under the default geometry.
            rows = cls._read_case_rows(root)
            geometry = compute_preprocessing_geometry(
                rows, cls.MODALITIES, median_spacing=median_spacing
            )
            return geometry["median_resolution_at_median_spacing"]

        return cls._geometry(root)["median_resolution_at_median_spacing"]

    @classmethod
    def find_transpose(
        cls,
        root: str | Path | None = None,
        median_resolution: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[int, int, int]:
        if median_resolution is not None:
            return transpose_from_resolution(median_resolution)

        return cls._geometry(root)["transpose"]

    @classmethod
    def median_resolution(cls):
        data_root = get_data_path()
        return cls._geometry(data_root)["median_resolution"]

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        sample = self.samples[index]

        images = []

        for image_path in sample["image_paths"]:

            # Resampling MUST happen before transposing because spacing
            # is specified in the original/native image axis order.
            if self.resample_spacing is not None:
                image, _, _ = resample_nifti(
                    image_path,
                    self.resample_spacing,
                )
            else:
                image, _, _ = load_nifti(
                    image_path,
                )

            image = ensure_3d(
                image,
                image_path,
            )

            # All samples are canonicalized, regardless of whether
            # resampling/resizing was requested.
            image = np.transpose(
                image,
                self.transpose,
            )

            if self.resize_to is not None:
                image = resize_volume(
                    image,
                    self.resize_to,
                )

            images.append(image)

        image = np.stack(
            images,
            axis=0,
        )

        target = sample["label"]

        if self.TASK_TYPE == "segmentation":

            if self.resample_spacing is not None:
                # Use exactly the same physical resampling as the image,
                # but nearest-neighbor interpolation for labels.
                target, *_ = resample_nifti(
                    target,
                    self.resample_spacing,
                    is_mask=True,
                )
            else:
                target, *_ = load_nifti(
                    target,
                    is_mask=True,
                )

            target = ensure_3d(
                target,
                sample["label"],
            )

            # Image and mask must use precisely the same permutation.
            target = np.transpose(
                target,
                self.transpose,
            )

            if self.resize_to is not None:
                target = resize_volume(
                    target,
                    self.resize_to,
                    is_mask=True,
                )

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

    def convert_to_nnunet_format(
        self,
        dataset_id: int,
        task_name: Optional[str] = None,
        n_splits: int = 5,
        seed: int = 1234,
    ) -> Path:
        import shutil
        import nibabel as nib
        import numpy as np

        from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
        from batchgenerators.utilities.file_and_folder_operations import (
            join,
            maybe_mkdir_p,
            save_json,
        )
        from nnunetv2.dataset_conversion.generate_dataset_json import (
            generate_dataset_json,
        )

        logger.info(f"Converting to nnunet format... {self.TASK_NAME}")

        task_name = task_name or self.TASK_NAME
        dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

        out_dir = Path(str(nnUNet_raw).replace('"', "")) / dataset_name
        images_tr_dir = out_dir / "imagesTr"
        labels_tr_dir = out_dir / "labelsTr"
        images_ts_dir = out_dir / "imagesTs"

        for d in (images_tr_dir, labels_tr_dir, images_ts_dir):
            d.mkdir(parents=True, exist_ok=True)

        for sample in self.samples:
            subject = sample["subject"]

            for modality_index, image_path in enumerate(sample["image_paths"]):
                shutil.copy(
                    image_path,
                    images_tr_dir / f"{subject}_{modality_index:04d}.nii.gz",
                )

            label_value = sample["label"]
            label_dest = labels_tr_dir / f"{subject}.nii.gz"
            label_txt_dest = labels_tr_dir / f"{subject}.txt"

            if isinstance(label_value, (int, float)):
                # It's a scalar value - create txt file with the value
                with open(label_txt_dest, "w") as f:
                    f.write(str(float(label_value)))

                # Create a nii.gz file with mask of non-zero voxels from the first image
                first_image_path = sample["image_paths"][0]
                nib_img = nib.load(first_image_path)
                img = nib_img.get_fdata()

                # Create boolean mask where image is non-zero
                mask_data = img != 0
                mask_img = nib.Nifti1Image(
                    mask_data.astype(np.uint8), nib_img.affine, nib_img.header
                )
                nib.save(mask_img, label_dest)
            else:
                # It's a path to .nii.gz file - copy it
                shutil.copy(label_value, label_dest)

        channel_names = {i: modality for i, modality in enumerate(self.MODALITIES)}

        labels = {"background": 0}
        for class_index in range(1, self.NUM_CLASSES):
            labels[f"class_{class_index}"] = class_index

        generate_dataset_json(
            str(out_dir),
            channel_names=channel_names,
            labels=labels,
            file_ending=".nii.gz",
            num_training_cases=len(self.samples),
        )

        subjects = np.array([sample["subject"] for sample in self.samples])
        indices = np.arange(len(subjects))

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_indices = list(splitter.split(indices))

        splits = [
            {
                "train": subjects[train_idx].tolist(),
                "val": subjects[val_idx].tolist(),
            }
            for train_idx, val_idx in fold_indices
        ]

        preprocessed_dir = (
            Path(str(nnUNet_preprocessed).replace('"', "")) / dataset_name
        )
        maybe_mkdir_p(str(preprocessed_dir))
        save_json(
            splits, join(str(preprocessed_dir), "splits_final.json"), sort_keys=False
        )

        return out_dir

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
            while True:
                indices = list(range(len(self)))
                rng.shuffle(indices)
                for idx in indices[global_worker_id::global_workers]:
                    yield self[idx]
        else:
            indices = list(range(len(self)))
            for idx in indices[global_worker_id::global_workers]:
                yield self[idx]

    def __len__(self) -> int:
        return len(self.samples)

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
        cache_key = (type(self), str(self.task_dir))
        cached = MedicalTaskDataset._sample_cache.get(cache_key)
        if cached is not None:
            return cached

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

        MedicalTaskDataset._sample_cache[cache_key] = samples
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

    def create_gallery(self) -> None:
        create_gallery(
            self.samples,
            self.NUM_MODALITIES,
            self.MODALITIES,
            self.TASK_TYPE,
            self.gallery_path,
            self.GALLERY_SIZE,
        )
