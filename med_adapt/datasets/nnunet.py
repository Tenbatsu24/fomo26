import random

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import blosc2
import torch
import numpy as np
import torch.distributed as dist

from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from sklearn.model_selection import KFold, StratifiedKFold

from med_adapt.registry import register_dataset
from med_adapt.utils.config import get_logger
from med_adapt.utils.paths import get_nnunet_processed_root

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_AGE_BUCKET_EDGES: Tuple[int, ...] = (40, 55, 70)


def _classification_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(sample["label"]) for sample in samples])


def _age_bucket_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    ages = np.asarray([float(sample["label"]) for sample in samples])
    return np.digitize(ages, _AGE_BUCKET_EDGES)


class NNDataset(IterableDataset):
    TASK_NAME: str = ""
    PLANS_NAME: str = ""

    TASK_TYPE: str = ""

    NUM_MODALITIES: int = 0
    NUM_CLASSES: Optional[int] = None

    def __init__(
        self,
        root: str | Path = get_nnunet_processed_root(),
        split: str = "train",
        fold: Optional[int] = None,
        seed: Optional[int] = None,
        n_splits: int = 5,
        transform=None,
    ):
        self.root = Path(root) / self.TASK_NAME
        self.split = split
        self.samples_dir = self.root / self.PLANS_NAME

        self.fold = fold
        self.seed = seed
        self.n_splits = n_splits

        if not self.samples_dir.exists():
            raise FileNotFoundError(
                f"Task directory does not exist: {self.samples_dir}"
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

        self.transform = transform

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        sample = self.samples[index]

        target = sample["label"]
        image_path = sample["image"]

        if self.TASK_TYPE == "classification":
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

        image = torch.from_numpy(blosc2.open(image_path)[:]).to(torch.float32)

        sample_dict = {
            "image": image,
            "label": target,
        }
        if self.transform is not None:
            sample_dict = self.transform(sample_dict)

        return sample_dict

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

    def _build_samples(self) -> List[Dict[str, Any]]:
        """Build list of samples from the samples directory."""
        samples = []

        # Get all .b2nd files in the samples directory
        image_files = sorted(self.samples_dir.glob("*.b2nd"))

        for image_path in image_files:
            # Get the corresponding .txt file with the same stem
            txt_path = image_path.with_suffix(".txt")

            if not txt_path.exists():
                logger.warning(
                    f"{self.TASK_NAME} | missing label file for {image_path.name}"
                )
                continue

            # Read the label from the .txt file
            try:
                with open(txt_path, "r") as f:
                    label_content = f.read().strip()

                # Try to convert to float first, fall back to string if needed
                try:
                    label = float(label_content)
                except ValueError:
                    label = label_content

                samples.append(
                    {
                        "image": str(image_path),
                        "label": label,
                    }
                )
            except Exception as e:
                logger.warning(f"{self.TASK_NAME} | error reading {txt_path}: {e}")
                continue

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


@register_dataset("Task1")
class Task1(NNDataset):
    TASK_NAME: str = "Dataset003_InfarctClassification"
    PLANS_NAME: str = "Spacing__1.00_1.00_1.00___Norm__Z_Z_Z"

    TASK_TYPE: str = "classification"

    NUM_MODALITIES: int = 3
    NUM_CLASSES = 2


@register_dataset("Task3")
class Task3(NNDataset):
    TASK_NAME: str = "Dataset004_BrainAgeRegression"
    PLANS_NAME: str = "Spacing__1.00_1.00_1.00___Norm__Z"

    TASK_TYPE: str = "regression"

    NUM_MODALITIES: int = 1
    NUM_CLASSES = 1


@register_dataset("Task5")
class Task5(NNDataset):
    TASK_NAME: str = "Dataset005_PolymicrogyriaClassification"
    PLANS_NAME: str = "Spacing__1.00_1.00_1.00___Norm__Z"

    TASK_TYPE: str = "classification"

    NUM_MODALITIES: int = 1
    NUM_CLASSES = 2


if __name__ == "__main__":
    _ds = Task4(seed=1234, fold=0, split="train", n_splits=5)
    print(len(_ds))
