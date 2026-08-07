"""Dataset task definitions.

Each class here is a concrete dataset configuration — it only sets
class-level constants and is registered via ``@register_dataset``.
The base class :class:`MedicalTaskDataset` lives in ``data.py``.
"""

from med_adapt.registry import register_dataset
from med_adapt.datasets.data import MedicalTaskDataset


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


@register_dataset("SEG002_FOMO26_Infarct")
class Task1InfarctSegmentation(MedicalTaskDataset):

    FOLDER_NAME = "Task_1"
    TASK_NAME = "Infarct Segmentation"
    TASK_TYPE = "segmentation"

    MODALITIES = (
        "adc",
        "dwi_b1000",
        "flair",
    )

    NUM_MODALITIES = 3

    NUM_CLASSES = 2

    LABEL_FILENAME = None

    MASK_FILENAME = "seg.nii.gz"


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

    from pathlib import Path

    from med_adapt.utils.config import get_logger

    logger = get_logger(__name__)

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
        # Task1InfarctClassification,
        Task1InfarctSegmentation,
        # Task2MeningiomaSegmentation,
        # Task3BrainAgeRegression,
        # Task4TrigeminalNeuralgiaSegmentation,
        # Task5PolymicrogyriaClassification,
    ]

    logger.info("=" * 100)
    logger.info("DATASET INSPECTION")
    logger.info(f"Root: {root}")
    logger.info("=" * 100)

    for dataset_cls in datasets:

        logger.info("")
        logger.info("=" * 100)
        logger.info(
            f"Inspecting {dataset_cls.TASK_NAME}",
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
                f"Finished inspection of {dataset_cls.TASK_NAME}",
            )

            logger.info(
                f"Statistics saved to: {dataset.statistics_path}",
            )

            logger.info(
                f"Per-case metadata saved to: {dataset.cases_path}",
            )

            logger.info(
                f"Histogram saved to: {dataset.histogram_path}",
            )

            logger.info(f"Creating Gallery...")
            dataset.create_gallery()
            logger.info("Finished creating gallery.")

        except Exception:

            logger.exception(
                f"Failed to inspect {dataset_cls.TASK_NAME}",
            )

    logger.info("")
    logger.info("=" * 100)
    logger.info("DATASET INSPECTION COMPLETE")
    logger.info("=" * 100)
