"""Main entry point for med_adapt training.

Usage:
    python main.py --config configs/default.json --fold 0
    python main.py --config configs/default.json --fold 0 --test
"""

import argparse

from pathlib import Path

import numpy as np
import torch
import lightning as pl

from torchvision import transforms
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from med_adapt.registry import STORE
from med_adapt.utils.naming import get_run_name
from med_adapt.datasets import build_dataloaders
from med_adapt.utils.config import get_config, get_logger
from med_adapt.utils.paths import get_results_path, get_data_path
from med_adapt.augs.default import (
    default_enable_aug,
    default_disable_aug,
)
from med_adapt.augs import (
    Resize3D,
    PadToShape3D,
    RandomResizedCrop3D,
    RandomFlipSpatialDims3D,
    RandomRotate90SpatialPlane3D,
    CenterCrop3D,
)
from med_adapt.trainer import (
    ClassificationTrainer,
    RegressionTrainer,
)

torch.set_float32_matmul_precision("medium")

logger = get_logger(__name__)

TRAINER_CLASSES = {
    "classification": ClassificationTrainer,
    "regression": RegressionTrainer,
}


def round_up_to_multiple(values, multiple: tuple[int, int, int] = (8, 8, 8)):
    return tuple(
        int(np.ceil(v / m) * m) if m != 1 else v for v, m in zip(values, multiple)
    )


# code injection
def check_monitor_top_k(self, trainer, current=None):
    if current is None:
        return False

    if self.save_top_k == -1:
        return True

    less_than_k_models = len(self.best_k_models) < self.save_top_k
    if less_than_k_models:
        return True

    monitor_op = {"min": torch.le, "max": torch.ge}[
        self.mode
    ]  # le and ge instead of lt and gt
    should_update_best_and_save = monitor_op(
        current, self.best_k_models[self.kth_best_model_path]
    )

    # If using multiple devices, make sure all processes are unanimous on the decision.
    should_update_best_and_save = trainer.strategy.reduce_boolean_decision(
        bool(should_update_best_and_save)
    )

    return should_update_best_and_save


ModelCheckpoint.check_monitor_top_k = check_monitor_top_k


def get_task_from_dataset_name(dataset_name: str) -> str:
    """Infer task type from dataset name prefix."""
    prefix = dataset_name[:3].upper()
    if prefix == "CLS":
        return "classification"
    elif prefix == "REG":
        return "regression"
    elif prefix == "SEG":
        return "segmentation"
    else:
        raise ValueError(f"Cannot infer task from dataset name: {dataset_name}")


def build_cpu_transforms(crop_size, stage, task):
    """Build CPU-side crop/pad/resize transforms."""
    label_key = "label" if task == "segmentation" else None
    if stage == "train":
        tforms = [
            RandomRotate90SpatialPlane3D(label_key=label_key),
            RandomFlipSpatialDims3D(label_key=label_key),
            PadToShape3D(size=crop_size, label_key=label_key),
            RandomResizedCrop3D(size=crop_size, label_key=label_key, scale=(0.5, 1.0)),
        ]
    else:  # stage == "val" or "test":
        tforms = [
            PadToShape3D(size=crop_size, label_key=label_key),
            CenterCrop3D(size=crop_size, label_key=label_key),
        ]
    return transforms.Compose(tforms)


def build_model(config, task, n_modalities, n_classes, lora, mea):
    """Build model from registry using config parameters."""
    size = config.model.size

    registry_key = f"vitv2_a_3d_{size}"

    builder = STORE.get("models", registry_key)

    return builder(
        n_modalities=n_modalities,
        task=task,
        classes=n_classes,
        lora=lora,
        mea=mea,
    )


def find_best_checkpoint(run_dir, metric, mode):
    """Find the best checkpoint in a run directory.

    Looks for a checkpoint whose name contains ``metric`` followed by ``=``
    and a numeric score (e.g. ``best-acc=0.950.ckpt``). Falls back to
    ``best.ckpt`` when no scored checkpoint is found.
    """
    ckpt_dir = Path(run_dir)
    if not ckpt_dir.exists():
        return None

    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    scored = [p for p in checkpoints if metric in p.name and "=" in p.name]
    if scored:

        def _score(p):
            for part in p.name.split("-"):
                if part.startswith(f"{metric}="):
                    try:
                        val = float(part.split("=")[1].split(".")[0])
                        return val if mode == "max" else -val
                    except (ValueError, IndexError):
                        pass
            return 0.0

        return max(scored, key=_score)

    # Fallback to best.ckpt / last.ckpt
    best = ckpt_dir / "best.ckpt"
    if best.exists():
        return best
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="med_adapt training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run standalone test evaluation on best checkpoint",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint for test mode"
    )
    args = parser.parse_args()

    config = get_config(args.config)

    dataset_name = config.data.dataset_name
    dataset_class = STORE.get("datasets", dataset_name)
    task = dataset_class.TASK_TYPE

    n_modalities = dataset_class.NUM_MODALITIES
    n_classes = dataset_class.NUM_CLASSES
    config["num_classes"] = n_classes
    config["n_modalities"] = n_modalities

    data_root = str(get_data_path())
    fold = args.fold
    seed = config.seed

    crop_size = config.data.crop_size
    logger.info(f"Using {crop_size=}")

    train_cpu_transforms = build_cpu_transforms(
        crop_size,
        stage="train",
        task=task,
    )
    val_cpu_transforms = build_cpu_transforms(
        crop_size,
        stage="val",
        task=task,
    )

    model = build_model(config, task, n_modalities, n_classes, config.model.lora, True)
    train_dl, val_dl, test_dl = build_dataloaders(
        dataset_class=dataset_class,
        root=data_root,
        fold=fold,
        seed=seed,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_transforms=train_cpu_transforms,
        val_transforms=val_cpu_transforms,
        test_transforms=val_cpu_transforms,
        val_drop_last=False,
    )

    if config.enable_aug:
        gpu_transforms = default_enable_aug(ndim=3)
    else:
        gpu_transforms = default_disable_aug(ndim=3)

    trainer = TRAINER_CLASSES[task](
        config=config,
        model=model,
        gpu_augmentations=gpu_transforms,
    )

    run_name = get_run_name(
        dataset_name,
        config.model.size,
        config.model.variant,
    )
    results_path = get_results_path()

    metric = (
        "f1"
        if task == "classification"
        else "dice" if task == "segmentation" else "mae"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(results_path, name=f"{run_name}/fold_{fold}")
    log_dir = Path(csv_logger.log_dir)

    metric_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename=f"step={{step}}-val_{metric}={{val/{metric}:.3f}}",
        monitor=f"val/{metric}",
        auto_insert_metric_name=False,
        save_top_k=3,
        mode="max" if task in ["classification", "segmentation"] else "min",
        save_last=False,
        enable_version_counter=False,
        save_weights_only=True,
    )

    loss_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename=f"step={{step}}-val_loss={{val/loss:.3f}}",
        monitor=f"val/loss",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="min",
        save_last=False,
        enable_version_counter=False,
        save_weights_only=True,
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="last",
        save_last=True,
        enable_version_counter=False,
        save_weights_only=True,
    )

    pl_trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[
            metric_callback,
            loss_callback,
            last_checkpoint_callback,
            lr_monitor,
        ],
        **config.trainer.to_dict(),
        logger=csv_logger,
    )

    pl_trainer.fit(trainer, train_dataloaders=train_dl, val_dataloaders=val_dl)

    logger.info("\n[main] Running test evaluation on fold {fold}...", fold=fold)
    pl_trainer.test(
        trainer,
        dataloaders=test_dl,
        ckpt_path=metric_callback.best_model_path,
        weights_only=True,
    )


if __name__ == "__main__":
    main()
