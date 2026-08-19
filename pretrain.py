"""Main entry point for med_adapt training.

Usage:
    python main.py --config configs/pretrain.json
"""

import argparse
import uuid

from pathlib import Path

import torch
import lightning as pl

from torchvision import transforms
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from med_adapt.datasets import build_pretrain_dataloaders
from med_adapt.registry import STORE
from med_adapt.utils.config import get_config, get_logger
from med_adapt.utils.paths import get_results_path, get_nnssl_preprocessed_path
from med_adapt.augs import (
    default_enable_aug,
    default_disable_aug,
    PadToShape3D,
    RandomResizedCrop3D,
    CenterCrop3D,
    RandomSwapSpatialDims3D,
)
from med_adapt.trainer import PretrainTrainer

logger = get_logger(__name__)

torch.set_float32_matmul_precision("medium")


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


def build_cpu_transforms(crop_size, training, task):
    label_key = "label" if task == "segmentation" else None
    if training:
        tforms = [
            RandomSwapSpatialDims3D(p=0.5, label_key=label_key),
            PadToShape3D(crop_size, label_key=label_key),
            RandomResizedCrop3D(crop_size, label_key=label_key),
        ]
    else:
        tforms = [
            PadToShape3D(crop_size, label_key=label_key),
            CenterCrop3D(crop_size, label_key=label_key),
        ]
    return transforms.Compose(tforms) if tforms else None


def build_model(config, n_modalities):
    """Build model from registry using config parameters."""
    size = config.model.size

    teacher_registry_key = f"vitv2_{size}"
    student_registry_key = f"vitv2_3d_{size}"

    teacher_cls = STORE.get("models", teacher_registry_key)
    student_cls = STORE.get("models", student_registry_key)

    teacher_model = teacher_cls(
        img_size=518,
        patch_size=14,
        in_chans=3,
    )

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    student_model = student_cls(
        volume_size=tuple(config.data.crop_size),
        volume_patch_size=tuple(config.model.volume_patch_size),
        med_in_channels=n_modalities,
        use_patch_decode=config.model.use_patch_decode,
        use_mask=config.model.use_mask,
    )

    return teacher_model, student_model


def main():
    parser = argparse.ArgumentParser(description="med_adapt training")
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    config = get_config(args.config)

    dataset_name = config.data.dataset_name
    dataset_class = STORE.get("datasets", dataset_name)
    task = dataset_class.TASK_TYPE

    n_modalities = dataset_class.NUM_MODALITIES
    n_classes = dataset_class.NUM_CLASSES
    config["num_classes"] = n_classes
    config["n_modalities"] = n_modalities

    data_root = str(get_nnssl_preprocessed_path())
    seed = config.seed
    crop_size = tuple(config.data.crop_size)

    train_cpu_transforms = build_cpu_transforms(crop_size, training=True, task=task)
    val_cpu_transforms = build_cpu_transforms(crop_size, training=False, task=task)

    teacher_model, student_model = build_model(config, n_modalities)
    train_dl, val_dl = build_pretrain_dataloaders(
        dataset_class=dataset_class,
        root=data_root,
        split_seed=seed,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_transforms=train_cpu_transforms,
        val_transforms=val_cpu_transforms,
    )

    if config.enable_aug:
        gpu_transforms = default_enable_aug(ndim=3)
    else:
        gpu_transforms = default_disable_aug(ndim=3)

    trainer = PretrainTrainer(
        config=config,
        model=student_model,
        teacher_model=teacher_model,
        gpu_augmentations=gpu_transforms,
    )

    run_name = "p_l2_n-t_r-s_nog-mask"
    results_path = get_results_path()

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # csv_logger = CSVLogger(results_path, name=f"{run_name}")
    wandb_logger = WandbLogger(run_name, save_dir=results_path, project="fomo26")

    model_dir = Path(results_path) / run_name / "base"
    print(model_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"step={{step}}-val_loss={{val/loss:.3f}}",
        monitor=f"val/loss",
        auto_insert_metric_name=False,
        save_top_k=-1,
        mode="min",
        save_last=False,
        enable_version_counter=False,
        save_weights_only=True,
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="last",
        save_last=True,
        enable_version_counter=False,
        save_weights_only=True,
    )

    pl_trainer = pl.Trainer(
        default_root_dir=results_path,
        callbacks=[checkpoint_callback, last_checkpoint_callback, lr_monitor],
        **config.trainer.to_dict(),
        logger=wandb_logger,
    )

    pl_trainer.fit(trainer, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
