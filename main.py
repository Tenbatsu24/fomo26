"""Main entry point for med_adapt training.

Usage:
    python main.py --config configs/default.json --fold 0
    python main.py --config configs/default.json --fold 0 --test
"""

import argparse

from pathlib import Path

import torch
import lightning as pl

from torchvision import transforms
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from gardening_tools.modules.transforms.cropping_and_padding import (
    Torch_CropPad,
    Torch_Pad,
    Torch_CenterCrop,
)

from med_adapt.registry import STORE
from med_adapt.utils.naming import get_run_name
from med_adapt.datasets import build_dataloaders
from med_adapt.utils.config import get_config, get_logger
from med_adapt.utils.paths import get_results_path, get_data_path
from med_adapt.augs.default import default_enable_aug, default_norm
from med_adapt.trainer import (
    ClassificationTrainer,
    RegressionTrainer,
    SegmentationTrainer,
)

logger = get_logger(__name__)

TRAINER_CLASSES = {
    "classification": ClassificationTrainer,
    "regression": RegressionTrainer,
    "segmentation": SegmentationTrainer,
}


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


def build_cpu_transforms(crop_size, training, task):
    """Build CPU-side crop/pad transforms."""
    label_key = "label" if task == "segmentation" else None
    if training:
        tforms = [Torch_CropPad(label_key=label_key, patch_size=crop_size)]
    else:
        tforms = [
            Torch_Pad(label_key=label_key, patch_size=crop_size),
            Torch_CenterCrop(label_key=label_key, target_size=crop_size),
        ]
    return transforms.Compose(tforms) if tforms else None


def build_model(config, task, n_modalities, n_classes):
    """Build model from registry using config parameters."""
    variant = config.model.variant
    size = config.model.size

    if variant == "2d":
        registry_key = f"vitv2_a_2d_{size}"
    else:
        registry_key = f"vitv2_a_3d_{size}"

    builder = STORE.get("models", registry_key)
    if variant == "2d":
        return builder(
            med_in_channels=n_modalities,
            task=task,
            classes=n_classes,
            lora=config.model.lora,
        )
    else:
        return builder(
            volume_size=tuple(config.data.crop_size),
            volume_patch_size=tuple(config.data.volume_patch_size),
            med_in_channels=n_modalities,
            task=task,
            classes=n_classes,
            lora=config.model.lora,
        )


def find_best_checkpoint(run_dir, metric, mode):
    """Find the best checkpoint in a run directory."""
    ckpt_dir = Path(run_dir)
    if not ckpt_dir.exists():
        return None

    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    def _score(p):
        name = p.name
        for part in name.split("-"):
            if metric in part:
                try:
                    val = float(part.split("=")[1].split(".")[0])
                    return val if mode == "max" else -val
                except (ValueError, IndexError):
                    pass
        return 0.0

    return max(checkpoints, key=_score)


def run_test_mode(
    config,
    dataset_name,
    dataset_class,
    task,
    n_modalities,
    n_classes,
    fold,
    seed,
    checkpoint_path=None,
):
    """Run standalone test evaluation on a saved checkpoint."""
    metric = (
        "acc" if task == "classification" else "iou" if task == "segmentation" else "l2"
    )

    run_name = get_run_name(
        dataset_name,
        config.model.size,
        config.model.variant,
        config.model.lora,
    )
    results_path = get_results_path()
    run_dir = Path(results_path) / run_name / f"fold{fold}"

    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(
            run_dir,
            metric,
            "max" if task in ["classification", "segmentation"] else "min",
        )

    if checkpoint_path is None:
        logger.warning("[main] No checkpoint found in {run_dir}. Nothing to test.")
        return

    logger.info("[main] Loading checkpoint: {ckpt}", ckpt=checkpoint_path)

    model = build_model(config, task, n_modalities, n_classes)
    crop_size = tuple(config.data.crop_size)
    test_transforms = build_cpu_transforms(crop_size, training=False, task=task)
    train_dl, val_dl, test_dl = build_dataloaders(
        dataset_class=dataset_class,
        root=str(get_data_path()),
        fold=fold,
        seed=seed,
        batch_size=1,
        num_workers=config.data.num_workers,
        test_transforms=test_transforms,
    )

    # Load checkpoint weights only
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)

    # Disable pretrained loading in test mode — we already loaded the test checkpoint above.
    config["pretrained"]["checkpoint"] = None
    trainer = TRAINER_CLASSES[task](config=config, model=model)

    run_name_test = f"{run_name}-test"
    results_path = get_results_path()
    logger_obj = CSVLogger(results_path, name=run_name_test, version=f"fold{fold}")

    pl_trainer = pl.Trainer(
        precision=config.trainer.precision,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy=config.trainer.strategy,
        logger=logger_obj,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    pl_trainer.test(trainer, dataloaders=test_dl)

    test_logs = logger_obj.experiment.aggregates
    logger.info(
        "\n[main] Test metrics for {run_name} fold {fold}:",
        run_name=run_name,
        fold=fold,
    )
    for key, val in test_logs.items():
        if isinstance(val, list) and val:
            logger.info("  {key}: {val:.4f}", key=key, val=val[-1])


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

    if args.test:
        run_test_mode(
            config=config,
            dataset_name=dataset_name,
            dataset_class=dataset_class,
            task=task,
            n_modalities=dataset_class.NUM_MODALITIES,
            n_classes=dataset_class.NUM_CLASSES,
            fold=args.fold,
            seed=config.seed,
            checkpoint_path=args.checkpoint,
        )
        return

    n_modalities = dataset_class.NUM_MODALITIES
    n_classes = dataset_class.NUM_CLASSES
    config["num_classes"] = n_classes
    config["n_modalities"] = n_modalities

    data_root = str(get_data_path())
    fold = args.fold
    seed = config.seed
    crop_size = tuple(config.data.crop_size)

    train_cpu_transforms = build_cpu_transforms(crop_size, training=True, task=task)
    val_cpu_transforms = build_cpu_transforms(crop_size, training=False, task=task)

    model = build_model(config, task, n_modalities, n_classes)
    train_dl, val_dl, _ = build_dataloaders(
        dataset_class=dataset_class,
        root=data_root,
        fold=fold,
        seed=seed,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_transforms=train_cpu_transforms,
        val_transforms=val_cpu_transforms,
        val_drop_last=(task == "segmentation"),
    )

    gpu_transforms = default_enable_aug(ndim=3)
    norm_transforms = default_norm()

    trainer = TRAINER_CLASSES[task](
        config=config,
        model=model,
        gpu_augmentations=gpu_transforms,
        normalisation=norm_transforms,
    )

    run_name = get_run_name(
        dataset_name,
        config.model.size,
        config.model.variant,
        config.model.lora,
    )
    results_path = get_results_path()

    metric = (
        "acc" if task == "classification" else "iou" if task == "segmentation" else "l2"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(results_path) / run_name / f"fold{fold}",
        filename=f"best",
        monitor=f"val/{metric}",
        save_top_k=1,
        mode="max" if task in ["classification", "segmentation"] else "min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(results_path) / run_name / f"fold{fold}",
        filename="last",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(results_path, name=run_name, version=f"fold{fold}")

    pl_trainer = pl.Trainer(
        default_root_dir=Path(results_path) / run_name / f"fold{fold}",
        callbacks=[checkpoint_callback, last_checkpoint_callback, lr_monitor],
        **config.trainer.to_dict(),
        logger=csv_logger,
    )

    pl_trainer.fit(trainer, train_dataloaders=train_dl, val_dataloaders=val_dl)

    logger.info("\n[main] Running test evaluation on fold {fold}...", fold=fold)
    pl_trainer.test(trainer, dataloaders=val_dl)


if __name__ == "__main__":
    main()
