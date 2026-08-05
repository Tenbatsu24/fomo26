import os
import json
import argparse
from pathlib import Path

import lightning as pl
import torch

from torchvision import transforms
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from gardening_tools.modules.transforms.cropping_and_padding import Torch_CropPad, Torch_Pad, Torch_CenterCrop

from fomo26.utils.naming import get_run_name
from fomo26.utils.config import load_yaml_config
from fomo26.trainer.regression import RegressionTrainer
from fomo26.aug.default import default_aug, default_norm
from fomo26.paths import get_results_path, get_config_path, get_data_path
from fomo26.trainer.segmentation import SegmentationTrainer
from fomo26.trainer.classification import ClassificationTrainer
from fomo26.dataset import (
    Task1InfarctClassification,
    Task2MeningiomaSegmentation,
    Task3BrainAgeRegression,
    Task4TrigeminalNeuralgiaSegmentation,
    Task5PolymicrogyriaClassification,
)
from fomo26.modules.data_modules.training import MedicalDataModule
from fomo26.models import (
    vitv2_a_2d_tiny,
    vitv2_a_2d_small,
    vitv2_a_2d_base,
    vitv2_a_2d_large,
    vitv2_a_3d_tiny,
    vitv2_a_3d_small,
    vitv2_a_3d_base,
    vitv2_a_3d_large,
)


MODEL_BUILDERS = {
    "2d": {
        "tiny": vitv2_a_2d_tiny,
        "small": vitv2_a_2d_small,
        "base": vitv2_a_2d_base,
        "large": vitv2_a_2d_large,
    },
    "3d": {
        "tiny": vitv2_a_3d_tiny,
        "small": vitv2_a_3d_small,
        "base": vitv2_a_3d_base,
        "large": vitv2_a_3d_large,
    }
}

TRAINER_CLASSES = {
    "classification": ClassificationTrainer,
    "regression": RegressionTrainer,
    "segmentation": SegmentationTrainer,
}

DATASET_CLASSES = {
    "CLS002_FOMO26_Infarct": Task1InfarctClassification,
    "SEG002_Meningioma": Task2MeningiomaSegmentation,
    "REG002_BrainAge": Task3BrainAgeRegression,
    "SEG002_TrigeminalNeuralgia": Task4TrigeminalNeuralgiaSegmentation,
    "CLS002_Polymicrogyria": Task5PolymicrogyriaClassification,
}


def get_task_from_dataset_name(dataset_name):
    prefix = dataset_name[:3].upper()
    if prefix == "CLS":
        return "classification"
    elif prefix == "REG":
        return "regression"
    elif prefix == "SEG":
        return "segmentation"
    else:
        raise ValueError(f"Cannot infer task from dataset name: {dataset_name}")


def get_dataset_class(dataset_name):
    cls = DATASET_CLASSES.get(dataset_name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Known datasets: {list(DATASET_CLASSES.keys())}"
        )
    return cls


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def build_cpu_transforms(crop_size, training, task):
    if task == "segmentation":
        label_key = "label"
    else:
        label_key = None
    if training:
        tforms = [Torch_CropPad(label_key=label_key, patch_size=crop_size)]
    else:
        tforms = [
            Torch_Pad(label_key=label_key, patch_size=crop_size),
            Torch_CenterCrop(label_key=label_key, target_size=crop_size)
        ]
    return transforms.Compose(tforms) if tforms else None


def build_model(config, task, n_modalities, n_classes):
    variant = config.get("model_variant", "2d")
    variant_dict = MODEL_BUILDERS[variant]
    size = config.get("model_size", "small")
    builder = variant_dict[size]
    if variant == "2d":
        return builder(
            med_in_channels=n_modalities,
            task=task,
            classes=n_classes,
            lora=config.get("lora", False),
        )
    else:
        return builder(
            volume_size=config.get("crop_size", (224, 224, 32)),
            volume_patch_size=config.get("volume_patch_size", (14, 14, 2)),
            med_in_channels=n_modalities,
            task=task,
            classes=n_classes,
            lora=config.get("lora", False),
        )


def build_datamodule(config, task, dataset_class, data_root, fold, seed):
    crop_size = config.get("crop_size", [224, 224, 32])
    train_cpu_transforms = build_cpu_transforms(crop_size, training=True, task=task)
    val_cpu_transforms = build_cpu_transforms(crop_size, training=False, task=task)

    return MedicalDataModule(
        batch_size=config.get("batch_size", 4),
        num_workers=config.get("num_workers", 2),
        dataset_class=dataset_class,
        root=data_root,
        fold=fold,
        seed=seed,
        train_transforms=train_cpu_transforms,
        val_transforms=val_cpu_transforms,
        val_drop_last=(task == "segmentation"),
    )


def build_trainer_module(config, task, model):
    gpu_transforms = default_aug(ndim=3)
    norm_transforms = default_norm()
    trainer_class = TRAINER_CLASSES[task]
    return trainer_class(
        model=model,
        config=config,
        gpu_transforms=gpu_transforms,
        norm_transforms=norm_transforms,
    )


def find_best_checkpoint(run_dir, metric, mode):
    """Find the best checkpoint in a run directory.

    Returns the path to the best checkpoint or None if none found.
    """
    ckpt_dir = Path(run_dir)
    if not ckpt_dir.exists():
        return None

    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    # Sort by the metric in the filename
    def _score(p):
        name = p.name
        # Extract metric value from filename like "epoch=10-val/iou=0.85.ckpt"
        for part in name.split("-"):
            if metric in part:
                try:
                    val = float(part.split("=")[1].split(".")[0])
                    return val if mode == "max" else -val
                except (ValueError, IndexError):
                    pass
        return 0.0

    return max(checkpoints, key=_score)


def run_test_mode(config, dataset_name, dataset_class, task, n_modalities, n_classes,
                  fold, seed, checkpoint_path=None):
    """Run standalone test evaluation on a saved checkpoint.

    Args:
        config: loaded config dict.
        checkpoint_path: path to checkpoint to evaluate. If None, uses the
            best checkpoint from the run directory.
    """
    metric = "acc" if task == "classification" else "iou" if task == "segmentation" else "l2"

    run_name = get_run_name(
        dataset_name,
        config.get("model_variant", "small"),
        config.get("model_size", "2d"),
        config.get("lora", False)
    )
    results_path = get_results_path()
    run_dir = os.path.join(results_path, run_name, f"fold{fold}")

    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(run_dir, metric,
            "max" if task in ["classification", "segmentation"] else "min")

    if checkpoint_path is None:
        print(f"[main] No checkpoint found in {run_dir}. Nothing to test.")
        return

    print(f"[main] Loading checkpoint: {checkpoint_path}")

    model = build_model(config, task, n_modalities, n_classes)
    datamodule = build_datamodule(config, task, dataset_class,
                                   config.get("data_root", get_data_path()), fold, seed)
    trainer_module = build_trainer_module(config, task, model)

    # Load checkpoint weights only (not optimizer state)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)

    run_name_test = f"{run_name}-test"
    logger = CSVLogger(results_path, name=run_name_test, version=f"fold{fold}")

    pl_trainer = pl.Trainer(
        precision=config.get("precision", "32-true"),
        accelerator="auto",
        devices=config.get("devices", "auto"),
        strategy=config.get("strategy", "auto"),
        logger=logger,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    pl_trainer.test(trainer_module, datamodule=datamodule)

    # Also print the test metrics to stdout
    test_logs = logger.experiment.aggregates
    print(f"\n[main] Test metrics for {run_name} fold {fold}:")
    for key, val in test_logs.items():
        if isinstance(val, list) and val:
            print(f"  {key}: {val[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--test", action="store_true",
                        help="Run standalone test evaluation on best checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for test mode (defaults to best in run dir)")
    args = parser.parse_args()

    config_path = Path(get_config_path()) / args.config
    config = load_yaml_config(config_path)

    dataset_name = config.get("dataset_name", "CLS002_FOMO26_Infarct")
    dataset_class = get_dataset_class(dataset_name)
    task = dataset_class.TASK_TYPE

    # Standalone test mode
    if args.test:
        run_test_mode(
            config=config,
            dataset_name=dataset_name,
            dataset_class=dataset_class,
            task=task,
            n_modalities=dataset_class.NUM_MODALITIES,
            n_classes=dataset_class.NUM_CLASSES,
            fold=args.fold,
            seed=config.get("seed", 42),
            checkpoint_path=args.checkpoint,
        )
        return

    metric = "acc" if task == "classification" else "iou" if task == "segmentation" else "l2"

    n_modalities = dataset_class.NUM_MODALITIES
    n_classes = dataset_class.NUM_CLASSES

    config["num_classes"] = n_classes
    config["n_modalities"] = n_modalities

    data_root = config.get("data_root", get_data_path())
    fold = args.fold
    seed = config.get("seed", 42)

    model = build_model(config, task, n_modalities, n_classes)
    datamodule = build_datamodule(config, task, dataset_class, data_root, fold, seed)
    trainer_module = build_trainer_module(config, task, model)

    run_name = get_run_name(
        dataset_name,
        config.get("model_variant", "small"),
        config.get("model_size", "2d"),
        config.get("lora", False)
    )

    results_path = get_results_path()

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(results_path, run_name, f"fold{args.fold}"),
        filename=f"{{epoch}}-{{val/{metric}:.2f}}",
        monitor=f"val/{metric}",
        save_top_k=1,
        mode="max" if task in ["classification", "segmentation"] else "min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(results_path, run_name, f"fold{args.fold}"),
        filename="last",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(results_path, name=run_name, version=f"fold{args.fold}")

    pl_trainer = pl.Trainer(
        max_steps=config.get("max_steps", 100_000),
        default_root_dir=os.path.join(results_path, run_name, f"fold{args.fold}"),
        callbacks=[checkpoint_callback, last_checkpoint_callback, lr_monitor],
        precision=config.get("precision", "bf16-mixed"),
        accelerator="auto",
        devices=config.get("devices", "auto"),
        strategy=config.get("strategy", "auto"),
        log_every_n_steps=config.get("max_steps", 100_000) // 100,
        gradient_clip_val=config.get("gradient_clip_val", None),
        val_check_interval=config.get("max_steps", 100_000) // 10,
        check_val_every_n_epoch=None,
        logger=logger
    )

    pl_trainer.fit(trainer_module, datamodule=datamodule)

    # Run test evaluation after training
    print(f"\n[main] Running test evaluation on fold {args.fold}...")
    pl_trainer.test(trainer_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
