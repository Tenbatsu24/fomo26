import os
import json
import argparse

import lightning as pl

from torchvision import transforms
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from gardening_tools.modules.transforms.cropping_and_padding import Torch_CropPad, Torch_CenterCrop

from fomo26.utils.naming import get_run_name
from fomo26.utils.config import load_yaml_config
from fomo26.trainer.regression import RegressionTrainer
from fomo26.aug.default import default_aug, default_norm
from fomo26.trainer.segmentation import SegmentationTrainer
from fomo26.trainer.classification import ClassificationTrainer
from fomo26.utils.dataset import get_dataset_metadata, load_fold
from fomo26.paths import get_models_path, get_results_path, get_config_path
from fomo26.models.extended import vitv2_tiny, vitv2_small, vitv2_base, vitv2_large
from fomo26.modules.data_modules.training import SegDataModule, ClsRegDataModule


MODEL_BUILDERS = {
    "tiny": vitv2_tiny,
    "small": vitv2_small,
    "base": vitv2_base,
    "large": vitv2_large,
}

TRAINER_CLASSES = {
    "cls": ClassificationTrainer,
    "reg": RegressionTrainer,
    "seg": SegmentationTrainer,
}


def get_task_from_dataset_name(dataset_name):
    prefix = dataset_name[:3].upper()
    if prefix == "CLS":
        return "cls"
    elif prefix == "REG":
        return "reg"
    elif prefix == "SEG":
        return "seg"
    else:
        raise ValueError(f"Cannot infer task from dataset name: {dataset_name}")


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def build_cpu_transforms(crop_size, training):
    tforms = []
    if training:
        tforms.append(Torch_CropPad(patch_size=crop_size))
    else:
        tforms.append(Torch_CenterCrop(target_size=crop_size))
    return transforms.Compose(tforms) if tforms else None


def build_model(config, task, n_modalities, n_classes):
    variant = config.get("model_variant", "small")
    builder = MODEL_BUILDERS[variant]
    return builder(
        med_in_channels=n_modalities,
        task=task,
        classes=n_classes,
        minibatch_size=config.get("minibatch_size", -1),
        patch_size=config.get("patch_size", 16),
        num_register_tokens=config.get("num_register_tokens", 4),
        lora=config.get("lora", False),
    )


def build_datamodule(config, task, train_files, val_files, crop_size):
    train_cpu_transforms = build_cpu_transforms(crop_size, training=True)
    val_cpu_transforms = build_cpu_transforms(crop_size, training=False)

    if task == "seg":
        return SegDataModule(
            batch_size=config.get("batch_size", 2),
            num_workers=config.get("num_workers", 8),
            train_split=train_files,
            val_split=val_files,
            train_transforms=train_cpu_transforms,
            val_transforms=val_cpu_transforms,
        )
    else:
        return ClsRegDataModule(
            batch_size=config.get("batch_size", 8),
            num_workers=config.get("num_workers", 8),
            train_split=train_files,
            val_split=val_files,
            train_transforms=train_cpu_transforms,
            val_transforms=val_cpu_transforms,
        )


def build_trainer_module(config, task, model):
    gpu_transforms = default_aug(ndim=3)
    norm_transforms = default_norm(normalize=True)
    trainer_class = TRAINER_CLASSES[task]
    return trainer_class(
        model=model,
        config=config,
        gpu_transforms=gpu_transforms,
        norm_transforms=norm_transforms,
    )


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    config_path = Path(get_config_path()) / args.config
    config = load_yaml_config(config_path)

    dataset_name = config.get("dataset_name", "CLS002_FOMO26_Infarct")
    task = get_task_from_dataset_name(dataset_name)
    crop_size = config.get("crop_size", [378, 378, 32])

    n_modalities, n_classes = get_dataset_metadata(dataset_name)
    train_files, val_files = load_fold(dataset_name, args.fold)

    model = build_model(config, task, n_modalities, n_classes)
    datamodule = build_datamodule(config, task, train_files, val_files, crop_size)
    trainer_module = build_trainer_module(config, task, model)

    run_name = get_run_name(dataset_name, config.get("model_variant", "small"), config.get("lora", False))

    models_path = get_models_path()
    results_path = get_results_path()

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(models_path, run_name, f"fold{args.fold}"),
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    pl_trainer = pl.Trainer(
        max_steps=config.get("max_steps", 100000),
        default_root_dir=os.path.join(results_path, run_name, f"fold{args.fold}"),
        callbacks=[checkpoint_callback, lr_monitor],
        precision=config.get("precision", "b16-mixed"),
        accelerator="auto",
        devices=config.get("devices", "auto"),
        strategy=config.get("strategy", "auto"),
        log_every_n_steps=config.get("log_every_n_steps", 50),
        gradient_clip_val=config.get("gradient_clip_val", None),
    )

    pl_trainer.fit(trainer_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
