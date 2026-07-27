import os
import json
import argparse

import lightning as pl

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
from fomo26.modules.data_modules.training import SegDataModule, ClsRegDataModule
from fomo26.models import (
    vitv2_a_2d_tiny,
    vitv2_a_2d_small,
    vitv2_a_2d_base,
    vitv2_a_2d_large,
    vitv2_a_3d_tiny,
    vitv2_a_3d_small,
    vitv2_a_3d_base,
    vitv2_a_3d_large
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

    if task == "segmentation":
        return SegDataModule(
            batch_size=config.get("batch_size", 2),
            num_workers=config.get("num_workers", 2),
            dataset_class=dataset_class,
            root=data_root,
            fold=fold,
            seed=seed,
            train_transforms=train_cpu_transforms,
            val_transforms=val_cpu_transforms,
        )
    else:
        return ClsRegDataModule(
            batch_size=config.get("batch_size", 8),
            num_workers=config.get("num_workers", 2),
            dataset_class=dataset_class,
            root=data_root,
            fold=fold,
            seed=seed,
            train_transforms=train_cpu_transforms,
            val_transforms=val_cpu_transforms,
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


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    config_path = Path(get_config_path()) / args.config
    config = load_yaml_config(config_path)

    dataset_name = config.get("dataset_name", "CLS002_FOMO26_Infarct")
    dataset_class = get_dataset_class(dataset_name)
    task = dataset_class.TASK_TYPE
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
        mode="max" if task in ["seg", "cls"] else "min",
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


if __name__ == "__main__":
    main()
