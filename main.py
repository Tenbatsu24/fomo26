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
from med_adapt.augs.default import (
    default_enable_aug,
    default_disable_aug,
    default_norm,
    Torch_Resize,
)
from med_adapt.trainer import (
    ClassificationTrainer,
    RegressionTrainer,
    SegmentationTrainer,
)
from med_adapt.utils.lora import merge_all_lora

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


def build_cpu_transforms(crop_size, training, task, resize_to=None):
    """Build CPU-side crop/pad/resize transforms."""
    label_key = "label" if task == "segmentation" else None
    tforms = []
    if resize_to is not None:
        tforms.append(Torch_Resize(label_key=label_key, target_size=resize_to))
    if training:
        tforms.append(Torch_CropPad(label_key=label_key, patch_size=crop_size))
    else:
        tforms.append(Torch_Pad(label_key=label_key, patch_size=crop_size))
        tforms.append(Torch_CenterCrop(label_key=label_key, target_size=crop_size))
    return transforms.Compose(tforms) if tforms else None


def build_model(config, task, n_modalities, n_classes, lora, mea):
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
            lora=lora,
            mea=mea,
        )
    else:
        return builder(
            volume_size=tuple(config.data.crop_size),
            volume_patch_size=tuple(config.data.volume_patch_size),
            med_in_channels=n_modalities,
            task=task,
            classes=n_classes,
            lora=lora,
            mea=mea,
        )


def export_model_to_onnx(
    model,
    config,
    task,
    n_modalities,
    run_dir: Path,
    checkpoint_name: str = "model",
):
    """Export the (optionally LoRA-merged) model to ONNX.

    Merges LoRA weights into the base model when ``config.model.lora`` is
    enabled, then exports to ONNX so the final weights are baked in.
    """
    import torch.onnx

    if config.model.lora:
        n_merged = merge_all_lora(model)
        logger.info("[main] Merged {n} LoRA layers before ONNX export.", n=n_merged)

    crop_size = tuple(config.data.crop_size)
    dummy_input = torch.randn(1, n_modalities, crop_size[0], crop_size[1], crop_size[2])

    onnx_path = run_dir / f"{checkpoint_name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )
    logger.info(f"[main] ONNX model saved to {onnx_path}")
    return onnx_path


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
        "acc"
        if task == "classification"
        else "dice" if task == "segmentation" else "l2"
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

    model = build_model(
        config, task, n_modalities, n_classes, lora=config.model.lora, mea=True
    )
    crop_size = tuple(config.data.crop_size)
    test_transforms = build_cpu_transforms(
        crop_size, training=False, task=task, resize_to=config.data.resize_to
    )
    train_dl, val_dl, test_dl = build_dataloaders(
        dataset_class=dataset_class,
        root=str(get_data_path()),
        fold=fold,
        seed=seed,
        batch_size=1,
        num_workers=config.data.num_workers,
        test_transforms=test_transforms,
        resample_spacing=config.data.resample_spacing,
        resize_to=config.data.resize_to,
    )

    # Load checkpoint weights only
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)

    # Disable pretrained loading in test mode — we already loaded the test checkpoint above.
    config["pretrained"]["checkpoint"] = None
    trainer = TRAINER_CLASSES[task](config=config, model=model)

    # Export the final model to ONNX before running test.
    export_model_to_onnx(
        model, config, task, n_modalities, run_dir, checkpoint_name="best"
    )

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

    train_cpu_transforms = build_cpu_transforms(
        crop_size, training=True, task=task, resize_to=config.data.resize_to
    )
    val_cpu_transforms = build_cpu_transforms(
        crop_size, training=False, task=task, resize_to=config.data.resize_to
    )

    model = build_model(config, task, n_modalities, n_classes, config.model.lora, True)
    train_dl, val_dl, _ = build_dataloaders(
        dataset_class=dataset_class,
        root=data_root,
        fold=fold,
        seed=seed,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_transforms=train_cpu_transforms,
        val_transforms=val_cpu_transforms,
        val_drop_last=False,
        resample_spacing=config.data.resample_spacing,
        resize_to=config.data.resize_to,
    )

    if config.enable_aug:
        gpu_transforms = default_enable_aug(ndim=3)
    else:
        gpu_transforms = default_disable_aug(ndim=3)

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
        "acc"
        if task == "classification"
        else "dice" if task == "segmentation" else "l2"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(results_path, name=f"{run_name}/fold_{fold}")
    log_dir = Path(csv_logger.log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename=f"step={{step}}-val_{metric}={{val/{metric}:.3f}}",
        monitor=f"val/{metric}",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="max" if task in ["classification", "segmentation"] else "min",
        save_last=False,
        enable_version_counter=False,
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="last",
        save_last=True,
        enable_version_counter=False,
    )

    pl_trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, last_checkpoint_callback, lr_monitor],
        **config.trainer.to_dict(),
        logger=csv_logger,
    )

    pl_trainer.fit(trainer, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Export both checkpoints to ONNX.
    # Best model
    model_best = build_model(
        config,
        task,
        n_modalities,
        n_classes,
    )
    ckpt_best = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
    if isinstance(ckpt_best, dict) and "state_dict" in ckpt_best:
        ckpt_best = ckpt_best["state_dict"]
    model_best.load_state_dict(ckpt_best, strict=False)
    export_model_to_onnx(
        model_best, config, task, n_modalities, log_dir, checkpoint_name="best"
    )

    # Last model
    last_path = Path(last_checkpoint_callback.last_model_path)
    if last_path.exists():
        model_last = build_model(config, task, n_modalities, n_classes)
        ckpt_last = torch.load(last_path, map_location="cpu")
        if isinstance(ckpt_last, dict) and "state_dict" in ckpt_last:
            ckpt_last = ckpt_last["state_dict"]
        model_last.load_state_dict(ckpt_last, strict=False)
        export_model_to_onnx(
            model_last, config, task, n_modalities, log_dir, checkpoint_name="last"
        )

    logger.info("\n[main] Running test evaluation on fold {fold}...", fold=fold)
    pl_trainer.test(trainer, dataloaders=val_dl)


if __name__ == "__main__":
    main()
