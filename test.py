import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from med_adapt.registry import STORE
from med_adapt.utils.naming import get_run_name
from med_adapt.utils.config import get_config, get_logger
from med_adapt.utils.paths import get_results_path, get_data_path
from med_adapt.datasets import build_dataloaders

from main import build_cpu_transforms, build_model, round_up_to_multiple

logger = get_logger(__name__)

N_FOLDS = 5
SUPPORTED_TASKS = ("classification", "regression")

# "best" direction for each trackable metric; used when --metric is given.
METRIC_MODES = {"loss": "min", "auroc": "max", "l2": "min"}
_CKPT_METRIC_RE = re.compile(r"^step=(\d+)-val_(\w+)=([0-9.]+)\.ckpt$")


def find_latest_version_dir(fold_dir: Path) -> Optional[Path]:
    """Return the highest-numbered version_* directory under fold_dir."""
    if not fold_dir.exists():
        return None
    version_dirs = sorted(
        (p for p in fold_dir.glob("version_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    return version_dirs[-1] if version_dirs else None


def find_checkpoint_in_version(
    version_dir: Path, metric: Optional[str]
) -> Optional[Path]:
    """Pick a checkpoint file inside a version dir.

    metric=None -> "last.ckpt". Otherwise, parse filenames of the form
    "step={step}-val_{metric}={value}.ckpt", keep only those matching
    `metric`, and return the one with the best value (per METRIC_MODES),
    breaking ties by the highest step.
    """
    if metric is None:
        ckpt_path = version_dir / "last.ckpt"
        return ckpt_path if ckpt_path.exists() else None

    if metric not in METRIC_MODES:
        raise ValueError(
            f"Unknown metric={metric!r}; expected one of {list(METRIC_MODES)} or None"
        )

    candidates = []
    for p in version_dir.glob("*.ckpt"):
        m = _CKPT_METRIC_RE.match(p.name)
        if m is None:
            continue
        step, name, value = m.groups()
        if name != metric:
            continue
        candidates.append((int(step), float(value), p))

    if not candidates:
        return None

    mode = METRIC_MODES[metric]
    sort_key = (lambda c: (c[1], c[0])) if mode == "max" else (lambda c: (-c[1], c[0]))
    return max(candidates, key=sort_key)[2]


def find_checkpoint_for_fold(
    results_path: Path, run_name: str, fold: int, metric: Optional[str]
) -> Optional[Path]:
    """Autodiscover a fold's checkpoint from its latest Lightning version dir.

    Mirrors the deterministic naming used in main.py:
        {results_path}/{run_name}/fold_{fold}/version_{n}/<checkpoint>.ckpt
    """
    fold_dir = results_path / run_name / f"fold_{fold}"
    version_dir = find_latest_version_dir(fold_dir)
    if version_dir is None:
        return None
    return find_checkpoint_in_version(version_dir, metric)


def load_model_from_checkpoint(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    if any(k.startswith("model.") for k in state_dict):
        state_dict = {
            k[len("model.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"[load_model_from_checkpoint] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[load_model_from_checkpoint] Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model


@torch.no_grad()
def run_inference(model, dataloader, task: str, device: torch.device) -> dict:
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="inference", leave=False):
        image = batch["image"].to(device)
        target = batch["label"]

        output = model(image)

        if isinstance(output, list):
            output = output[-1]

        if task == "classification":
            output = torch.softmax(output, dim=1)

        all_preds.append(output.detach().cpu())
        all_targets.append(target.detach().cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return {"preds": preds, "targets": targets}


def save_fold_outputs(out_dir: Path, fold: int, outputs: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"fold_{fold}_outputs.npz", **outputs)


def load_fold_outputs(out_dir: Path, fold: int) -> Optional[dict]:
    path = out_dir / f"fold_{fold}_outputs.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_classification_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    pred_labels = preds.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(targets, pred_labels),
        "precision": precision_score(
            targets, pred_labels, average="macro", zero_division=0
        ),
        "recall": recall_score(targets, pred_labels, average="macro", zero_division=0),
        "f1": f1_score(targets, pred_labels, average="macro", zero_division=0),
    }

    try:
        n_classes = preds.shape[1]
        if n_classes > 2:
            metrics["auroc"] = roc_auc_score(
                targets, preds, multi_class="ovr", average="macro"
            )
        else:
            metrics["auroc"] = roc_auc_score(targets, preds[:, 1])
    except ValueError as e:
        logger.warning(f"[compute_classification_metrics] Could not compute AUROC: {e}")
        metrics["auroc"] = float("nan")

    return metrics


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    correlation, _ = pearsonr(targets, preds)

    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": mean_absolute_error(targets, preds),
        "r2": r2_score(targets, preds),
        "correlation": correlation,
    }


METRIC_FNS = {
    "classification": compute_classification_metrics,
    "regression": compute_regression_metrics,
}


def main():
    parser = argparse.ArgumentParser(description="med_adapt standalone evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help=f"Subset of folds to evaluate (default: all {N_FOLDS} folds)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=list(METRIC_MODES),
        default=None,
        help="Which tracked metric to select the checkpoint by (default: last.ckpt)",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only run inference and cache fold outputs; skip metric computation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if cached fold outputs already exist",
    )
    args = parser.parse_args()

    config = get_config(args.config)
    folds = args.folds if args.folds is not None else list(range(N_FOLDS))

    dataset_name = config.data.dataset_name
    dataset_class = STORE.get("datasets", dataset_name)
    task = dataset_class.TASK_TYPE

    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"task={task} is not supported by this script (only {SUPPORTED_TASKS})."
        )

    n_modalities = dataset_class.NUM_MODALITIES
    n_classes = dataset_class.NUM_CLASSES
    config["num_classes"] = n_classes
    config["n_modalities"] = n_modalities

    data_root = str(get_data_path())
    seed = config.seed

    crop_size = config.data.crop_size
    test_time_resize = config.data.test_time_resize

    if isinstance(crop_size, str) and crop_size == "median":
        crop_size = round_up_to_multiple(dataset_class.median_resolution(), multiple=8)
    else:
        crop_size = tuple(crop_size) if crop_size is not None else None

    if isinstance(test_time_resize, str) and test_time_resize == "median":
        test_time_resize = round_up_to_multiple(
            dataset_class.median_resolution(), multiple=8
        )
    else:
        test_time_resize = (
            tuple(test_time_resize) if test_time_resize is not None else None
        )

    config.data.crop_size = crop_size
    config.data.test_time_resize = test_time_resize

    eval_cpu_transforms = build_cpu_transforms(
        crop_size, stage="test", task=task, test_time_resize=test_time_resize
    )

    run_name = get_run_name(
        dataset_name, config.model.size, config.model.variant, config.model.lora
    )
    results_path = get_results_path()
    out_dir = results_path / run_name / "eval_outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in folds:
        cached = None if args.force else load_fold_outputs(out_dir, fold)
        if cached is not None:
            logger.info(f"[fold {fold}] Cached outputs found, skipping inference.")
            continue

        ckpt_path = find_checkpoint_for_fold(results_path, run_name, fold, args.metric)
        if ckpt_path is None:
            logger.warning(
                f"[fold {fold}] No checkpoint found under "
                f"{results_path / run_name / f'fold_{fold}'} (metric={args.metric}); skipping."
            )
            continue
        logger.info(f"[fold {fold}] Loading checkpoint: {ckpt_path}")

        _, val_dl, _ = build_dataloaders(
            dataset_class=dataset_class,
            root=data_root,
            fold=fold,
            seed=seed,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            train_transforms=eval_cpu_transforms,
            val_transforms=eval_cpu_transforms,
            test_transforms=eval_cpu_transforms,
            val_drop_last=False,
            resample_spacing=config.data.resample_spacing,
        )

        model = build_model(
            config, task, n_modalities, n_classes, config.model.lora, True
        )
        model = load_model_from_checkpoint(model, ckpt_path, device)

        outputs = run_inference(model, val_dl, task, device)
        save_fold_outputs(out_dir, fold, outputs)
        logger.info(
            f"[fold {fold}] Saved outputs -> {out_dir / f'fold_{fold}_outputs.npz'}"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.skip_metrics:
        return

    all_preds, all_targets = [], []
    for fold in folds:
        data = load_fold_outputs(out_dir, fold)
        if data is None:
            logger.warning(
                f"[fold {fold}] No cached outputs found; excluding from pooled metrics."
            )
            continue
        all_preds.append(data["preds"])
        all_targets.append(data["targets"])

    if not all_preds:
        logger.warning("No fold outputs available; cannot compute metrics.")
        return

    pooled_preds = np.concatenate(all_preds, axis=0)
    pooled_targets = np.concatenate(all_targets, axis=0)

    metrics = METRIC_FNS[task](pooled_preds, pooled_targets)

    logger.info(
        f"Pooled metrics over {len(pooled_targets)} samples across folds {folds}:"
    )
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    metrics_path = out_dir / "pooled_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    logger.info(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
