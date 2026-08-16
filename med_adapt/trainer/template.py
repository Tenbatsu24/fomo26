"""Generic trainer template with registry-based losses and metrics.

Provides shared infrastructure (pretrained loading, optimizer/scheduler setup,
logging helpers, training/validation/test scaffolding) that all task trainers
inherit. Subclasses override ``__init__``, ``batch_to_loss``, and/or
``test_step`` for task-specific behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union, Any

import torch
import lightning as pl
import torchmetrics
from loguru import logger
from lightning import Callback
from ml_collections import ConfigDict

from med_adapt.loss import get_loss
from med_adapt.metric import get_metric
from med_adapt.augs import default_disable_aug
from med_adapt.optim import init_optims_from_config
from med_adapt.scheduling import Schedule, Scheduler
from med_adapt.utils import get_models_path, load_lora_state_dict, mark_trainable


def apply_lr_multiplier(loc, step, sched):
    return loc.get("lr_multiplier", 1.0) * sched(step)


def apply_wd_multiplier(loc, step, sched):
    return loc.get("wd_multiplier", 1.0) * sched(step)


class TemplateTrainer(pl.LightningModule):
    """Base trainer with shared infrastructure.

    Subclasses set ``config["loss"]`` and ``config["metrics"]`` in their
    ``__init__`` and override ``batch_to_loss`` and/or ``test_step`` as needed.
    """

    def __init__(
        self,
        config: ConfigDict,
        model: torch.nn.Module,
        gpu_augmentations=default_disable_aug,
        normalisation: torch.nn.Module | None = None,
    ):
        super().__init__()

        # Resolve num_classes from config or model head
        num_classes = config.num_classes
        if num_classes is None and hasattr(model, "head"):
            out_features = getattr(model.head, "out_features", None)
            if out_features is not None:
                num_classes = out_features
        config["num_classes"] = num_classes

        self.config = config
        self.gpu_aug = gpu_augmentations
        self.normalisation = normalisation
        self.criterion = self.make_criterion()
        self.num_classes: int = self.config.num_classes

        metrics = self.make_metrics()
        metrics = torchmetrics.MetricCollection(metrics)

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.model = model
        self._load_pretrained()

        mark_trainable(self.model, additional_keys=self.model.additional_trainable())

        self.optims, self.scheduler = self.make_opt_sched()

    # ------------------------------------------------------------------
    # Pretrained checkpoint loading
    # ------------------------------------------------------------------

    def _load_pretrained(self) -> None:
        """Load a pretrained checkpoint if configured."""
        ckpt_path = self.config.pretrained.checkpoint
        if ckpt_path is None:
            logger.info("[TemplateTrainer] No checkpoint specified.")
            return

        ckpt_path = Path(get_models_path()) / ckpt_path
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        lora = self.config.model.lora
        if lora:
            missing, unexpected = load_lora_state_dict(
                self.model,
                state_dict,
                strict=False,
                ignore_loading=self.model.do_not_load(),
            )
        else:
            if to_ignore := self.model.do_not_load():
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if all(ig not in k for ig in to_ignore)
                }
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        logger.info(
            "[TemplateTrainer] Loaded checkpoint from {path}. Missing: {miss}, Unexpected: {unexp}",
            path=ckpt_path,
            miss=missing,
            unexp=unexpected,
        )

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------
    def get_modules_for_opt(self):
        return [self.model]

    def make_opt_sched(self) -> tuple[list[torch.optim.Optimizer], Scheduler]:
        """Create optimizers and scheduler from config."""
        opt = init_optims_from_config(self.config, self.get_modules_for_opt())

        scheduler = Scheduler()
        sched_config = self.config.scheduler

        for key, sched in sched_config:
            if key == "lr":
                for group_num in range(len(opt.param_groups)):
                    scheduler.add(
                        opt.param_groups[group_num],
                        key,
                        Schedule.parse(sched),
                        apply_lr_multiplier,
                    )
            if key == "weight_decay":
                for group_num in range(len(opt.param_groups)):
                    scheduler.add(
                        opt.param_groups[group_num],
                        key,
                        Schedule.parse(sched),
                        apply_wd_multiplier,
                    )
        return [opt], scheduler

    # ------------------------------------------------------------------
    # Loss / metrics factories
    # ------------------------------------------------------------------

    def make_criterion(self):
        """Create the loss function. Override in subclass."""
        loss_cfg = self.config.loss
        return get_loss(loss_cfg["type"], **dict(loss_cfg.get("params", {})))

    def make_metrics(self):
        """Create the metrics. Override in subclass."""
        metrics_cfg = self.config.metrics
        if metrics_cfg is None:
            return {}

        return {
            short_name: get_metric(
                m["type"], **{k: v for k, v in m.items() if k != "type"}
            )
            for short_name, m in metrics_cfg.items()
        }

    # ------------------------------------------------------------------
    # Forward / batch helpers
    # ------------------------------------------------------------------
    def preprocess_batch(self, batch, train: bool) -> tuple[Any, Any]:
        if train and self.gpu_aug is not None:
            batch = self.gpu_aug(batch)
        if self.normalisation is not None:
            batch = self.normalisation(batch)

        image, label = batch["image"], batch["label"]
        return image, label

    def forward(self, x):
        return self.model(x)

    def batch_to_loss(self, batch, train=False):
        image, label = self.preprocess_batch(batch, train)

        outputs = self(image)

        if isinstance(outputs, list):
            # Deep supervision: weighted sum of per-block losses
            num_preds = len(outputs)
            total_loss = None
            for i, pred in enumerate(outputs):
                weight = 2 ** (i - (num_preds - 1))
                if isinstance(pred, list):
                    # Regression: list of per-class tensors
                    pred_loss = sum(self.criterion(p, label) for p in pred) / len(pred)
                else:
                    pred_loss = self.criterion(pred, label)
                if total_loss is None:
                    total_loss = weight * pred_loss
                else:
                    total_loss = total_loss + weight * pred_loss
            logits = outputs[-1]
        else:
            logits = (
                outputs
                if isinstance(outputs, torch.Tensor)
                else outputs.get("logits", outputs)
            )
            total_loss = self.criterion(logits, label)

        return total_loss, (logits, label)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_loss(self, loss, prefix, prog_bar, on_epoch, on_step):
        if isinstance(loss, dict):
            for key, value in loss.items():
                self.log(
                    f"{prefix}/{key}",
                    value.detach(),
                    prog_bar=prog_bar,
                    on_epoch=on_epoch,
                    on_step=on_step,
                    sync_dist=on_epoch,
                )
            return loss["loss"]
        else:
            self.log(
                f"{prefix}/loss",
                loss,
                prog_bar=prog_bar,
                on_epoch=on_epoch,
                on_step=on_step,
                sync_dist=on_epoch,
            )
            return loss

    def log_metrics(self, batch_metrics):
        metric_dict = {
            k: torch.mean(v) if len(v.shape) > 0 else v
            for k, v in batch_metrics.items()
            if not torch.all(torch.isnan(v))
        }
        self.log_dict(metric_dict, prog_bar=True, on_epoch=False, on_step=True)

    # ------------------------------------------------------------------
    # Training / validation / test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, for_metrics = self.batch_to_loss(batch, train=True)
        loss = self.log_loss(
            loss, prefix="train", prog_bar=True, on_epoch=False, on_step=True
        )
        if for_metrics:
            pred, gt = for_metrics
            with torch.no_grad():
                try:
                    batch_metrics = self.train_metrics(pred, gt)
                    self.log_metrics(batch_metrics)
                except Exception as e:
                    logger.error(
                        f"Error computing training metrics {pred.shape=}, {gt.shape=}: {e}"
                    )
        return loss["loss"] if isinstance(loss, dict) else loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, for_metrics = self.batch_to_loss(batch, train=False)
        loss = self.log_loss(
            loss, prefix="val", prog_bar=True, on_epoch=True, on_step=False
        )
        if for_metrics:
            pred, gt = for_metrics
            try:
                self.val_metrics.update(pred, gt)
            except Exception as e:
                logger.error(f"Error computing validation metrics: {e}")
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """Run test evaluation. Override in subclass for sliding-window inference."""
        loss, for_metrics = self.batch_to_loss(batch, train=False)
        loss = self.log_loss(
            loss, prefix="test", prog_bar=True, on_epoch=True, on_step=False
        )
        if for_metrics:
            pred, gt = for_metrics
            try:
                self.test_metrics.update(pred, gt)
            except Exception as e:
                logger.error(f"Error computing test metrics: {e}")
        return loss

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            self.test_metrics.compute(),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.test_metrics.reset()

    # ------------------------------------------------------------------
    # Optimizer configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return self.optims, []

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.scheduler]
