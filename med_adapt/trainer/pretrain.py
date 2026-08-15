from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union, Any

import torch
import lightning as pl
import torch.nn.functional as F

from einops import rearrange
from loguru import logger
from lightning import Callback
from ml_collections import ConfigDict

from med_adapt.utils import get_models_path
from med_adapt.augs import default_disable_aug
from med_adapt.optim import init_optims_from_config
from med_adapt.scheduling import Schedule, Scheduler


def apply_lr_multiplier(loc, step, sched):
    return loc.get("lr_multiplier", 1.0) * sched(step)


def apply_wd_multiplier(loc, step, sched):
    return loc.get("wd_multiplier", 1.0) * sched(step)


class PretrainTrainer(pl.LightningModule):

    def __init__(
        self,
        config: ConfigDict,
        model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        gpu_augmentations=default_disable_aug,
        normalisation: torch.nn.Module | None = None,
    ):
        super().__init__()

        self.config = config
        self.gpu_aug = gpu_augmentations
        self.normalisation = normalisation
        self.distill_from = self.config.distill_from

        self.model = model
        self.teacher_model = teacher_model

        self._load_pretrained()

        self.optims, self.scheduler = self.make_opt_sched()

        # Will be moved to the correct device in on_train_start via self.device
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _load_pretrained(self) -> None:
        """Load a pretrained checkpoint if configured."""
        ckpt_path = self.config.pretrained.checkpoint
        if ckpt_path is None:
            logger.error(
                f"[TemplateTrainer] No checkpoint specified. Nothing to distill from: {ckpt_path=}"
            )
            raise ValueError()

        ckpt_path = Path(get_models_path()) / ckpt_path
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = self.teacher_model.load_state_dict(
            state_dict, strict=False
        )

        logger.info(
            "[TemplateTrainer] Loaded checkpoint from {path}. Missing: {miss}, Unexpected: {unexp}",
            path=ckpt_path,
            miss=missing,
            unexp=unexpected,
        )

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

    def preprocess_batch(self, batch, train: bool) -> tuple[Any, Any]:
        if train and self.gpu_aug is not None:
            batch = self.gpu_aug(batch)

        if self.normalisation is not None:
            batch = self.normalisation(batch)

        image = batch["image"]
        return image

    def _teacher_forward(self, volume: torch.Tensor):
        b, c, *_ = volume.shape

        vol_flat = rearrange(volume, "b c h w d -> (b d) c h w")

        ch_min = vol_flat.min(dim=1, keepdim=True).values
        ch_max = vol_flat.max(dim=1, keepdim=True).values
        denom = ch_max - ch_min
        denom[denom == 0] = 1.0
        vol_norm = (vol_flat - ch_min) / denom

        vol_norm = (vol_norm - self.imagenet_mean) / self.imagenet_std

        intermediates = self.teacher_model(vol_norm, distill_from=self.distill_from)

        outputs = []
        for layer_idx, (cls_token, patch_token) in enumerate(intermediates):
            cls_token = rearrange(cls_token, "(b d) c -> b c d", b=b).mean(dim=-1)
            spatial = rearrange(patch_token, "(b d) c h_p w_p -> b c h_p w_p d", b=b)
            outputs.append((cls_token, spatial))

        return outputs

    def _distill_loss(self, teacher_out, student_out, recon=None, volume=None):
        cls_total, token_total = 0.0, 0.0
        n = len(teacher_out)

        for (t_c, t_p), (s_c, s_p, *_) in zip(teacher_out, student_out):
            s_interp = F.interpolate(
                s_p,
                size=(t_p.shape[2], t_p.shape[3], t_p.shape[4]),
                mode="trilinear",
                align_corners=False,
            )
            cls_total += -torch.cosine_similarity(t_c, s_c, dim=1).mean()
            token_total += (
                -torch.cosine_similarity(t_p, s_interp, dim=1)
                .mean(dim=(1, 2, 3))
                .mean()
            )

        if recon is not None and volume is not None:
            huber = F.huber_loss(recon, volume)
        else:
            huber = torch.zeros(1, device=self.device)

        return {
            "loss": (2 - 2 * ((0.2 * cls_total + 0.8 * token_total) / n)) + huber,
            "cls_cos": 2 - 2 * (cls_total / n),
            "token_cos": 2 - 2 * (token_total / n),
            "huber": huber,
        }

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def batch_to_loss(self, batch, train=False):
        with torch.no_grad():
            teacher_outs = self._teacher_forward(batch["image"])

        image = self.preprocess_batch(batch, train)

        student_outs, recon = self(image, distill_from=self.distill_from)

        return self._distill_loss(teacher_outs, student_outs, recon, image)

    def on_fit_start(self) -> None:
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

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

    def training_step(self, batch, batch_idx):
        loss = self.batch_to_loss(batch, train=True)
        loss = self.log_loss(
            loss, prefix="train", prog_bar=True, on_epoch=False, on_step=True
        )
        return loss["loss"] if isinstance(loss, dict) else loss

    def validation_step(self, batch, batch_idx):
        loss = self.batch_to_loss(batch, train=False)
        loss = self.log_loss(
            loss, prefix="val", prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def configure_optimizers(self):
        return self.optims, []

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.scheduler]
