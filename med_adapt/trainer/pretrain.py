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
from med_adapt.utils.masking import generate_masks
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

        self.mask_enabled = self.config.model.use_mask

        # Mask generation hyper-parameters (safe defaults when absent)
        mask_cfg = getattr(self.config.model, "mask", None) or {}
        self.mask_prob = mask_cfg.get("mask_prob", 0.75)
        self.per_sample_range = tuple(mask_cfg.get("per_sample_range", [0.05, 0.1]))

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

    def _teacher_forward(self, volume: torch.Tensor, chunk_size: int = 16):
        b, c, h, w, d = volume.shape
        layer_outputs = (
            None  # list-of-lists: [layer][chunk] -> (cls_chunk, spatial_chunk)
        )

        for d_start in range(0, d, chunk_size):
            d_end = min(d_start + chunk_size, d)
            vol_chunk = volume[..., d_start:d_end]  # b c h w d_chunk
            d_chunk = d_end - d_start

            vol_flat = rearrange(vol_chunk, "b c h w d -> (b d) c h w")
            ch_min = vol_flat.min(dim=1, keepdim=True).values
            ch_max = vol_flat.max(dim=1, keepdim=True).values
            denom = ch_max - ch_min
            denom[denom == 0] = 1.0
            vol_norm = (vol_flat - ch_min) / denom
            vol_norm = (vol_norm - self.imagenet_mean) / self.imagenet_std

            intermediates = self.teacher_model(vol_norm, distill_from=self.distill_from)

            if layer_outputs is None:
                layer_outputs = [[] for _ in intermediates]

            for layer_idx, (cls_token, patch_token) in enumerate(intermediates):
                # b=b, d=d_chunk pins the unflatten to THIS chunk's own (b d) ordering,
                # so samples from different chunks never get interleaved.
                cls_token = rearrange(cls_token, "(b d) c -> b c d", b=b, d=d_chunk)
                spatial = rearrange(
                    patch_token, "(b d) c h_p w_p -> b c h_p w_p d", b=b, d=d_chunk
                )
                layer_outputs[layer_idx].append((cls_token, spatial))

        outputs = []
        for chunks in layer_outputs:
            cls_chunks, spatial_chunks = zip(*chunks)
            # chunks were produced in increasing d_start order, so concatenating
            # along dim=-1 reconstructs the original depth ordering exactly.
            cls_full = torch.cat(cls_chunks, dim=-1).mean(dim=-1)
            spatial_full = torch.cat(spatial_chunks, dim=-1)
            outputs.append((cls_full, spatial_full))

        return outputs

    def _distill_loss(
        self, teacher_out, student_out, recon=None, volume=None, mask=None
    ):
        cls_total, token_total = 0.0, 0.0
        n = len(teacher_out)

        # Build voxel-space keep mask once when masking is active, so both
        # cls and token cosine terms share the same spatial weighting.
        mask_3d = None
        mask_up = None

        if mask is not None:
            mask_3d = mask.unflatten(1, self.model.patch_embed.patches_resolution)
            mask_up = torch.repeat_interleave(mask_3d, self.model.patch_size[0], dim=1)
            mask_up = torch.repeat_interleave(mask_up, self.model.patch_size[1], dim=2)
            mask_up = torch.repeat_interleave(mask_up, self.model.patch_size[2], dim=3)

        for (t_c, t_p), (s_c, s_p, *_) in zip(teacher_out, student_out):
            t_interp = F.interpolate(
                t_p,
                size=(s_p.shape[2], s_p.shape[3], s_p.shape[4]),
                mode="trilinear",
                align_corners=False,
            )
            t_cn, s_cn = F.normalize(t_c, p=2, eps=1e-6, dim=1), F.normalize(
                s_c, p=2, eps=1e-6, dim=1
            )
            cls_total += 2 - 2 * (t_cn * s_cn).sum(dim=1).mean()
            t_pn, s_pn = F.normalize(t_interp, p=2, eps=1e-6, dim=1), F.normalize(
                s_p, p=2, eps=1e-6, dim=1
            )

            if mask_3d is None:
                token_total += (
                    2 - 2 * (t_pn * s_pn).sum(dim=1).mean(dim=(2, 3, 4)).mean()
                )
            else:
                cos_map = (t_pn * s_pn).sum(dim=1)  # (B, D', H', W')
                masked_cos_sum = (cos_map * mask_3d).sum(dim=(1, 2, 3))
                num_masked_tok = mask_3d.sum(dim=(1, 2, 3))
                has_masked_tok = num_masked_tok > 0
                denom = num_masked_tok.clamp(min=1)

                token_loss_per_sample = torch.where(
                    has_masked_tok,
                    2 - 2 * (masked_cos_sum / denom),
                    torch.zeros_like(masked_cos_sum),
                )
                n_masked_tok_samples = has_masked_tok.sum().clamp(min=1)
                token_total += token_loss_per_sample.sum() / n_masked_tok_samples

        loss_dict = {
            "loss": (0.2 * cls_total + 0.8 * token_total) / n,
            "cls_cos": cls_total / n,
            "token_cos": token_total / n,
        }

        if recon is not None and mask is not None:
            # loss_mask == 1 on dropped (masked) regions
            masked_recon = recon * mask_up
            masked_volume = volume * mask_up
            # Per-element huber, then normalise per batch element by its
            # own masked voxel count, sum across the batch, and average.
            huber_per_elem = F.huber_loss(masked_recon, masked_volume, reduction="none")

            num_masked = mask_up.sum(dim=(1, 2, 3))
            has_masked = num_masked > 0
            num_masked_denom = num_masked.clamp(min=1)

            huber_per_elem = (
                huber_per_elem.sum(dim=(2, 3, 4)).mean(dim=1) / num_masked_denom
            )
            # Average only over samples that actually had masking applied
            n_masked_samples = has_masked.sum().clamp(min=1)
            huber = huber_per_elem.sum() / n_masked_samples

            loss_dict["loss"] += huber
            loss_dict["huber"] = huber
        elif recon is not None:
            huber = F.huber_loss(recon, volume, reduction="mean")
            loss_dict["loss"] += huber
            loss_dict["huber"] = huber

        return loss_dict

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def _generate_masks(self, batch_size: int) -> torch.Tensor:
        """Generate 3-D masks on the current device."""
        H, W, D = self._batch_spatial_shape()
        ph = H // self.model.patch_size[0]
        pw = W // self.model.patch_size[1]
        pd = D // self.model.patch_size[2]
        masks = generate_masks(
            patch_resolution=(ph, pw, pd),
            number_of_samples=batch_size,
            mask_prob=self.mask_prob,
            per_sample_range=self.per_sample_range,
        )
        return masks.to(self.device)

    def _batch_spatial_shape(self) -> tuple[int, int, int]:
        """Return (H, W, D) of the current batch — cached after first call."""
        if not hasattr(self, "_spatial_shape"):
            # Peek at a dummy forward to infer spatial dims from patch_embed
            # (no actual forward needed; we read the resolved resolution).
            # Fallback: read from patch_embed patches_resolution scaled up.
            pr = self.model.patch_embed.patches_resolution
            ps = self.model.patch_size
            self._spatial_shape = (pr[0] * ps[0], pr[1] * ps[1], pr[2] * ps[2])
        return self._spatial_shape

    def batch_to_loss(self, batch, train=False):
        with torch.no_grad():
            teacher_outs = self._teacher_forward(batch["image"])

        image = self.preprocess_batch(batch, train)

        mask = None
        if self.mask_enabled and train:
            mask = self._generate_masks(image.shape[0])

        student_outs, recon = self(image, distill_from=self.distill_from, mask=mask)

        return self._distill_loss(teacher_outs, student_outs, recon, image, mask=mask)

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
