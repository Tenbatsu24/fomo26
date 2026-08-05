import logging
from typing import Literal
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from fomo26.models.base import ViTv2
from fomo26.adapter import InputChannelAdapter, AttentionPooling
from fomo26.layers import Block, ScaleBlock, MemEffAttention, LoRAMemEffAttention

LOGGER = logging.getLogger(__name__)


class ViTv2Adaption(ViTv2):

    def __init__(
        self,
        med_in_channels: int,
        task: Literal["regression", "classification", "segmentation", "none"],
        classes: int,
        *args,
        volume_size=None,
        volume_patch_size=None,
        **kwargs,
    ):
        super(ViTv2Adaption, self).__init__(*args, **kwargs)
        if volume_size is not None or volume_patch_size is not None:
            LOGGER.warning(
                "volume_size=%s and volume_patch_size=%s are not used in this 2D adaptation. Ignored.",
                volume_size,
                volume_patch_size,
            )

        self.task = task

        if task in ["regression", "classification", "segmentation"]:
            self.input_adapter = InputChannelAdapter(in_channels=med_in_channels)
        else:
            self.input_adapter = nn.Conv3d(med_in_channels, 3, 1, bias=False)

        if task in ["regression", "classification", "none"]:
            self.attn_pool = AttentionPooling(self.embed_dim)
            if task == "classification":
                self.head = nn.Linear(self.embed_dim, classes)
            elif task == "regression":
                self.head = nn.Linear(self.embed_dim, 1)
            else:
                self.head = nn.Identity()
        else:
            self.upscale = nn.Sequential(
                ScaleBlock(self.embed_dim),
                ScaleBlock(self.embed_dim // 2),
                ScaleBlock(self.embed_dim // 4),
                ScaleBlock(self.embed_dim // 8),
            )
            self.head = nn.Linear(self.embed_dim // 16, classes)

    def forward(self, x, masks=None, last_self_attention=False, **kwargs):
        b, c, h, w, d = x.shape

        adapted_x = self.input_adapter(x)

        # go from 3d to folding depth into batch axes
        reshaped_x = rearrange(adapted_x, "b c ... d -> (b d) c ...")

        out = super().forward(
            reshaped_x,
            masks=masks,
            last_self_attention=last_self_attention,
            **kwargs,
        )
        latents = out["latent"]
        patch_latents = out["patch_latent"]

        if self.task in ["regression", "classification", "none"]:
            reshaped_out = rearrange(patch_latents, "(b d) n c -> b (d n) c", b=b)
            attended = self.attn_pool(reshaped_out)
            return self.head(attended)
        elif self.task == "segmentation":
            # make spatial
            hp, wp = h // self.patch_size, w // self.patch_size

            spatial = rearrange(patch_latents, "b (hp wp) c -> b c hp wp", hp=hp, wp=wp)

            upscaled = self.upscale(spatial)

            # final classifier
            rearranged = rearrange(upscaled, "b c ... -> b ... c")
            logits = rearrange(self.head(rearranged), "(b d) ... c -> b c ... d", b=b)
            return F.interpolate(
                logits, size=(h, w, d), mode="trilinear", align_corners=False
            )
        else:
            # reshaped latents
            pooled = rearrange(latents, "(b d) c -> b d c", b=b).mean(dim=-2)

            return pooled


def vitv2_a_2d_tiny(lora=False, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption(
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=4,
        **kwargs,
    )
    return model


def vitv2_a_2d_small(lora=False, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=4,
        **kwargs,
    )
    return model


def vitv2_a_2d_base(lora=False, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption(
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=4,
        **kwargs,
    )
    return model


def vitv2_a_2d_large(lora=False, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 1e-5
    model = ViTv2Adaption(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=4,
        **kwargs,
    )
    return model


