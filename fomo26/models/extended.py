from typing import Literal
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from fomo26.models.base import ViTv2
from fomo26.adapter import InputChannelAdapter, AttentionPooling
from fomo26.layers import Block, ScaleBlock, MemEffAttention, LoRAMemEffAttention


class ViTv2Adaption(ViTv2):

    def __init__(
        self,
        med_in_channels: int,
        task: Literal["reg", "cls", "seg"],
        classes,
        *args,
        **kwargs,
    ):
        super(ViTv2Adaption, self).__init__(*args, **kwargs)

        self.task = task

        if task in ["reg", "cls", "seg"]:
            self.input_adapter = InputChannelAdapter(in_channels=med_in_channels)
        else:
            self.input_adapter = nn.Conv1d(med_in_channels, 3, 1, bias=False)

        if task in ["reg", "cls"]:
            self.attn_pool = AttentionPooling(self.embed_dim)
            if task == "cls":
                self.head = nn.Linear(self.embed_dim, classes)
            elif task == "seg":
                self.head = nn.Linear(self.embed_dim, 1)
            else:
                self.head = nn.Identity()
        else:
            self.upscale = ScaleBlock(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, classes)

    def forward(self, x, masks=None, last_self_attention=False, **kwargs):
        b, c, d, h, w = x.shape

        adapted_x = self.input_adapter(x)

        # go from 3d to folding depth into batch axes
        reshaped_x = rearrange(adapted_x, "b c d h w -> (b d) c h w")
        out = super().forward(
            reshaped_x, masks=masks, last_self_attention=last_self_attention, **kwargs
        )

        patch_latents = out["patch_latents"]
        if self.task in ["reg", "cls"]:
            reshaped_out = rearrange(patch_latents, "(b d) n c -> b (d n) c", b=b, d=d)
            attnended = self.attn_pool(reshaped_out)
            return self.head(attnended)
        elif self.task == "seg":
            # make spatial
            patch_size = self.patch_size
            hp, wp = h // patch_size, w // patch_size

            spatial = rearrange(
                patch_latents, "(b d) (hp wp) c -> b d c hp wp", b=b, d=d, hp=hp, wp=wp
            )

            # apply scale block
            scaled = self.upscale(spatial)  # (b d) c h w

            # final classifier
            return self.head(scaled)
        else:
            out_latents = out["latents"]  # (b d) c

            # reshaped latents
            reshaped_latents = rearrange(out_latents, "(b d) c -> b d c", b=b, d=d)

            # pool along depth in some clever but not learnt way
            pooled_out = reshaped_latents.mean(dim=-2)

            return pooled_out


def vitv2_tiny(patch_size=16, num_register_tokens=4, lora=False, **kwargs):
    model = ViTv2Adaption(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vitv2_small(patch_size=16, num_register_tokens=4, lora=False, **kwargs):
    model = ViTv2Adaption(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vitv2_base(patch_size=16, num_register_tokens=4, lora=False, **kwargs):
    model = ViTv2Adaption(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vitv2_large(patch_size=16, num_register_tokens=4, lora=False, **kwargs):
    model = ViTv2Adaption(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
