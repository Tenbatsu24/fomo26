import logging

from typing import Literal
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from fomo26.models.base import ViTv2
from fomo26.adapter import PatchEmbed3D, AttentionPooling, TaskTokens
from fomo26.layers import Block, ScaleBlock, MemEffAttention, LoRAMemEffAttention

LOGGER = logging.getLogger(__name__)


class ViTv2Adaption3D(ViTv2):

    def __init__(
        self,
        volume_size: tuple[int, int, int],
        volume_patch_size: tuple[int, int, int],
        med_in_channels: int,
        task: Literal["regression", "classification", "segmentation", "none"],
        classes: int,
        *args,
        task_token: bool = False,
        task_token_insertion: Literal["beginning", "middle"] = "beginning",
        task_token_block: int = 6,
        **kwargs,
    ):
        super(ViTv2Adaption3D, self).__init__(*args, **kwargs)

        self.task = task

        self.img_size = volume_size
        self.patch_size = volume_patch_size
        self.patch_embed = PatchEmbed3D(
            img_size=volume_size,
            patch_size=volume_patch_size,
            in_chans=med_in_channels,
            embed_dim=self.embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, self.patch_embed.num_patches + self.num_tokens, self.embed_dim
            )
        )

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
                ScaleBlock(self.embed_dim, conv_type="3d"),
                ScaleBlock(self.embed_dim // 2, conv_type="3d"),
                ScaleBlock(self.embed_dim // 4, conv_type="3d"),
                ScaleBlock(self.embed_dim // 8, conv_type="3d"),
            )
            self.head = nn.Linear(self.embed_dim // 16, classes)

        # Task tokens
        self.task_token_enabled = task_token
        self.task_token_insertion = task_token_insertion
        self.task_token_block = task_token_block
        if task_token:
            num_task_tokens = classes if task in ["classification", "segmentation"] else 1
            self.task_tokens = TaskTokens(
                num_tokens=num_task_tokens,
                embed_dim=self.embed_dim,
                insertion=task_token_insertion,
            )
            LOGGER.info(
                "Task tokens enabled: %d tokens, insertion=%s, block=%d",
                num_task_tokens, task_token_insertion, task_token_block,
            )
        else:
            self.task_tokens = None

    def interpolate_pos_encoding(self, x, h, w, d):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and h == w == d:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        h0 = h // self.patch_size[0]
        w0 = w // self.patch_size[1]
        d0 = d // self.patch_size[2]

        h0 = h0 + self.interpolate_offset
        w0 = w0 + self.interpolate_offset
        d0 = d0 + self.interpolate_offset

        ph, pw, pd = self.patch_embed.patches_resolution

        sh = float(h0) / ph
        sw = float(w0) / pw
        sd = float(d0) / pd

        patch_pos_embed = patch_pos_embed.reshape(1, ph, pw, pd, dim).permute(
            0, 4, 1, 2, 3
        )

        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            scale_factor=(sh, sw, sd),
            mode="trilinear",
            align_corners=False,
        )

        assert int(h0) == patch_pos_embed.shape[-3]
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(d0) == patch_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, h, w, d = x.shape
        x = self.patch_embed(x)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, h, w, d)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        # Inject task tokens after CLS + register tokens
        if self.task_tokens is not None and self.task_token_insertion == "beginning":
            num_prefix = 1 + (self.num_register_tokens if self.register_tokens is not None else 0)
            prefix = x[:, :num_prefix]
            rest = x[:, num_prefix:]
            x = torch.cat((prefix, self.task_tokens.tokens.expand(B, -1, -1), rest), dim=1)

        return x

    def forward(self, x, masks=None, last_self_attention=False, **kwargs):
        b, c, h, w, d = x.shape

        # Run through transformer blocks manually to support middle insertion
        x = self.prepare_tokens_with_masks(x, masks)

        for i, blk in enumerate(self.blocks):
            if (
                self.task_tokens is not None
                and self.task_token_insertion == "middle"
                and i == self.task_token_block
            ):
                num_prefix = 1 + (self.num_register_tokens if self.register_tokens is not None else 0)
                prefix = x[:, :num_prefix]
                patches = x[:, num_prefix:]
                x = torch.cat((prefix, self.task_tokens.tokens.expand(x.shape[0], -1, -1), patches), dim=1)

            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x = blk(x, return_attention=last_self_attention)

        cls_tokens = self.norm(x[:, : self.num_register_tokens + 1])
        patch_tokens = self.norm(x[:, self.num_register_tokens + 1:])

        if self.task in ["regression", "classification", "none"]:
            return self.head(cls_tokens[:, 0])
        elif self.task == "segmentation":
            psh, psw, psd = self.patch_size
            hp, wp, dp = h // psh, w // psw, d // psd

            spatial = rearrange(
                patch_tokens, "b (hp wp dp) c -> b c hp wp dp", hp=hp, wp=wp, dp=dp
            )

            upscaled = self.upscale(spatial)
            rearranged = rearrange(upscaled, "b c ... -> b ... c")
            logits = rearrange(self.head(rearranged), "b ... c -> b c ...")
            return F.interpolate(
                logits, size=(h, w, d), mode="trilinear", align_corners=False
            )
        else:
            return cls_tokens[:, 0]

    def additional_trainable(self):
        return ["patch_embed", "pos_embed"]

    def do_not_load(self):
        return ["pos_embed", "patch_embed"]


def vitv2_a_3d_tiny(
    volume_size=(224, 224, 32), volume_patch_size=(14, 14, 2), lora=False, **kwargs
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption3D(
        volume_size=volume_size,
        volume_patch_size=volume_patch_size,
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


def vitv2_a_3d_small(
    volume_size=(224, 224, 32), volume_patch_size=(14, 14, 2), lora=False, **kwargs
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption3D(
        volume_size=volume_size,
        volume_patch_size=volume_patch_size,
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


def vitv2_a_3d_base(
    volume_size=(224, 224, 32), volume_patch_size=(14, 14, 2), lora=False, **kwargs
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption3D(
        volume_size=volume_size,
        volume_patch_size=volume_patch_size,
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


def vitv2_a_3d_large(
    volume_size=(224, 224, 32), volume_patch_size=(14, 14, 2), lora=False, **kwargs
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 1e-5
    model = ViTv2Adaption3D(
        volume_size=volume_size,
        volume_patch_size=volume_patch_size,
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
