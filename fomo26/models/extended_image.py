import logging
from typing import Literal
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from fomo26.models.base import ViTv2
from fomo26.adapter import InputChannelAdapter, AttentionPooling, TaskTokens
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
        task_token: bool = False,
        task_token_insertion: Literal["beginning", "middle"] = "beginning",
        task_token_block: int = 6,
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

    def prepare_tokens_with_masks(self, x, masks=None):
        # x here is the 2D reshaped input: (B*d, C, H, W)
        B_orig, c, h, w = x.shape  # B_orig = B * d (depth-folded)

        x = self.patch_embed(x)
        if masks is not None:
            # masks are per-volume (B, H, W) — expand for depth-folded
            B_vol = B_orig // max(1, h // self.patch_size * w // self.patch_size)
            # Simpler: just use the mask as-is if shapes match
            if masks.shape[0] == B_vol:
                masks_exp = masks.unsqueeze(1).expand(-1, c, -1, -1).reshape(B_orig, -1, 1)
                x = torch.where(
                    masks_exp.bool(),
                    self.mask_token.to(x.dtype).unsqueeze(0), x
                )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        # Inject task tokens after CLS (+ register) prefix
        if self.task_tokens is not None and self.task_token_insertion == "beginning":
            num_prefix = 1 + (self.num_register_tokens if self.register_tokens is not None else 0)
            prefix = x[:, :num_prefix]
            rest = x[:, num_prefix:]
            x = torch.cat(
                (prefix, self.task_tokens.tokens.expand(x.shape[0], -1, -1), rest),
                dim=1,
            )

        return x

    def forward(self, x, masks=None, last_self_attention=False, **kwargs):
        b, c, h, w, d = x.shape

        adapted_x = self.input_adapter(x)
        reshaped_x = rearrange(adapted_x, "b c ... d -> (b d) c ...")

        # Run through transformer blocks manually to support middle insertion
        x = self.prepare_tokens_with_masks(reshaped_x, masks)

        for i, blk in enumerate(self.blocks):
            if (
                self.task_tokens is not None
                and self.task_token_insertion == "middle"
                and i == self.task_token_block
            ):
                num_prefix = 1 + (self.num_register_tokens if self.register_tokens is not None else 0)
                prefix = x[:, :num_prefix]
                patches = x[:, num_prefix:]
                x = torch.cat(
                    (prefix, self.task_tokens.tokens.expand(x.shape[0], -1, -1), patches),
                    dim=1,
                )

            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x = blk(x, return_attention=last_self_attention)

        cls_tokens = self.norm(x[:, : self.num_register_tokens + 1])
        patch_tokens = self.norm(x[:, self.num_register_tokens + 1:])

        if self.task in ["regression", "classification", "none"]:
            # patch_tokens: (B*d, N_patches, D) -> (B, d*N_patches, D)
            reshaped_out = rearrange(patch_tokens, "(b d) n c -> b (d n) c", b=b)
            attended = self.attn_pool(reshaped_out)
            return self.head(attended)
        elif self.task == "segmentation":
            hp, wp = h // self.patch_size, w // self.patch_size
            spatial = rearrange(patch_tokens, "b (hp wp) c -> b c hp wp", hp=hp, wp=wp)
            upscaled = self.upscale(spatial)
            rearranged = rearrange(upscaled, "b c ... -> b ... c")
            logits = rearrange(self.head(rearranged), "(b d) ... c -> b c ... d", b=b)
            return F.interpolate(
                logits, size=(h, w, d), mode="trilinear", align_corners=False
            )
        else:
            pooled = rearrange(cls_tokens, "(b d) c -> b d c", b=b).mean(dim=-2)
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
