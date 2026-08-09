from typing import Literal
from functools import partial

import torch

from einops import rearrange

from med_adapt.models.base import ViTv2
from med_adapt.registry import register_model
from med_adapt.utils.config import get_logger
from med_adapt.adapter import InputChannelAdapter
from med_adapt.layers import Block, MemEffAttention, LoRAMemEffAttention

logger = get_logger(__name__)


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
            logger.warning(
                f"volume_size={volume_size} and volume_patch_size={volume_patch_size} are not used in this 2D adaptation. Ignored.",
            )

        self.input_adapter = InputChannelAdapter(in_channels=med_in_channels)

        self.task = task

        ...

    def prepare_tokens(self, x):
        # x here is the 2D reshaped input: (B*d, C, H, W)
        b_orig, c, h, w = x.shape  # B_orig = B * d (depth-folded)

        x = self.patch_embed(x)

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

        return x

    def forward(self, x, **kwargs):
        b, c, h, w, d = x.shape

        adapted_x = self.input_adapter(x)
        reshaped_x = rearrange(adapted_x, "b c ... d -> (b d) c ...")

        # Run through transformer blocks manually to support middle insertion
        x = self.prepare_tokens_with_masks(reshaped_x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        cls_tokens = self.norm(x[:, : self.num_register_tokens + 1])
        patch_tokens = self.norm(x[:, self.num_register_tokens + 1 :])

        ...


@register_model("vitv2_a_2d_tiny")
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


@register_model("vitv2_a_2d_small")
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


@register_model("vitv2_a_2d_base")
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


@register_model("vitv2_a_2d_large")
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
