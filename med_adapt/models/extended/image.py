from typing import Literal
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from med_adapt.models.base import ViTv2
from med_adapt.registry import register_model
from med_adapt.utils.config import get_logger
from med_adapt.adapter import InputChannelAdapter
from med_adapt.layers import (
    Block,
    ScaleBlock,
    MemEffAttention,
    LoRAMemEffAttention,
    # CrossAttentionBlock,
)

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
        query_from: int = -6,
        **kwargs,
    ):
        super(ViTv2Adaption, self).__init__(*args, **kwargs)
        if volume_size is not None or volume_patch_size is not None:
            logger.warning(
                f"volume_size={volume_size} and volume_patch_size={volume_patch_size} are not used in this 2D adaptation. Ignored.",
            )

        self.task = task
        self.input_adapter = InputChannelAdapter(in_channels=med_in_channels)

        # Query tokens: one set per volume, not per slice
        self.num_q_tokens = 1 if task == "classification" else classes
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_q_tokens, self.embed_dim), requires_grad=True
        )
        nn.init.normal_(self.query_tokens, std=1e-6)

        # Cross-attention blocks: one per transformer block from query_from onward
        self.num_blocks = len(self.blocks)
        self.query_from = query_from
        self.query_from = (
            self.num_blocks + self.query_from
            if self.query_from < 0
            else self.query_from
        )
        # self.num_cross_attn_blocks = self.num_blocks - self.query_from

        # self.cross_attn_blocks = nn.ModuleList(
        #     [
        #         CrossAttentionBlock(
        #             dim=self.embed_dim,
        #             num_heads=self.num_heads,
        #         )
        #         for _ in range(self.num_cross_attn_blocks)
        #     ]
        # )

        # Task-specific heads
        if task == "segmentation":
            self.upscale = nn.Sequential(
                ScaleBlock(self.embed_dim, conv_type="3d"),
                ScaleBlock(self.embed_dim // 2, conv_type="3d"),
                # ScaleBlock(self.embed_dim // 4, conv_type="3d"),
                # ScaleBlock(self.embed_dim // 8, conv_type="3d"),
            )
            self.query_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim // 2, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim // 2, self.embed_dim // 4, bias=False),
            )
        elif task == "classification":
            self.query_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim // 4, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim // 4, classes, bias=False),
            )
        else:  # regression
            self.query_mlp = nn.ModuleDict(
                {
                    f"class_{i}": nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                        nn.GELU(),
                        nn.Linear(self.embed_dim, self.embed_dim // 4, bias=True),
                        nn.GELU(),
                        nn.Linear(self.embed_dim // 4, 1, bias=True),
                    )
                    for i in range(self.num_q_tokens)
                }
            )

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

        x = self.prepare_tokens(reshaped_x)

        preds = []
        attn_bias = None

        for i, blk in enumerate(self.blocks):
            if i == self.query_from:
                # Insert query tokens repeated for each folded slice
                x = torch.cat((self.query_tokens.repeat(b * d, 1, 1), x), dim=1)

            # logger.debug(f"Depth: {i=}, {x.shape}")
            x = blk(x, attn_bias=attn_bias)

            if i >= self.query_from:
                num_q = self.num_q_tokens
                num_reg = self.num_register_tokens

                # Split folded sequence
                query_tokens = x[:, :num_q, :]
                register_and_cls = x[:, num_q : num_q + num_reg + 1, :]
                patch_tokens = x[:, num_q + num_reg + 1 :, :]

                # # Flatten patches across depth: (B*D, N, E) -> (B, D, N, E) -> (B, D*N, E)
                # patches = patch_tokens.reshape(b, d, patch_tokens.shape[1], -1).reshape(
                #     b, d * patch_tokens.shape[1], -1
                # )
                #
                # # Cross-attention: queries attend over all slices
                # cross_idx = i - self.query_from
                # queries = self.cross_attn_blocks[cross_idx](queries, patches)

                # Aggregate queries across depth: (B*D, Q, E) -> (B, Q, E)
                queries = query_tokens.reshape(b, d, num_q, -1).mean(dim=1)

                # Replace queries in sequence (repeat for each slice) for subsequent blocks
                x = torch.cat(
                    (queries.repeat(d, 1, 1), register_and_cls, patch_tokens), dim=1
                )

                # Generate prediction
                if self.task == "segmentation":
                    patch_latents = x[:, num_q + num_reg + 1 :, :]
                    h_p = h // self.patch_size
                    w_p = w // self.patch_size
                    # Reshape to 3D volume: (B, D, H_p, W_p, E) -> (B, E, D, H_p, W_p)
                    patch_latents = patch_latents.reshape(
                        b, d, h_p, w_p, self.embed_dim
                    )
                    patch_latents = patch_latents.permute(0, 4, 1, 2, 3)

                    patch_decode = self.upscale(patch_latents)

                    query_logits = self.query_mlp(queries)
                    seg_pred = torch.einsum(
                        "b c d h w, b q c -> b q d h w",
                        patch_decode,
                        query_logits,
                    )

                    upscaled = F.interpolate(
                        seg_pred,
                        size=(h, w, d),
                        mode="trilinear",
                        align_corners=False,
                    )
                    preds.append(upscaled)

                elif self.task == "classification":
                    cls_pred = self.query_mlp(queries.squeeze(1))
                    preds.append(cls_pred)
                else:  # regression
                    reg_pred = [
                        self.query_mlp[f"class_{i}"](queries[:, i, :])
                        for i in range(self.num_q_tokens)
                    ]
                    preds.append(reg_pred)

        return preds

    def additional_trainable(self):
        return [
            "query_mlp",
            "query_tokens",
            "cross_attn_blocks",
            "upscale",
            "input_adapter",
        ]

    def do_not_load(self):
        return None


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


if __name__ == "__main__":
    _m = vitv2_a_2d_small(
        med_in_channels=1, task="segmentation", classes=2, lora=True
    ).to("cuda")
    _m(torch.randn(1, 1, 196, 196, 16, device="cuda", dtype=torch.float32))
