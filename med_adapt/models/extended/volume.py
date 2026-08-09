from typing import Literal
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange, einsum

from med_adapt.models.base import ViTv2
from med_adapt.adapter import PatchEmbed3D
from med_adapt.registry import register_model
from med_adapt.utils.config import get_logger
from med_adapt.layers import Block, ScaleBlock, MemEffAttention, LoRAMemEffAttention

logger = get_logger(__name__)


class ViTv2Adaption3D(ViTv2):

    def __init__(
        self,
        volume_size: tuple[int, int, int],
        volume_patch_size: tuple[int, int, int],
        med_in_channels: int,
        task: Literal["regression", "classification", "segmentation", "none"],
        classes: int,
        *args,
        query_from: int = -6,
        **kwargs,
    ):
        super(ViTv2Adaption3D, self).__init__(*args, **kwargs)

        self.task = task
        self.query_from = (
            len(self.blocks) + query_from if query_from < 0 else query_from
        )

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

        self.num_q_tokens = 1 if task == "classification" else classes
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_q_tokens, self.embed_dim), requires_grad=True
        )
        nn.init.normal_(self.query_tokens, std=1e-6)

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
                nn.Linear(self.embed_dim, self.embed_dim // 2, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim // 2, self.embed_dim // 4, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim // 4, classes, bias=False),
            )
        else:
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

    def prepare_tokens(self, x):
        B, nc, h, w, d = x.shape
        x = self.patch_embed(x)

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

        return x

    def _mask_logits(self, patch_tokens, h, w, d) -> Tensor:
        psh, psw, psd = self.patch_size
        hp, wp, dp = h // psh, w // psw, d // psd

        spatial = rearrange(
            patch_tokens, "b (hp wp dp) c -> b c hp wp dp", hp=hp, wp=wp, dp=dp
        )

        upscaled = self.upscale(spatial)
        return upscaled

    def forward(self, x, **kwargs):
        b, c, h, w, d = x.shape

        # Run through transformer blocks manually to support middle insertion
        x = self.prepare_tokens(x)

        preds = []

        attn_bias = None
        for i, blk in enumerate(self.blocks):
            if i == self.query_from:
                x = torch.cat((self.query_tokens.repeat(b, 1, 1), x), dim=1)

            # logger.debug(f"Depth: {i=}, {x.shape}")
            x = blk(x, attn_bias=attn_bias)
            if i >= self.query_from:
                if self.task == "segmentation":
                    mask_logits = self._mask_logits(
                        x[:, self.num_q_tokens + self.num_register_tokens + 1 :, :],
                        h,
                        w,
                        d,
                    )  # [B, d, ...]
                    query_logits = self.query_mlp(
                        x[:, : self.num_q_tokens, :]
                    )  # [B, q, d]
                    segmentation_pred = einsum(
                        mask_logits, query_logits, "b d ..., b q d -> b q ..."
                    )
                    # logger.debug(segmentation_pred.shape)
                    preds.append(
                        F.interpolate(
                            segmentation_pred,
                            size=(h, w, d),
                            mode="trilinear",
                            align_corners=False,
                        )
                    )
                else:
                    query_logits = x[:, : self.num_q_tokens, :]  # [B, q, d]
                    if self.task == "classification":
                        cls_pred = self.query_mlp(query_logits.squeeze(1))
                        # logger.debug(cls_pred.shape)
                        preds.append(cls_pred)
                    else:
                        reg_pred = [
                            self.query_mlp[f"class_{i}"](query_logits[:, i, :])
                            for i in range(self.num_q_tokens)
                        ]
                        # logger.debug([reg.shape for reg in reg_pred])
                        preds.append(reg_pred)

        return preds

    def additional_trainable(self):
        return [
            "patch_embed",
            "pos_embed",
            "query_mlp",
            "query_token",
            "upscale",
            "head",
        ]

    def do_not_load(self):
        return ["pos_embed", "patch_embed"]


@register_model("vitv2_a_3d_tiny")
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


@register_model("vitv2_a_3d_small")
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


@register_model("vitv2_a_3d_base")
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


@register_model("vitv2_a_3d_large")
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


if __name__ == "__main__":
    _m = vitv2_a_3d_tiny(
        med_in_channels=1, task="segmentation", classes=2, lora=True
    ).to("cuda")
    _m(torch.randn(1, 1, 196, 196, 28, device="cuda", dtype=torch.float32))
