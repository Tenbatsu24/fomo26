from functools import partial
from typing import Literal, Mapping, Any


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange, einsum

from med_adapt.models.base import ViTv2
from med_adapt.registry import register_model
from med_adapt.utils.config import get_logger
from med_adapt.layers import (
    Block,
    ScaleBlock,
    Attention,
    PatchEmbed3D,
    MemEffAttention,
    LoRAAttention,
    LoRAMemEffAttention,
)

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

        self.img_size = volume_size
        self.in_channels = med_in_channels
        self.patch_size = volume_patch_size

        self.patch_embed = PatchEmbed3D(
            img_size=volume_size,
            patch_size=volume_patch_size,
            in_chans=med_in_channels,
            embed_dim=self.embed_dim,
        )

        # from dinov2 / neco models
        self._pos_embed_grid_size = 37
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self._pos_embed_grid_size**3 + self.num_tokens,
                self.embed_dim,
            )
        )

        self.query_from = (
            len(self.blocks) + query_from if query_from < 0 else query_from
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

        ph = pw = pd = self._pos_embed_grid_size

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
                        cls_pred = self.query_mlp(query_logits[:, 0])
                        # logger.debug(cls_pred.shape)
                        preds.append(cls_pred)
                    else:
                        reg_pred = torch.stack(
                            [
                                self.query_mlp[f"class_{i}"](query_logits[:, i, :])[
                                    :, 0
                                ]
                                for i in range(self.num_q_tokens)
                            ],
                            dim=-1,
                        )
                        # logger.debug(reg.shape)
                        preds.append(reg_pred)

        return preds

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        model_ps = self.patch_size

        new_state_dict = {**state_dict}

        ckpt_ps = state_dict["patch_embed.proj.weight"].shape[2:]
        if model_ps != ckpt_ps:
            if model_ps[0] == ckpt_ps[0] and model_ps[1] == ckpt_ps[1]:
                w_ckpt = state_dict["patch_embed.proj.weight"]
                kd_model = model_ps[2]
                kd_ckpt = ckpt_ps[2]
                logger.info(f"Resampling patch_embed depth: {kd_ckpt} -> {kd_model}")
                w_resampled = F.interpolate(
                    w_ckpt,
                    size=(ckpt_ps[0], ckpt_ps[1], kd_model),
                    mode="trilinear",
                    align_corners=False,
                )
                new_state_dict["patch_embed.proj.weight"] = w_resampled
                logger.info(f"  weight: {w_ckpt.shape} -> {w_resampled.shape}")
            else:
                logger.error(
                    f"Only depth patch size may differ: model={model_ps}, ckpt={ckpt_ps}"
                )
                new_state_dict["patch_embed.proj.weight"] = self.patch_embed.weight
                new_state_dict["patch_embed.proj.bias"] = self.patch_embed.bias

        ckpt_inch = new_state_dict["patch_embed.proj.weight"].shape[1]
        if self.in_channels != ckpt_inch:
            logger.info(f" in_ch: {ckpt_inch} -> {self.in_channels}")
            new_proj = (
                new_state_dict["patch_embed.proj.weight"]
                .mean(dim=1, keepdim=True)
                .repeat(1, self.in_channels, 1, 1, 1)
            )
            new_state_dict["patch_embed.proj.weight"] = new_proj

        del state_dict

        return super().load_state_dict(new_state_dict, strict=strict, assign=assign)

    def additional_trainable(self):
        return [
            # "patch_embed",
            # "pos_embed",
            "query_mlp",
            "query_tokens",
            "upscale",
            "head",
        ]


@register_model("vitv2_a_3d_tiny")
def vitv2_a_3d_tiny(
    volume_size=(224, 224, 32),
    volume_patch_size=(14, 14, 2),
    lora=False,
    mea=True,
    **kwargs,
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
            Block,
            attn_class=(
                (LoRAMemEffAttention if mea else LoRAAttention)
                if lora
                else (MemEffAttention if mea else Attention)
            ),
        ),
        num_register_tokens=4,
        **kwargs,
    )
    return model


@register_model("vitv2_a_3d_small")
def vitv2_a_3d_small(
    volume_size=(224, 224, 32),
    volume_patch_size=(14, 14, 2),
    lora=False,
    mea=True,
    **kwargs,
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
            Block,
            attn_class=(
                (LoRAMemEffAttention if mea else LoRAAttention)
                if lora
                else (MemEffAttention if mea else Attention)
            ),
        ),
        num_register_tokens=4,
        **kwargs,
    )
    return model


@register_model("vitv2_a_3d_base")
def vitv2_a_3d_base(
    volume_size=(224, 224, 32),
    volume_patch_size=(14, 14, 2),
    lora=False,
    mea=True,
    **kwargs,
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
            Block,
            attn_class=(
                (LoRAMemEffAttention if mea else LoRAAttention)
                if lora
                else (MemEffAttention if mea else Attention)
            ),
        ),
        num_register_tokens=4,
        **kwargs,
    )
    return model


@register_model("vitv2_a_3d_large")
def vitv2_a_3d_large(
    volume_size=(224, 224, 32),
    volume_patch_size=(14, 14, 2),
    lora=False,
    mea=True,
    **kwargs,
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
            Block,
            attn_class=(
                (LoRAMemEffAttention if mea else LoRAAttention)
                if lora
                else (MemEffAttention if mea else Attention)
            ),
        ),
        num_register_tokens=4,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    _m = vitv2_a_3d_small(
        volume_size=(196, 196, 28),
        volume_patch_size=(14, 14, 2),
        med_in_channels=1,
        task="classification",
        classes=2,
        lora=False,
    ).to("cuda")

    _m.load_state_dict(
        torch.load("../../../checkpoints/small/neco_3d/encoder_teacher.ckpt"),
        strict=False,
    )
    _m(torch.randn(1, 1, 196, 196, 28, device="cuda", dtype=torch.float32))
