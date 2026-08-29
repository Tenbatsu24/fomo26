from typing import Literal, Optional
from functools import partial

import torch
import torch.nn as nn

from einops import rearrange

from med_adapt.models.base import ViTv2
from med_adapt.registry import register_model
from med_adapt.utils.config import get_logger
from med_adapt.adapter import InputChannelAdapter, AttentionPooling
from med_adapt.layers import (
    Block,
    ScaleDecode,
    Attention,
    MemEffAttention,
    LoRAAttention,
    LoRAMemEffAttention,
)

logger = get_logger(__name__)


def _maybe_to_2_tuple(val):
    if isinstance(val, (tuple, list)):
        assert len(val) == 2, f"Found {len(val)=}, expected 2"
        return tuple(val)
    else:
        return tuple([val for _ in range(2)])


class ViTv2Adaption(ViTv2):

    def __init__(
        self,
        n_modalities,
        task: Literal["regression", "classification", "segmentation", "none"],
        classes: int,
        depth_last=True,
        query_from: Optional[int] = None,
        num_q_tokens: Optional[int] = None,
        **kwargs,
    ):
        super(ViTv2Adaption, self).__init__(**kwargs)

        self.task = task
        self.classes = classes
        self.n_modalities = n_modalities
        self.depth_last = depth_last

        if query_from is None:
            query_from = -3
        self.query_from = (
            len(self.blocks) + query_from if query_from < 0 else query_from
        )

        if n_modalities != 3:
            self.input_adapter = InputChannelAdapter(in_channels=n_modalities)
        else:
            self.input_adapter = nn.Identity()

        if task in ["classification", "regression"]:
            self.num_q_tokens = num_q_tokens if num_q_tokens is not None else 4
            self.query_tokens = nn.Parameter(
                torch.zeros(1, self.num_q_tokens, self.embed_dim)
            )
            nn.init.normal_(self.query_tokens, std=1e-6)
            self.query_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.attn_head = nn.Sequential(
                AttentionPooling(self.embed_dim, num_classes=1, num_heads=4),
                nn.Linear(self.embed_dim, classes),
            )
        else:  # task == "segmentation":
            self.num_q_tokens = 0
            self.query_tokens = None
            if self.task == "segmentation":
                if self.depth_last:
                    scale_decode_patch_size = (self.patch_size, self.patch_size, 1)
                else:
                    scale_decode_patch_size = (1, self.patch_size, self.patch_size)
                self.head = ScaleDecode(
                    scale_decode_patch_size, self.embed_dim, classes
                )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1),
        )

    def prepare_tokens(self, x):
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
        if self.depth_last:
            b, c, h, w, d = x.shape
        else:
            b, c, d, h, w = x.shape
        lp = tuple(l // p for l, p in zip([h, w], [self.patch_size, self.patch_size]))

        volume = self.input_adapter(x)

        ch_min = volume.amin(dim=(1, 2, 3, 4), keepdim=True)
        ch_max = volume.amax(dim=(1, 2, 3, 4), keepdim=True)

        denom = ch_max - ch_min
        denom = torch.where(denom < 0.1, 1.0, denom)

        vol_norm = (volume - ch_min) / denom
        vol_norm = (vol_norm - self.imagenet_mean) / self.imagenet_std

        if self.depth_last:
            folded_vol = rearrange(vol_norm, "b ... d -> (b d) ...")
        else:
            folded_vol = rearrange(vol_norm, "b c d ... -> (b d) c ...")

        x = self.prepare_tokens(folded_vol)  # [b * d, n, c]

        for i, blk in enumerate(self.blocks):
            if (self.query_tokens is not None) and (i == self.query_from):
                x = torch.cat((self.query_tokens.repeat(b * d, 1, 1), x), dim=1)
            x = blk(x)

        unfolded_x = rearrange(x, "(b d) n c -> b n d c", b=b, d=d)

        if self.task == "segmentation":
            patch_tokens = self.norm(
                unfolded_x[:, self.num_q_tokens + self.num_register_tokens + 1 :, ...]
            )
            spatial = patch_tokens.unflatten(1, lp).permute(0, -1, 1, 2, 3)
            if not self.depth_last:
                spatial = spatial.permute(0, 1, -1, 2, 3)
            pred = self.head(spatial)
        elif self.task in ["classification", "regression"]:
            query_latent = self.query_norm(
                unfolded_x[:, : self.num_q_tokens, ...].flatten(1, 2)
            )  # [b q*d c]
            pred = self.attn_head(query_latent).squeeze(1)
            if self.task == "regression":
                pred = 100 * torch.sigmoid(pred)
        else:
            all_cls = self.norm(unfolded_x[:, : self.num_register_tokens + 1, ...])[
                :, 0, ...
            ]
            pairwise_dist = torch.cdist(all_cls, all_cls, p=2)
            medoid_idx = pairwise_dist.sum(dim=-1).argmin(dim=-1)
            pred = all_cls[
                torch.arange(b, device=all_cls.device),
                medoid_idx,
            ]
        return pred

    def additional_trainable(self):
        return ["query_tokens", "query_norm", "attn_head", "input_adapter", "head"]

    def do_not_load(self):
        return None


@register_model("vitv2_a_2d_tiny")
def vitv2_a_2d_tiny(lora=False, mea=True, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption(
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


@register_model("vitv2_a_2d_small")
def vitv2_a_2d_small(lora=False, mea=True, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption(
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


@register_model("vitv2_a_2d_base")
def vitv2_a_2d_base(lora=False, mea=True, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2Adaption(
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


@register_model("vitv2_a_2d_large")
def vitv2_a_2d_large(lora=False, mea=True, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 1e-5
    model = ViTv2Adaption(
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
    _m = vitv2_a_2d_small(n_modalities=1, task="segmentation", classes=2).to("cuda")
    print(_m(torch.randn(1, 1, 196, 196, 16, device="cuda", dtype=torch.float32)).shape)
