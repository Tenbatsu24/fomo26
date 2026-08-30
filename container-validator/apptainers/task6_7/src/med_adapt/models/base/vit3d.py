from __future__ import annotations

from functools import partial
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers import PatchEmbed3D, Block, MemEffAttention
from .vit2d import ViTv2, init_weights_vit

TUP3 = Tuple[int, int, int]
INT_TUP3 = Union[int, TUP3]


def _maybe_to_3_tuple(size, name="tuple"):
    if isinstance(size, tuple):
        if len(size) != 3:
            raise ValueError(f"Incorrect length for tuple: {name}={len(size)}, {size}")
        return size

    return size, size, size


class ViT3D(ViTv2):
    """3-D ViT that processes volumes ``(B, C, H, W, D)`` directly."""

    def __init__(
        self,
        volume_size=(296, 296, 296),
        volume_patch_size=(8, 8, 8),
        med_in_channels=1,
        use_patch_decode=False,
        use_mask=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **{
                **kwargs,
                "img_size": _maybe_to_3_tuple(volume_size),
                "patch_size": _maybe_to_3_tuple(volume_patch_size),
                "in_chans": med_in_channels,
                "embed_layer": PatchEmbed3D,
            },
        )

        self.use_mask = use_mask
        self.use_patch_decode = use_patch_decode

        if self.use_mask:
            self.mask_token = torch.nn.Parameter(
                torch.zeros(1, self.embed_dim),
                requires_grad=True,
            )

        self.init_weights()

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

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).reshape(1, -1, dim)

        return torch.cat(
            (
                class_pos_embed.unsqueeze(0),
                patch_pos_embed,
            ),
            dim=1,
        ).to(previous_dtype)

    def prepare_tokens(self, x, mask=None):
        B, nc, h, w, d = x.shape

        x = self.patch_embed(x)

        # This is the model's original masking mechanism.
        # image_mask from forward() is deliberately NOT passed here.
        if mask is not None and self.mask_token is not None:
            x = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.to(x.dtype).unsqueeze(0),
                x,
            )

        x = torch.cat(
            (
                self.cls_token.expand(x.shape[0], -1, -1),
                x,
            ),
            dim=1,
        )

        x = x + self.interpolate_pos_encoding(
            x,
            h,
            w,
            d,
        )

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(
                        x.shape[0],
                        -1,
                        -1,
                    ),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    @staticmethod
    def _prepare_patch_mask(
        image_mask: torch.Tensor,
        patch_resolution: tuple[int, int, int],
    ) -> torch.Tensor:

        if image_mask.ndim == 4:
            image_mask = image_mask.unsqueeze(1)

        if image_mask.ndim != 5:
            raise ValueError(
                "image_mask must have shape [B, H, W, D] or "
                f"[B, 1, H, W, D], got {tuple(image_mask.shape)}"
            )

        if image_mask.shape[1] != 1:
            raise ValueError(
                "image_mask must have exactly one channel, got "
                f"shape {tuple(image_mask.shape)}"
            )

        mask_occupancy = F.interpolate(
            image_mask.float(),
            size=patch_resolution,
            mode="area",
        )

        patch_mask = mask_occupancy > 0.0

        return patch_mask.flatten(1)

    @staticmethod
    def _masked_cls_pool(
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if cls_token.ndim != 2:
            raise ValueError(f"Expected cls_token [B, C], got {tuple(cls_token.shape)}")

        if patch_tokens.ndim != 3:
            raise ValueError(
                "Expected patch_tokens [B, N, C], got " f"{tuple(patch_tokens.shape)}"
            )

        if patch_mask.shape != patch_tokens.shape[:2]:
            raise ValueError(
                "patch_mask shape must match the first two dimensions of "
                f"patch_tokens. Got mask={tuple(patch_mask.shape)}, "
                f"tokens={tuple(patch_tokens.shape)}"
            )

        cls_normalized = F.normalize(
            cls_token.float(),
            dim=-1,
        )

        patch_normalized = F.normalize(
            patch_tokens.float(),
            dim=-1,
        )

        similarity = torch.einsum(
            "bc,bnc->bn",
            cls_normalized,
            patch_normalized,
        )

        valid_samples = patch_mask.any(dim=1)

        masked_similarity = similarity.masked_fill(
            ~patch_mask,
            float("-inf"),
        )

        weights = torch.zeros_like(similarity)

        if valid_samples.any():
            weights[valid_samples] = F.softmax(
                masked_similarity[valid_samples],
                dim=-1,
            )

        weights_for_pooling = weights.to(patch_tokens.dtype)

        embedding = torch.einsum(
            "bn,bnc->bc",
            weights_for_pooling,
            patch_tokens,
        )

        if (~valid_samples).any():
            embedding = torch.where(
                valid_samples.unsqueeze(-1),
                embedding,
                cls_token,
            )

        return embedding, weights

    def forward(
        self,
        x,
        image_mask=None,
        **kwargs,
    ):
        *_, h, w, d = x.shape

        patch_resolution = tuple(
            length // patch
            for length, patch in zip(
                (h, w, d),
                self.patch_size,
            )
        )

        x = self.prepare_tokens(
            x,
            mask=None,
        )

        for blk in self.blocks:
            x = blk(x)

        cls_token = self.norm(x[:, : self.num_register_tokens + 1])[:, 0]

        patch_tokens_flat = self.norm(x[:, self.num_register_tokens + 1 :])

        if image_mask is not None:
            patch_mask = self._prepare_patch_mask(
                image_mask=image_mask,
                patch_resolution=patch_resolution,
            )

            foreground_embedding, cls_patch_weights = self._masked_cls_pool(
                cls_token=cls_token,
                patch_tokens=patch_tokens_flat,
                patch_mask=patch_mask,
            )
        else:
            # Keep output dimensionality fixed when no mask is available.
            foreground_embedding = cls_token

        embedding = torch.cat(
            (cls_token, foreground_embedding),
            dim=-1,
        )

        return embedding


def init_weights_vit_3d(
    module: nn.Module,
    name: str = "",
) -> None:
    """Initialisation that handles both 2-D and 3-D modules."""
    init_weights_vit(module, name)

    if isinstance(module, PatchEmbed3D):
        nn.init.trunc_normal_(
            module.proj.weight,
            std=0.02,
        )

        if module.proj.bias is not None:
            nn.init.zeros_(module.proj.bias)

        module.norm.reset_parameters()


def vitv2_3d_small(
    volume_size: INT_TUP3 = 296,
    volume_patch_size: INT_TUP3 = 8,
    med_in_channels=1,
    num_register_tokens=0,
    **kwargs,
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1

    return ViT3D(
        volume_size=volume_size,
        volume_patch_size=volume_patch_size,
        med_in_channels=med_in_channels,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(
            Block,
            attn_class=MemEffAttention,
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
