from __future__ import annotations

from functools import partial
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from med_adapt.layers import PatchEmbed3D, ScaleDecode
from med_adapt.registry import register_model
from med_adapt.layers import Block, MemEffAttention
from med_adapt.models.base.vit2d import ViTv2, init_weights_vit

TUP3 = Tuple[int, int, int]
INT_TUP3 = Union[int, TUP3]


def _maybe_to_3_tuple(size, name="tuple"):
    if isinstance(size, tuple):
        if len(size) != 3:
            raise ValueError(
                f"Incorrect length for tuple: {name}={len(tuple)=}, {size}"
            )
        else:
            return size
    else:
        return size, size, size


class ViT3D(ViTv2):
    """3-D ViT that processes volumes ``(B, C, H, W, D)`` directly."""

    def __init__(
        self,
        volume_size,
        volume_patch_size,
        med_in_channels,
        use_patch_decode=True,
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

        if self.use_patch_decode:
            self.patch_decode = ScaleDecode(
                self.patch_size, self.embed_dim, self.in_channels
            )

        if self.use_mask:
            self.mask_token = torch.nn.Parameter(
                torch.zeros(1, self.embed_dim), requires_grad=True
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

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens(self, x, mask=None):
        B, nc, h, w, d = x.shape
        x = self.patch_embed(x)

        if mask is not None and self.mask_token is not None:
            x = torch.where(
                mask.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
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

        return x

    def forward(self, x, distill_from=-1, mask=None, return_dict=False, **kwargs):
        *_, h, w, d = x.shape
        lp = tuple(l // p for l, p in zip([h, w, d], self.patch_size))

        x = self.prepare_tokens(x, mask=mask)

        resolved_idx = (
            self.n_blocks + distill_from if distill_from < 0 else distill_from
        )

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i >= resolved_idx:
                cls_token = self.norm(x[:, : self.num_register_tokens + 1])[:, 0]
                patch_tokens = (
                    self.norm(x[:, self.num_register_tokens + 1 :])
                    .unflatten(1, lp)
                    .permute(0, -1, 1, 2, 3)
                )

                outs.append((cls_token, patch_tokens))

        if self.use_patch_decode:
            recon = self.patch_decode(outs[-1][-1])
        else:
            recon = None

        if return_dict:
            return {"latent": outs[-1][0], "patch_latent": outs[-1][1], "recon": recon}

        return outs, recon


def init_weights_vit_3d(module: nn.Module, name: str = "") -> None:
    """Initialisation that handles both 2-D and 3-D modules."""
    init_weights_vit(module, name)
    if isinstance(module, PatchEmbed3D):
        nn.init.trunc_normal_(module.proj.weight, std=0.02)
        if module.proj.bias is not None:
            nn.init.zeros_(module.proj.bias)
        module.norm.reset_parameters()


@register_model("vitv2_3d_tiny")
def vitv2_3d_tiny(
    volume_size: INT_TUP3 = 196,
    volume_patch_size: INT_TUP3 = 14,
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
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


@register_model("vitv2_3d_small")
def vitv2_3d_small(
    volume_size: INT_TUP3 = 196,
    volume_patch_size: INT_TUP3 = 14,
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
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


@register_model("vitv2_3d_base")
def vitv2_3d_base(
    volume_size: INT_TUP3 = 196,
    volume_patch_size: INT_TUP3 = 14,
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
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


@register_model("vitv2_3d_large")
def vitv2_3d_large(
    volume_size: INT_TUP3 = 196,
    volume_patch_size: INT_TUP3 = 14,
    med_in_channels=1,
    num_register_tokens=0,
    **kwargs,
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 1e-5

    return ViT3D(
        volume_size=volume_size,
        volume_patch_size=volume_patch_size,
        med_in_channels=med_in_channels,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


if __name__ == "__main__":
    # import thop

    model = vitv2_3d_small(
        volume_size=(196, 196, 28), volume_patch_size=(14, 14, 2), med_in_channels=3
    ).cuda()

    vol = torch.randn(1, 3, 196, 196, 28, device="cuda")
    mask = torch.rand(1, 14, 14, 14, device="cuda").flatten(1) > 0.5
    # macs, params = thop.profile(model, (vol,))
    # print("Model FLOPs & Params:")
    # print("\t".join(thop.clever_format([macs, params], "%.3f")))

    with torch.no_grad():
        out, recon = model(vol, distill_from=-2, mask=mask)

    for layer_out in out:
        print([cls_patch.shape for cls_patch in layer_out])

    print(recon.shape)
