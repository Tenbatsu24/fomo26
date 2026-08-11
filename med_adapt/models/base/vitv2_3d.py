"""3-D extension of the 2-D ViT base model.

Inherits from :class:`ViTv2` and replaces the 2-D patch embedding with a
3-D counterpart while keeping every transformer block unchanged.  The
positional embedding is expanded from a 2-D grid into a cubic 3-D grid
by taking the element-wise maximum across three axis-aligned orientations.
"""

from __future__ import annotations

import math
from pathlib import Path

from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from med_adapt.registry import register_model
from med_adapt.models.base.vitv2 import ViTv2, init_weights_vit


class ViTv2_3D(ViTv2):
    """3-D ViT that processes volumes ``(B, C, H, W, D)`` directly."""

    def __init__(self, *args, **kwargs):
        ps = kwargs.get("patch_size", 14)
        if isinstance(ps, tuple):
            kwargs["patch_size"] = ps[0]

        super().__init__(*args, **kwargs)
        h = self.img_size

        if isinstance(ps, int):
            ps = (ps, ps, ps)

        self.patch_size = ps

    def interpolate_pos_encoding(
        self, x: torch.Tensor, h: int, w: int, d: int
    ) -> torch.Tensor:
        """Bicubic-interpolate the 3-D positional embedding to match *x*.

        Parameters
        ----------
        x : torch.Tensor
            Token tensor ``[B, N, E]`` whose spatial size we match against.
        h, w, d : int
            Desired output grid dimensions (in patches).
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and h == w and d == self.patch_embed.patches_resolution[0]:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        ps = self.patch_size
        if isinstance(ps, int):
            ps = (ps, ps, ps)
        h0 = h // ps[0] + self.interpolate_offset
        w0 = w // ps[1] + self.interpolate_offset
        d0 = d // ps[2] + self.interpolate_offset

        sh = float(h0) / self._pos_embed_grid_size
        sw = float(w0) / self._pos_embed_grid_size
        sd = float(d0) / self._pos_embed_grid_size

        patch_pos_embed = patch_pos_embed.reshape(
            1,
            self._pos_embed_grid_size,
            self._pos_embed_grid_size,
            self._pos_embed_grid_size,
            dim,
        ).permute(0, 4, 1, 2, 3)

        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            scale_factor=(sh, sw, sd),
            mode="trilinear",
            align_corners=False,
        )

        assert int(h0) == patch_pos_embed.shape[2]
        assert int(w0) == patch_pos_embed.shape[3]
        assert int(d0) == patch_pos_embed.shape[4]

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        cls_tokens = self.norm(x[:, : self.num_register_tokens + 1])
        patch_tokens = self.norm(x[:, self.num_register_tokens + 1 :])
        return {
            "latent": cls_tokens[:, 0],
            "patch_latent": patch_tokens,
            "raw_latent": x[:, 0],
        }


def init_weights_vit_3d(module: nn.Module, name: str = "") -> None:
    """Initialisation that handles both 2-D and 3-D modules."""
    init_weights_vit(module, name)
    if isinstance(module, PatchEmbed3D):
        nn.init.trunc_normal_(module.proj.weight, std=0.02)
        if module.proj.bias is not None:
            nn.init.zeros_(module.proj.bias)
        module.norm.reset_parameters()


def build_3d_from_2d_checkpoint(ckpt_path: str | Path) -> dict:
    """Build a 3-D checkpoint from a 2-D ViT checkpoint.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to the 2-D checkpoint.

    Returns
    -------
    dict
        The new 3-D state dict.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    w2d = ckpt["patch_embed.proj.weight"]  # [out, in, kH, kW]
    bias2d = ckpt["patch_embed.proj.bias"]

    w3d = spectral_lifting(w2d)
    # w3d = w2d.unsqueeze(4).repeat(1, 1, 1, 1, w2d.shape[2]) / w2d.shape[2]
    # w3d[..., :] = 0.

    # Position embedding: expand 2-D grid to cubic 3-D via max-combine
    pos_2d = ckpt["pos_embed"]
    cls = pos_2d[:, :1, :]
    patches = pos_2d[:, 1:, :]
    num_patches_2d = patches.shape[1]
    grid_size = int(math.sqrt(num_patches_2d))
    assert (
        grid_size * grid_size == num_patches_2d
    ), f"pos_embed patch count {num_patches_2d} is not a perfect square"

    p = patches.view(1, grid_size, grid_size, -1)
    a = p.unsqueeze(3)  # [1, H, W, 1, C]
    s = p.unsqueeze(2)  # [1, H, 1, W, C]
    o = p.unsqueeze(1)  # [1, 1, H, W, C]
    p3 = torch.maximum(torch.maximum(a, s), o)
    p3 = p3.view(1, grid_size * grid_size * grid_size, -1)
    pos_3d = torch.cat([cls, p3], dim=1)

    ckpt_3d = {
        "cls_token": ckpt["cls_token"],
        "pos_embed": pos_3d,
        "register_tokens": ckpt.get("register_tokens", ckpt["cls_token"]),
        "patch_embed.proj.weight": w3d,
        "patch_embed.proj.bias": bias2d,
        "norm.weight": ckpt["norm.weight"],
        "norm.bias": ckpt["norm.bias"],
    }

    for k, v in ckpt.items():
        if k.startswith("blocks."):
            ckpt_3d[k] = v

    out_dir = ckpt_path.parent.parent / f"{ckpt_path.parent.name}_3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ckpt_path.name}"
    torch.save(ckpt_3d, out_path)
    print(f"Saved 3-D checkpoint → {out_path}")
    print(f"  patch_embed.proj.weight: {w2d.shape} → {w3d.shape}")
    print(f"  pos_embed: {pos_2d.shape} → {pos_3d.shape}")
    return ckpt_3d


def spectral_lifting(w2d) -> Any:
    eps = 1e-12

    # 2D Fourier transform
    H2 = torch.fft.fftn(w2d, dim=(-2, -1))

    # Embed into 3D frequency space via three axis-aligned orientations
    Hxy = H2.unsqueeze(2)
    Hxz = H2.unsqueeze(3)
    Hyz = H2.unsqueeze(4)

    mag = ((Hxy.abs() + eps) * (Hxz.abs() + eps) * (Hyz.abs() + eps)).pow(1 / 3)

    phase = torch.angle(Hxy + Hxz + Hyz)

    H3 = mag * torch.exp(1j * phase)
    # H3 = (Hxy + Hxz + Hyz) / 3.0

    w3d = torch.fft.ifftn(H3, dim=(-3, -2, -1)).real
    return w3d


@register_model("vitv2_3d_small")
def vitv2_3d_small(patch_size=14, num_register_tokens=4, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    from functools import partial
    from med_adapt.layers import Block, MemEffAttention

    model = ViTv2_3D(
        img_size=518,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def load_3d_checkpoint_with_anisotropic_patch(
    ckpt_path: str | Path,
    model: nn.Module,
) -> tuple[list[str], list[str]]:
    """Load a 3-D checkpoint into *model*, resampling the patch_embed
    depth dimension when the model's depth patch size differs from the
    checkpoint's.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to the 3-D checkpoint (cubic patch_size).
    model : ViTv2_3D
        Target model — its ``patch_embed.proj.weight`` shape determines
        the target kernel size.

    Returns
    -------
    (missing, unexpected)
        As returned by :meth:`torch.nn.Module.load_state_dict`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_ps = model.patch_embed.patch_size
    ckpt_ps = ckpt["patch_embed.proj.weight"].shape[2:]

    state = dict(ckpt)
    if model_ps != ckpt_ps:
        assert (
            model_ps[0] == ckpt_ps[0] and model_ps[1] == ckpt_ps[1]
        ), f"Only depth patch size may differ: model={model_ps}, ckpt={ckpt_ps}"
        w_ckpt = ckpt["patch_embed.proj.weight"]
        kd_model = model_ps[2]
        kd_ckpt = ckpt_ps[2]
        print(f"Resampling patch_embed depth: {kd_ckpt} → {kd_model}")
        w_resampled = F.interpolate(
            w_ckpt,
            size=(ckpt_ps[0], ckpt_ps[1], kd_model),
            mode="trilinear",
            align_corners=False,
        )
        state["patch_embed.proj.weight"] = w_resampled
        print(f"  weight: {w_ckpt.shape} → {w_resampled.shape}")

    return model.load_state_dict(state, strict=False)


if __name__ == "__main__":
    ckpt_path_2d = (
        Path(__file__).resolve().parents[3]
        / "checkpoints"
        / "small"
        / "neco"
        / "encoder_teacher.ckpt"
    )
    build_3d_from_2d_checkpoint(ckpt_path_2d)

    model = vitv2_3d_small().cuda().eval()
    ckpt_path_3d = ckpt_path_2d.parent.parent / "neco_3d" / "encoder_teacher.ckpt"
    ckpt_3d = torch.load(ckpt_path_3d, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt_3d, strict=False)
    print(f"Load — missing: {missing}, unexpected: {unexpected}")

    vol = torch.randn(1, 3, 196, 196, 196).cuda()
    with torch.no_grad():
        out = model(vol)
    print(
        f"Output — latent: {out['latent'].shape}, "
        f"patch_latent: {out['patch_latent'].shape}"
    )

    del model, vol

    model_aniso = vitv2_3d_small(patch_size=(14, 14, 2)).cuda().eval()
    missing, unexpected = load_3d_checkpoint_with_anisotropic_patch(
        ckpt_path_3d, model_aniso
    )
    print(f"Load — missing: {missing}, unexpected: {unexpected}")

    vol = torch.randn(1, 3, 196, 196, 28).cuda()
    with torch.no_grad():
        out = model_aniso(vol)
    print(
        f"Output — latent: {out['latent'].shape}, "
        f"patch_latent: {out['patch_latent'].shape}"
    )
