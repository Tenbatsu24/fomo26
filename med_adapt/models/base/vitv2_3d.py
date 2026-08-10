"""3-D extension of the 2-D ViT base model.

Inherits from :class:`ViTv2` and replaces the 2-D patch embedding with a
3-D counterpart while keeping every transformer block unchanged.  The
positional embedding is expanded from a 2-D grid into a cubic 3-D grid
by repeating each 2-D slice along the new depth axis.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from med_adapt.models.base.vitv2 import ViTv2, named_apply, init_weights_vit
from med_adapt.registry import register_model


class PatchEmbed3D(nn.Module):
    """3-D spatial patch embedding: ``(B, C, H, W, D) → (B, N, E)``.

    The convolution kernel is cubic with size ``patch_size × patch_size ×
    patch_size`` and stride equal to the kernel size, so the output grid
    is ``(H//ps, W//ps, D//ps)``.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        patch_size: int | Tuple[int, int, int],
        in_chans: int,
        embed_dim: int,
        norm_layer=nn.Identity,
    ):
        super().__init__()
        self.img_size = img_size
        # Only the depth component may differ from H/W; enforce h==w.
        ps = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        assert ps[0] == ps[1], "patch_size[0] (height) must equal patch_size[1] (width)"
        self.patch_size = ps
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        h, w, d = img_size
        ph, pw, pd = h // ps[0], w // ps[1], d // ps[2]
        self.patches_resolution = (ph, pw, pd)
        self.num_patches = ph * pw * pd

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=ps,
            stride=ps,
        )
        # No norm layer — matches the original 2-D checkpoint which
        # also omits patch_embed.norm weights.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W, D]
        x = self.proj(x)  # [B, E, ph, pw, pd]
        x = x.flatten(2)  # [B, E, ph·pw·pd]
        x = x.transpose(1, 2)  # [B, ph·pw·pd, E]
        return x


class ViTv2_3D(ViTv2):
    """3-D ViT that processes volumes ``(B, C, H, W, D)`` directly."""

    def __init__(self, *args, **kwargs):
        # The base ViTv2 expects an int for patch_size (2-D PatchEmbed uses
        # make_2tuple).  We store the real (possibly 3-D) patch_size before
        # calling super, then overwrite patch_embed afterwards.
        ps = kwargs.get("patch_size", 14)
        if isinstance(ps, tuple):
            kwargs["patch_size"] = ps[0]  # pass the H/W component to base
        super().__init__(*args, **kwargs)
        # Replace the 2-D patch embed with a 3-D one.
        h = self.img_size
        if isinstance(ps, int):
            ps = (ps, ps, ps)
        self.patch_size = ps  # restore the real patch_size
        self.patch_embed = PatchEmbed3D(
            img_size=(h, h, h),
            patch_size=ps,
            in_chans=3,
            embed_dim=self.embed_dim,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
        )
        # pos_embed is always initialised on the cubic 37×37×37 grid so that
        # the checkpoint can always be loaded directly.  Interpolation at
        # forward time handles any anisotropic input or anisotropic ps.
        self._pos_embed_grid_size = 37  # matches the teacher checkpoint
        # Re-create pos_embed on the cubic grid
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self._pos_embed_grid_size**3 + self.num_tokens,
                self.embed_dim,
            )
        )

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

        ph, pw, pd = self.patch_embed.patches_resolution
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


def build_3d_from_2d_checkpoint(ckpt_path: str) -> dict:
    import pathlib

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # --- patch_embed: conv2d → conv3d via (a+s+o)/3 ------------------------
    w2d = ckpt["patch_embed.proj.weight"]  # [out, in, kH, kW]
    bias2d = ckpt["patch_embed.proj.bias"]

    a = w2d.unsqueeze(2)  # [out, in, 1, kH, kW]
    s = w2d.unsqueeze(3)  # [out, in, kH, 1, kW]
    o = w2d.unsqueeze(4)  # [out, in, kH, kW, 1]
    w3d = (a + s + o) / 3
    # w3d = w2d.unsqueeze(-1).repeat(1, 1, 1, 1, w2d.size(2))

    # --- pos_embed: 2-D grid → cubic 3-D grid ------------------------------
    pos_2d = ckpt["pos_embed"]
    cls = pos_2d[:, :1, :]
    patches = pos_2d[:, 1:, :]
    # The 2-D grid is sqrt(num_patches) × sqrt(num_patches)
    num_patches_2d = patches.shape[1]
    grid_size = int(math.sqrt(num_patches_2d))
    assert (
        grid_size * grid_size == num_patches_2d
    ), f"pos_embed patch count {num_patches_2d} is not a perfect square"
    patches_3d = (
        patches.view(1, grid_size, grid_size, -1)
        .unsqueeze(3)  # [1, H, W, 1, D]
        .repeat(1, 1, 1, grid_size, 1)  # [1, H, W, H, D]
        .view(1, grid_size * grid_size * grid_size, -1)
    )
    pos_3d = torch.cat([cls, patches_3d], dim=1)

    # --- Assemble new state dict -------------------------------------------
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

    # --- Save --------------------------------------------------------------
    out_dir = pathlib.Path(ckpt_path).parent.parent / f"neco_3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "encoder_teacher.ckpt"
    torch.save(ckpt_3d, out_path)
    print(f"Saved 3-D checkpoint → {out_path}")
    print(f"  patch_embed.proj.weight: {w2d.shape} → {w3d.shape}")
    print(f"  pos_embed: {pos_2d.shape} → {pos_3d.shape}")
    return ckpt_3d


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
    ckpt_path: str,
    model: nn.Module,
) -> tuple[list[str], list[str]]:
    """Load a 3-D checkpoint into *model*, resampling the patch_embed
    depth dimension when the model's depth patch size differs from the
    checkpoint's.

    Parameters
    ----------
    ckpt_path : str
        Path to the 3-D checkpoint (cubic patch_size).
    model : ViTv2_3D
        Target model — its ``patch_embed.proj.weight`` shape determines
        the target kernel size.
    depth_patch_size : int, optional
        If given, override the model's depth patch size.  Useful when the
        registered factory only accepts an integer.

    Returns
    -------
    (missing, unexpected)
        As returned by :meth:`torch.nn.Module.load_state_dict`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_ps = model.patch_embed.patch_size  # e.g. (14, 14, 2)
    ckpt_ps = ckpt["patch_embed.proj.weight"].shape[2:]  # e.g. (14, 14, 14)

    state = dict(ckpt)
    if model_ps != ckpt_ps:
        # Only the depth component may differ.
        assert (
            model_ps[0] == ckpt_ps[0] and model_ps[1] == ckpt_ps[1]
        ), f"Only depth patch size may differ: model={model_ps}, ckpt={ckpt_ps}"
        w_ckpt = ckpt["patch_embed.proj.weight"]  # [out, in, kh, kw, kd_ckpt]
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
    # Build and save the 3-D checkpoint from the 2-D source.
    ckpt_path_2d = "./checkpoints/small/neco/encoder_teacher.ckpt"
    build_3d_from_2d_checkpoint(ckpt_path_2d)

    # Quick smoke-test: load the new model and the new checkpoint.
    model = vitv2_3d_small().cuda().eval()

    ckpt_path_3d = "./checkpoints/small/neco_3d/encoder_teacher.ckpt"
    ckpt_3d = torch.load(ckpt_path_3d, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt_3d, strict=False)
    print(f"Load — missing: {missing}, unexpected: {unexpected}")

    # Forward pass with a small cubic volume.
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

    # Forward pass with a small cubic volume.
    vol = torch.randn(1, 3, 196, 196, 28).cuda()
    with torch.no_grad():
        out = model_aniso(vol)
    print(
        f"Output — latent: {out['latent'].shape}, "
        f"patch_latent: {out['patch_latent'].shape}"
    )
