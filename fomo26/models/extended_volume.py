from typing import Literal
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from fomo26.models.base import ViTv2
from fomo26.adapter import PatchEmbed3D, AttentionPooling
from fomo26.layers import Block, ScaleBlock, MemEffAttention, LoRAMemEffAttention


class ViTv2Adaption3D(ViTv2):

    def __init__(
        self,
        volume_size: tuple[int, int, int],
        volume_patch_size: tuple[int, int, int],
        med_in_channels: int,
        task: Literal["reg", "cls", "seg"],
        classes: int,
        *args,
        **kwargs,
    ):
        super(ViTv2Adaption3D, self).__init__(*args, **kwargs)

        self.task = task

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

        if task in ["reg", "cls"]:
            self.attn_pool = AttentionPooling(self.embed_dim)
            if task == "cls":
                self.head = nn.Linear(self.embed_dim, classes)
            elif task == "reg":
                self.head = nn.Linear(self.embed_dim, 1)
            else:
                self.head = nn.Identity()
        else:
            self.upscale = nn.Sequential(
                ScaleBlock(self.embed_dim, conv_type="3d"),
                ScaleBlock(self.embed_dim // 2, conv_type="3d"),
                ScaleBlock(self.embed_dim // 4, conv_type="3d"),
                ScaleBlock(self.embed_dim // 8, conv_type="3d"),
            )
            self.head = nn.Linear(self.embed_dim // 16, classes)

    def interpolate_pos_encoding(self, x, h, w, d):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        # If the patch count matches exactly and dimensions match, skip interpolation
        if npatch == N and h == w == d:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        # self.patch_size is assumed to be a 3-tuple (patch_h, patch_w, patch_d)
        h0 = h // self.patch_size[0]
        w0 = w // self.patch_size[1]
        d0 = d // self.patch_size[2]

        # Avoid floating point errors during scale-factor mapping
        h0 = h0 + self.interpolate_offset
        w0 = w0 + self.interpolate_offset
        d0 = d0 + self.interpolate_offset

        ph, pw, pd = self.patch_embed.patches_resolution

        sh = float(h0) / ph
        sw = float(w0) / pw
        sd = float(d0) / pd

        # Reshape from flattened patches to structural HWD grid format:
        # (1, cube_root_N, cube_root_N, cube_root_N, dim) -> Permute to PyTorch standard Conv3D shape: (1, dim, H, W, D)
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

        # Revert permutation back from (1, dim, H, W, D) to (1, H, W, D, dim) and flatten to (1, N_new, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(self, x, masks=None):
        # Expected input sequence: B, C, H, W, D
        B, nc, h, w, d = x.shape
        x = self.patch_embed(x)  # Outputs flattened shape: (B, N, E)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # Pass HWD sequence parameters down to interpolation module
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

    def forward(self, x, masks=None, last_self_attention=False, **kwargs):
        b, c, h, w, d = x.shape

        out = super().forward(
            x,
            masks=masks,
            last_self_attention=last_self_attention,
            **kwargs,
        )
        latents = out["latent"]
        patch_latents = out["patch_latent"]

        if self.task in ["reg", "cls"]:
            return self.head(latents)
        elif self.task == "seg":
            psh, psw, psd = self.patch_size
            hp, wp, dp = h // psh, w // psw, d // psd

            spatial = rearrange(
                patch_latents, "b (hp wp dp) c -> b c hp wp dp", hp=hp, wp=wp, dp=dp
            )

            upscaled = self.upscale(spatial)

            # final classifier
            rearranged = rearrange(upscaled, "b c ... -> b ... c")
            logits = rearrange(self.head(rearranged), "b ... c -> b c ...")
            return F.interpolate(
                logits, size=(h, w, d), mode="trilinear", align_corners=False
            )
        else:
            return latents

    def additional_trainable(self):
        return ["patch_embed", "pos_embed"]

    def do_not_load(self):
        return ["pos_embed", "patch_embed"]


def vitv2_a_3d_tiny(
    volume_size=(224, 224, 28), volume_patch_size=(14, 14, 4), lora=False, **kwargs
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


def vitv2_a_3d_small(
    volume_size=(224, 224, 28), volume_patch_size=(14, 14, 4), lora=False, **kwargs
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


def vitv2_a_3d_base(
    volume_size=(224, 224, 28), volume_patch_size=(14, 14, 4), lora=False, **kwargs
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


def vitv2_a_3d_large(
    volume_size=(224, 224, 28), volume_patch_size=(14, 14, 4), lora=False, **kwargs
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
    from fomo26.utils.trainable import mark_trainable

    # Quick configuration settings for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on device: {device}\n" + "=" * 50)

    # 1. Define dummy input dimensions
    # Typical 3D medical volume chunk (Batch, Channels, Depth/Slices, Height, Width)
    B, C_in, D, H, W = 4, 1, 224, 224, 128
    classes = 3
    patch_size = 14

    # Generate dummy input tensor
    dummy_input = torch.randn(B, C_in, D, H, W).to(device)
    print(f"Input Shape: {dummy_input.shape} (B={B}, C={C_in}, D={D}, H={H}, W={W})")
    print("=" * 50)

    # 2. Define tasks to test
    tasks = ["cls", "reg", "seg", "none"]

    for task in tasks:
        print(f"\n--- Testing Task: '{task.upper()}' ---")
        # Instantiate the model using the tiny configuration
        model = vitv2_a_3d_tiny(
            volume_size=(224, 224, 128),
            volume_patch_size=(14, 14, 8),
            med_in_channels=C_in,
            task=task,
            classes=classes,
            lora=True,
        ).to(device)
        trainable_names, _ = mark_trainable(
            model, additional_keys=model.additional_trainable()
        )

        # Calculate parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,} = {trainable_params/total_params:.2%})"
        )

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        print(f"v Success! Output shape: {output.shape}")
