from typing import Literal
from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from fomo26.models.base import ViTv2
from fomo26.adapter import InputChannelAdapter, AttentionPooling
from fomo26.layers import Block, ScaleBlock, MemEffAttention, LoRAMemEffAttention


class ViTv2Adaption(ViTv2):

    def __init__(
        self,
        med_in_channels: int,
        task: Literal["reg", "cls", "seg"],
        classes,
        minibatch_size: int,
        *args,
        **kwargs,
    ):
        super(ViTv2Adaption, self).__init__(*args, **kwargs)

        self.task = task
        self.minibatch_size = minibatch_size

        if task in ["reg", "cls", "seg"]:
            self.input_adapter = InputChannelAdapter(in_channels=med_in_channels)
        else:
            self.input_adapter = nn.Conv3d(med_in_channels, 3, 1, bias=False)

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
                ScaleBlock(self.embed_dim),
                ScaleBlock(self.embed_dim // 2),
                ScaleBlock(self.embed_dim // 4),
                ScaleBlock(self.embed_dim // 8),
            )
            self.head = nn.Linear(self.embed_dim // 16, classes)

    def forward(self, x, masks=None, last_self_attention=False, **kwargs):
        b, c, d, h, w = x.shape

        adapted_x = self.input_adapter(x)

        # go from 3d to folding depth into batch axes
        reshaped_x = rearrange(adapted_x, "b c d ... -> (b d) c ...")

        if self.minibatch_size > 0:
            outs = {"patch_latent": [], "latent": []}
            for start in range(0, b * d, self.minibatch_size):
                minibatch = reshaped_x[start : start + self.minibatch_size]
                out = super().forward(
                    minibatch,
                    masks=masks,
                    last_self_attention=last_self_attention,
                    **kwargs,
                )
                outs["patch_latent"].append(out["patch_latent"])
                outs["latent"].append(out["latent"])
            latents = torch.concat(outs["latent"], dim=0)
            patch_latents = torch.concat(outs["patch_latent"], dim=0)
        else:
            out = super().forward(
                reshaped_x,
                masks=masks,
                last_self_attention=last_self_attention,
                **kwargs,
            )
            latents = out["latent"]
            patch_latents = out["patch_latent"]

        if self.task in ["reg", "cls"]:
            reshaped_out = rearrange(patch_latents, "(b d) n c -> b (d n) c", b=b, d=d)
            attnended = self.attn_pool(reshaped_out)
            return self.head(attnended)
        elif self.task == "seg":
            # make spatial
            hp, wp = h // self.patch_size, w // self.patch_size

            spatial = rearrange(patch_latents, "b (hp wp) c -> b c hp wp", hp=hp, wp=wp)

            upscaled = self.upscale(spatial)

            # final classifier
            rearranged = rearrange(upscaled, "b c ... -> b ... c")
            logits = rearrange(
                self.head(rearranged), "(b d) ... c -> b c d ...", b=b, d=d
            )
            return F.interpolate(
                logits, size=(d, h, w), mode="trilinear", align_corners=False
            )
        else:
            # reshaped latents
            pooled = rearrange(latents, "(b d) c -> b d c", b=b, d=d).mean(dim=-2)

            return pooled


def vitv2_tiny(patch_size=16, num_register_tokens=4, lora=False, **kwargs):
    model = ViTv2Adaption(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vitv2_small(patch_size=16, num_register_tokens=4, lora=False, **kwargs):
    model = ViTv2Adaption(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vitv2_base(patch_size=16, num_register_tokens=4, lora=False, **kwargs):
    model = ViTv2Adaption(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vitv2_large(patch_size=16, num_register_tokens=4, lora=False, **kwargs):
    model = ViTv2Adaption(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(
            Block, attn_class=LoRAMemEffAttention if lora else MemEffAttention
        ),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    import torch

    from fomo26.utils.trainable import mark_trainable

    # Quick configuration settings for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on device: {device}\n" + "=" * 50)

    # 1. Define dummy input dimensions
    # Typical 3D medical volume chunk (Batch, Channels, Depth/Slices, Height, Width)
    B, C_in, D, H, W = 4, 1, 128, 224, 224
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
        model = vitv2_tiny(
            med_in_channels=C_in,
            task=task,
            classes=classes,
            patch_size=patch_size,
            num_register_tokens=4,
            lora=False,
            minibatch_size=-1,
        ).to(device)
        trainable_names, _ = mark_trainable(model)

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
