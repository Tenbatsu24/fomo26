# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import math

import torch

from torch import nn
from torch import Tensor

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, return_attn=False) -> Tensor:
        """
        Adapted from https://gitlab.com/ziegleto-machine-learning/dino/-/tree/main/
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Adaptation for returing attentions
        if return_attn:
            return attn
        return x


class MemEffAttention(Attention):
    """
    Adapted from https://gitlab.com/ziegleto-machine-learning/dino/-/tree/main/
    """

    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            # Change this line
            # return super().forward(x)
            # Adaptation for returing attentions
            return super().forward(x, return_attn)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        if return_attn:
            # Support for XFORMERS to return attention
            # Adapted from https://github.com/facebookresearch/dinov2/issues/90#issuecomment-1574001076
            attn = x.permute(0, 2, 1, 3) @ v.permute(0, 2, 3, 1)
            return attn
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a frozen base weight plus a
    trainable low-rank update: y = base(x) + (alpha / r) * B(A(x)).

    The base linear layer's weight/bias are frozen by default (standard LoRA
    usage: adapt a pretrained model cheaply). Set `freeze_base=False` to also
    fine-tune the base weights alongside the LoRA path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 4,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha if lora_alpha is not None else r
        self.scaling = self.lora_alpha / self.r

        self.base = nn.Linear(in_features, out_features, bias=bias)
        if freeze_base:
            self.base.weight.requires_grad = False
            if self.base.bias is not None:
                self.base.bias.requires_grad = False

        self.lora_dropout = (
            nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Standard LoRA init: A ~ Kaiming, B = 0 -> LoRA path starts as a no-op
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        lora_update = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * lora_update

    @torch.no_grad()
    def merge_into_base(self):
        """Fold the LoRA update into the base weight (for inference-time export)."""
        self.base.weight += self.scaling * (self.lora_B @ self.lora_A)


class LoRAAttention(nn.Module):
    """
    LoRA version of `Attention`: same math, but `qkv` and `proj` are
    `LoRALinear` instead of `nn.Linear`, so only the low-rank adapters (and
    optionally biases) are trained while the base projection weights stay
    frozen.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lora_r: int = 4,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = LoRALinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LoRALinear(
            dim,
            dim,
            bias=proj_bias,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, return_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return attn
        return x


class LoRAMemEffAttention(LoRAAttention):
    """
    LoRA + xFormers memory-efficient attention. Mirrors `MemEffAttention`
    but built on top of `LoRAAttention` so `qkv`/`proj` carry LoRA adapters.
    """

    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x, return_attn)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        if return_attn:
            attn = x.permute(0, 2, 1, 3) @ v.permute(0, 2, 3, 1)
            return attn
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _att = MemEffAttention(dim=32, num_heads=4).to(device)
    print(_att(torch.randn(4, 16, 32, device=device), return_attn=True).shape)
    print(_att(torch.randn(4, 16, 32, device=device)).shape)

    # LoRA version: same output shapes, but only a small fraction of params trainable
    _lora_att = LoRAMemEffAttention(dim=32, num_heads=4, lora_r=4).to(device)
    print(_lora_att(torch.randn(4, 16, 32, device=device), return_attn=True).shape)
    print(_lora_att(torch.randn(4, 16, 32, device=device)).shape)

    mark_only_lora_as_trainable(_lora_att)
    total = sum(p.numel() for p in _lora_att.parameters())
    trainable = sum(p.numel() for p in _lora_att.parameters() if p.requires_grad)
    print(
        f"LoRA trainable params: {trainable}/{total} ({100 * trainable / total:.2f}%)"
    )
