import torch
import torch.nn as nn

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

from med_adapt.utils.config import get_logger

logger = get_logger(__name__)


class AttentionPooling(nn.Module):
    """
    Attention pooling over a sequence of patch tokens.

    A learnable query token attends over the input tokens; the resulting
    weighted sum is the pooled representation. Uses xformers' memory-efficient
    attention if available, otherwise falls back to a manual implementation.

    Input:  x of shape (B, N, D)
    Output: pooled representation of shape (B, D)
    """

    def __init__(
        self,
        num_queries: int,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.classes = num_queries

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.query = nn.Parameter(torch.zeros(1, self.classes, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, D) patch token representations
        mask: optional (B, N) boolean tensor, True = valid, False = padded
        returns: (B, D) attention-pooled representation
        """
        B, N, D = x.shape
        x = self.norm(x)

        q = self.query.expand(B, -1, -1)  # (B, c, D)
        q = self.q_proj(q)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if XFORMERS_AVAILABLE and x.is_cuda:
            out = self._attn_xformers(q, k, v, mask)
        else:
            out = self._attn_manual(q, k, v, mask)

        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out  # (B, c, D)

    def _attn_xformers(self, q, k, v, mask):
        B = q.shape[0]
        # xformers expects (B, N, H, hd)
        q = q.reshape(B, self.classes, self.num_heads, self.head_dim)
        k = k.reshape(B, -1, self.num_heads, self.head_dim)
        v = v.reshape(B, -1, self.num_heads, self.head_dim)

        attn_bias = None
        if mask is not None:
            # build additive bias: 0 for valid, -inf for masked, shape (B, H, 1, N)
            N = k.shape[1]
            attn_bias = torch.zeros(
                B, self.num_heads, 1, N, dtype=q.dtype, device=q.device
            )
            attn_bias.masked_fill_(
                ~mask[:, None, None, :].to(torch.bool), float("-inf")
            )

        out = memory_efficient_attention(
            q, k, v, attn_bias=attn_bias, p=self.dropout if self.training else 0.0
        )  # (B, 1, H, hd)
        return out.reshape(B, self.classes, self.dim)

    def _attn_manual(self, q, k, v, mask):
        B, N = q.shape[0], k.shape[1]
        q = q.reshape(B, self.classes, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, 1, hd)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, N, hd)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, N, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, N)
        if mask is not None:
            attn = attn.masked_fill(
                ~mask[:, None, None, :].to(torch.bool), float("-inf")
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, 1, hd)
        return out.transpose(1, 2).reshape(B, self.classes, self.dim)


if __name__ == "__main__":
    pool = AttentionPooling(2, dim=768, num_heads=8)
    tokens = torch.randn(4, 196, 768)  # B=4, N=196 patches, D=768
    pooled = pool(tokens)  # (4, 768)

    logger.info(f"Input shape: {tokens.shape}, Output shape: {pooled.shape}")
