import torch
import torch.nn as nn

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class AttentionPooledHead(nn.Module):
    """
    Class-query attention classifier.

    Each class owns a learnable query token. The query attends over the
    patch/token sequence and produces a class-specific representation.
    A shared scalar scorer converts each representation into a logit.

    Input:
        x: (B, N, D)

    Output:
        logits: (B, C)
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        # One query per class
        self.query = nn.Parameter(torch.zeros(1, num_classes, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.norm = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Shared scalar scorer
        self.classifier = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)

        Returns:
            logits: (B, C)
        """
        B, N, D = x.shape

        x = self.norm(x)

        q = self.q_proj(self.query.expand(B, -1, -1))  # (B, C, D)

        k = self.k_proj(x)  # (B, N, D)
        v = self.v_proj(x)  # (B, N, D)

        if XFORMERS_AVAILABLE and x.is_cuda:
            cls_repr = self._attn_xformers(q, k, v)
        else:
            cls_repr = self._attn_manual(q, k, v)

        cls_repr = self.out_proj(cls_repr)
        cls_repr = self.proj_drop(cls_repr)

        logits = self.classifier(cls_repr).squeeze(-1)

        return logits

    def _attn_xformers(self, q, k, v):
        B = q.shape[0]

        q = q.reshape(
            B,
            self.num_classes,
            self.num_heads,
            self.head_dim,
        )

        k = k.reshape(
            B,
            -1,
            self.num_heads,
            self.head_dim,
        )

        v = v.reshape(
            B,
            -1,
            self.num_heads,
            self.head_dim,
        )

        out = memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            p=self.dropout if self.training else 0.0,
        )

        return out.reshape(B, self.num_classes, self.dim)

    def _attn_manual(self, q, k, v):
        B, N = q.shape[0], k.shape[1]

        q = q.reshape(
            B,
            self.num_classes,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        return out.transpose(1, 2).reshape(
            B,
            self.num_classes,
            self.dim,
        )


if __name__ == "__main__":
    _head = AttentionPooledHead(384, 1, 4, qkv_bias=True, dropout=0.1)

    print(_head(torch.randn(2, 196, 384)).shape)
