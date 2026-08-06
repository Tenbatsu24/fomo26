"""Learnable task tokens that attend through the full transformer stack.

Unlike ``AttentionPooling``, which applies a shallow attention layer on top
of the frozen backbone, task tokens are prepended to the patch sequence and
participate in every transformer block. This gives them the opportunity to
learn task-specific representations through the full depth of the network.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn

from med_adapt.utils.config import get_logger

logger = get_logger(__name__)


class TaskTokens(nn.Module):
    """A set of learnable tokens injected into the token sequence.

    Args:
        num_tokens: number of task tokens to create. For classification and
            segmentation this is typically ``num_classes``; for regression it
            is ``1``.
        embed_dim: embedding dimension (must match the backbone).
        insertion: where to insert the tokens in the sequence. Options are
            ``"beginning"`` (after the CLS token, before register tokens) or
            an integer specifying the block index at which to inject them
            (the tokens are added after that block's output).
    """

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        insertion: Literal["beginning", "middle"] = "beginning",
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.insertion = insertion
        self.tokens = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.tokens, std=0.02)
        logger.info(
            "TaskTokens: %d tokens, embed_dim=%d, insertion=%s",
            num_tokens,
            embed_dim,
            insertion,
        )

    def forward(
        self,
        x: torch.Tensor,
        block_index: Optional[int] = None,
        num_blocks: Optional[int] = None,
    ) -> torch.Tensor:
        """Inject task tokens into the token sequence.

        Args:
            x: token tensor of shape ``(B, N, D)``.
            block_index: current block index (used when
                ``insertion=="middle"``).
            num_blocks: total number of blocks (used when
                ``insertion=="middle"``).

        Returns:
            Expanded token tensor with task tokens concatenated.
        """
        if self.insertion == "beginning":
            return torch.cat(
                (x[:, :1], self.tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1
            )
        elif self.insertion == "middle":
            # Insert after the specified block; for simplicity we always
            # insert at the given block_index position in the sequence.
            if block_index is None or num_blocks is None:
                raise ValueError(
                    "block_index and num_blocks are required when insertion='middle'"
                )
            # Insert after block_index % num_blocks worth of tokens — we
            # treat the sequence as a flat list and splice in.
            insert_pos = (block_index % num_blocks) * max(1, x.shape[1] // num_blocks)
            insert_pos = min(insert_pos, x.shape[1])
            return torch.cat(
                (
                    x[:, :insert_pos],
                    self.tokens.expand(x.shape[0], -1, -1),
                    x[:, insert_pos:],
                ),
                dim=1,
            )
        else:
            raise ValueError(f"Unknown insertion mode: {self.insertion}")

    def additional_trainable_keys(self) -> list[str]:
        """Return parameter-name substrings for ``mark_trainable``."""
        return ["task_tokens"]
