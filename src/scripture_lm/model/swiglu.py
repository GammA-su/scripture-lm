"""SwiGLU feed-forward network implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripture_lm.model.config import TransformerConfig


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network without linear bias.

    References:
        Shazeer (2020): GLU Variants Improve Transformer.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.mlp_hidden, bias=config.linear_bias)
        self.up_proj = nn.Linear(config.d_model, config.mlp_hidden, bias=config.linear_bias)
        self.down_proj = nn.Linear(config.mlp_hidden, config.d_model, bias=config.linear_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation: down_proj(silu(gate_proj(x)) * up_proj(x))."""
        out: torch.Tensor = self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )
        return out
