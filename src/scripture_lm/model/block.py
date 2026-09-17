"""Transformer decoder block with pre-norm RMSNorm, Attention, and SwiGLU."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from scripture_lm.model.attention import CausalSelfAttention
from scripture_lm.model.config import TransformerConfig
from scripture_lm.model.rmsnorm import RMSNorm
from scripture_lm.model.rope import RotaryEmbedding
from scripture_lm.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block combining RMSNorm, CausalSelfAttention, and SwiGLU."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
        rotary_emb: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_eps)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx, rotary_emb=rotary_emb)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.rms_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Any | None = None,
        start_pos: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with pre-layer norm residual connections."""
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x
