"""Multi-head Causal Self-Attention with Rotary Position Embeddings and SDPA."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripture_lm.model.config import TransformerConfig
from scripture_lm.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention utilizing RoPE and Flash SDPA."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
        rotary_emb: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.linear_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.linear_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.linear_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.linear_bias)

        self.resid_dropout = nn.Dropout(config.dropout)

        if rotary_emb is not None:
            self.rotary = rotary_emb
        else:
            self.rotary = RotaryEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_context_length,
                theta=config.rope_theta,
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Any | None = None,
        start_pos: int | None = None,
    ) -> torch.Tensor:
        """Forward pass for causal self-attention with optional KV caching.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            kv_cache: Optional preallocated KVCache instance.
            start_pos: Optional starting position offset for RoPE. Inferred from cache if None.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)

        if start_pos is None:
            start_pos = kv_cache.get_seq_len(self.layer_idx) if kv_cache is not None else 0

        # Apply rotary position embeddings to query and key
        q, k = self.rotary(q, k, seq_len=t, start_pos=start_pos)

        if kv_cache is not None:
            if start_pos == 0:
                # Cache is empty -> prompt prefill with causal mask
                is_causal = True
            else:
                # Cache is non-empty -> single-token incremental decoding only
                if t != 1:
                    raise ValueError(
                        "Non-empty KV cache currently supports single-token decoding only"
                    )
                is_causal = False
            k, v = kv_cache.update(self.layer_idx, k, v)
        else:
            is_causal = True

        # Ensure attention dropout is strictly disabled during eval / generation
        dropout_p = self.attention_dropout if self.training else 0.0

        # Memory-efficient Flash Attention
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Recombine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)

        out = self.o_proj(attn_out)
        res: torch.Tensor = self.resid_dropout(out)
        return res
