"""Multi-head Causal Self-Attention with Rotary Position Embeddings and SDPA."""

from __future__ import annotations

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
        rotary_emb: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings to query and key
        q, k = self.rotary(q, k, seq_len=t)

        # Ensure attention dropout is strictly disabled during eval / generation
        dropout_p = self.attention_dropout if self.training else 0.0

        # Memory-efficient causal Flash Attention without dense mask materialization
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # Recombine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)

        out = self.o_proj(attn_out)
        res: torch.Tensor = self.resid_dropout(out)
        return res
