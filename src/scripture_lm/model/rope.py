"""Rotary Position Embeddings (RoPE) implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the hidden dimension in half and rotate (-x2, x1)."""
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to Query or Key tensor.

    Args:
        x: Tensor of shape (batch, heads, seq_len, head_dim).
        cos: Cosine tensor of shape (seq_len, head_dim // 2).
        sin: Sine tensor of shape (seq_len, head_dim // 2).

    Returns:
        Rotated tensor of shape (batch, heads, seq_len, head_dim).
    """
    # Duplicate cos and sin to match full head_dim
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)

    # Broadcast to (1, 1, seq_len, head_dim)
    cos_full = cos_full.unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0).to(x.dtype)

    return (x * cos_full) + (rotate_half(x) * sin_full)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module caching frequencies up to max_seq_len."""

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Inverse frequencies: theta^(-2i / dim)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute initial cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to queries and keys of shape (B, heads, T, head_dim)."""
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)
        return q_rot, k_rot
