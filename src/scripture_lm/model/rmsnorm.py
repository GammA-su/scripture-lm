"""Root Mean Square Layer Normalization (RMSNorm) implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization without learnable bias.

    References:
        Zhang & Sennrich (2019): Root Mean Square Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to input tensor x of shape (..., dim)."""
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        norm_x = (x_float * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return norm_x * self.weight

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"
