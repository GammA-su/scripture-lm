"""Preallocated Key-Value cache for autoregressive Transformer decoding."""

from __future__ import annotations

import torch


class KVCache:
    """Preallocated per-layer Key-Value cache for autoregressive generation.

    Preallocates tensors of shape (batch_size, heads, max_context_length, head_dim)
    to eliminate repeated dynamic memory allocations during incremental decoding.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        heads: int,
        head_dim: int,
        max_context_length: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.heads = heads
        self.head_dim = head_dim
        self.max_context_length = max_context_length
        self.device = torch.device(device)
        self.dtype = dtype

        self.seq_lens: list[int] = [0] * num_layers

        # Preallocate key and value tensors for each transformer layer
        self.k_cache: list[torch.Tensor] = [
            torch.zeros(
                (batch_size, heads, max_context_length, head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(num_layers)
        ]
        self.v_cache: list[torch.Tensor] = [
            torch.zeros(
                (batch_size, heads, max_context_length, head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(num_layers)
        ]
        # Resolve aliases such as "cuda" to the actual allocated device (e.g. "cuda:0").
        if self.k_cache:
            self.device = self.k_cache[0].device

    @property
    def seq_len(self) -> int:
        """Current cached sequence length for the initial layer."""
        return self.seq_lens[0] if self.seq_lens else 0

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Return cached sequence length for a specific layer."""
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        return self.seq_lens[layer_idx]

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert newly projected keys and values into the preallocated cache.

        Args:
            layer_idx: Transformer layer index.
            k: Projected and RoPE-rotated query key tensor of shape (batch, heads, t, head_dim).
            v: Projected value tensor of shape (batch, heads, t, head_dim).

        Returns:
            Tuple of (all_keys, all_values) spanning sequence positions [0, current_seq_len + t).
        """
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

        # 1. Device consistency check
        if k.device != self.device:
            raise ValueError(f"Key tensor device mismatch: expected {self.device}, got {k.device}")
        if v.device != self.device:
            raise ValueError(
                f"Value tensor device mismatch: expected {self.device}, got {v.device}"
            )

        # 2. Dtype consistency check
        if k.dtype != self.dtype:
            raise ValueError(f"Key tensor dtype mismatch: expected {self.dtype}, got {k.dtype}")
        if v.dtype != self.dtype:
            raise ValueError(f"Value tensor dtype mismatch: expected {self.dtype}, got {v.dtype}")

        # 3. Shape validation
        b, h, t, d = k.shape
        if b != self.batch_size:
            raise ValueError(f"Batch size mismatch: expected {self.batch_size}, got {b}")
        if h != self.heads:
            raise ValueError(f"Attention heads mismatch: expected {self.heads}, got {h}")
        if d != self.head_dim:
            raise ValueError(f"Head dimension mismatch: expected {self.head_dim}, got {d}")
        if v.shape != k.shape:
            raise ValueError(f"Key and value shape mismatch: k is {k.shape}, v is {v.shape}")

        # 4. Context capacity check
        curr_len = self.seq_lens[layer_idx]
        if curr_len + t > self.max_context_length:
            raise ValueError(
                f"KVCache context overflow: current seq_len {curr_len} + new tokens {t} "
                f"exceeds max_context_length {self.max_context_length}."
            )

        # 5. Write into preallocated buffer slice
        self.k_cache[layer_idx][:, :, curr_len : curr_len + t] = k
        self.v_cache[layer_idx][:, :, curr_len : curr_len + t] = v
        self.seq_lens[layer_idx] = curr_len + t

        # 6. Slice active history view
        all_k = self.k_cache[layer_idx][:, :, : self.seq_lens[layer_idx]]
        all_v = self.v_cache[layer_idx][:, :, : self.seq_lens[layer_idx]]
        return all_k, all_v

    def reset(self) -> None:
        """Reset sequence length counters to 0, invalidating past cache."""
        for i in range(self.num_layers):
            self.seq_lens[i] = 0
