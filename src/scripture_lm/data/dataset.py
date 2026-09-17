"""PyTorch Dataset implementation memory-mapping chunked Scripture-LM binary streams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from scripture_lm.data.chunk_index import ChunkMetadata
from scripture_lm.tokenization.base import PAD_ID


class ScriptureChunkDataset(Dataset[dict[str, Any]]):
    """Memory-mapped dataset serving fixed-context training chunks with target masking."""

    def __init__(
        self,
        chunks: list[ChunkMetadata],
        base_dir: Path | str = Path("data/encoded"),
        context_length: int | None = None,
    ) -> None:
        """Initialize dataset from a list of chunk metadata.

        Args:
            chunks: List of ChunkMetadata objects.
            base_dir: Base directory where bin_path relative paths are located.
            context_length: Context length L. Inferred from chunks if not provided.
        """
        self.chunks = chunks
        self.base_dir = Path(base_dir)

        if context_length is not None:
            self.context_length = context_length
        elif chunks:
            # Infer from maximum valid token count minus 1 (window_size = L + 1)
            max_v = max(c.valid_token_count for c in chunks)
            self.context_length = max_v - 1
        else:
            self.context_length = 512

        self.window_size = self.context_length + 1
        self._memmaps: dict[str, np.memmap] = {}

    def _get_memmap(self, rel_path: str) -> np.memmap:
        """Retrieve or open a read-only memory-mapped uint16 array."""
        if rel_path not in self._memmaps:
            full_path = self.base_dir / rel_path
            if not full_path.is_file():
                # Try path relative to current working directory as fallback
                fallback = Path(rel_path)
                if fallback.is_file():
                    full_path = fallback
                else:
                    raise FileNotFoundError(f"Binary token file not found: {full_path}")
            self._memmaps[rel_path] = np.memmap(full_path, dtype=np.uint16, mode="r")
        return self._memmaps[rel_path]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chunk = self.chunks[idx]
        memmap = self._get_memmap(chunk.bin_path)

        start = chunk.token_start
        v_count = chunk.valid_token_count
        raw_tokens = np.array(memmap[start : start + v_count], dtype=np.int64)

        # Allocate full window of length window_size = context_length + 1 initialized with PAD_ID
        tokens = np.full(self.window_size, PAD_ID, dtype=np.int64)
        tokens[:v_count] = raw_tokens

        x_np = tokens[: self.context_length]
        y_np = tokens[1 : self.window_size].copy()

        # Mask padded target positions with -100 (PyTorch CrossEntropyLoss ignore_index)
        # Valid targets correspond to tokens at indices 1 .. v_count - 1
        # Target indices >= v_count - 1 are padding
        valid_targets_count = max(0, v_count - 1)
        if valid_targets_count < self.context_length:
            y_np[valid_targets_count:] = -100

        return {
            "input_ids": torch.from_numpy(x_np),
            "target_ids": torch.from_numpy(y_np),
            "valid_token_count": v_count,
            "target_token_count": valid_targets_count,
            "raw_character_count": chunk.raw_character_count,
            "family": chunk.family,
            "chunk_id": chunk.chunk_id,
        }

    def close(self) -> None:
        """Close open memory maps."""
        self._memmaps.clear()
