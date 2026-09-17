"""Batching and DataLoader construction for Scripture-LM training and evaluation."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler

from scripture_lm.data.dataset import ScriptureChunkDataset


def collate_chunks(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of dataset chunk dictionaries into a batched dictionary.

    Args:
        batch: List of dictionaries from ScriptureChunkDataset.__getitem__.

    Returns:
        Batched dictionary with stacked input_ids, target_ids, and exposure aggregates.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    target_ids = torch.stack([item["target_ids"] for item in batch], dim=0)

    batch_raw_chars = sum(int(item["raw_character_count"]) for item in batch)
    batch_target_tokens = sum(int(item["target_token_count"]) for item in batch)
    families = [str(item["family"]) for item in batch]
    chunk_ids = [str(item["chunk_id"]) for item in batch]

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "raw_characters": batch_raw_chars,
        "target_tokens": batch_target_tokens,
        "families": families,
        "chunk_ids": chunk_ids,
    }


def create_dataloader(
    dataset: ScriptureChunkDataset,
    sampler: Sampler[int],
    batch_size: int = 8,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    """Construct a PyTorch DataLoader with deterministic consumption and no dropped batches.

    Args:
        dataset: ScriptureChunkDataset instance.
        sampler: NaturalSampler, TemperatureSampler, or SequentialSampler.
        batch_size: Microbatch size.
        drop_last: Always False by default so partial final batches are never discarded.
        num_workers: Set to 0 for exact deterministic sampler state synchronization.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_chunks,
        drop_last=drop_last,
        num_workers=num_workers,
    )
