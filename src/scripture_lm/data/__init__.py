"""Data package for Scripture-LM: encoding, chunk indexing, datasets, samplers, and batching."""

from scripture_lm.data.batching import collate_chunks, create_dataloader
from scripture_lm.data.chunk_index import (
    ChunkMetadata,
    EncodingProvenance,
    build_chunks_from_stream,
    load_chunk_index,
    save_chunk_index,
)
from scripture_lm.data.dataset import ScriptureChunkDataset
from scripture_lm.data.encode import (
    compute_token_character_credits,
    encode_dataset,
)
from scripture_lm.data.sampler import (
    NaturalSampler,
    SequentialSampler,
    TemperatureSampler,
    calculate_temperature_parameters,
    simulate_sampling,
)

__all__ = [
    "ChunkMetadata",
    "EncodingProvenance",
    "NaturalSampler",
    "ScriptureChunkDataset",
    "SequentialSampler",
    "TemperatureSampler",
    "build_chunks_from_stream",
    "calculate_temperature_parameters",
    "collate_chunks",
    "compute_token_character_credits",
    "create_dataloader",
    "encode_dataset",
    "load_chunk_index",
    "save_chunk_index",
    "simulate_sampling",
]
