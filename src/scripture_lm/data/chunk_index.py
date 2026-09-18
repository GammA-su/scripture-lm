"""Persistent chunk indexing and exposure metadata for Scripture-LM datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ENCODING_PROVENANCE_FILENAME = "encoding_provenance.json"


def encoding_provenance_path(encoded_dir: Path) -> Path:
    """Use the encoder's canonical filename, falling back to older dataset metadata.

    Canonical provenance takes precedence when both files exist; an obsolete legacy
    snapshot must not shadow a newly encoded dataset. Invalid canonical data is never
    silently replaced with legacy data.
    """
    canonical = encoded_dir / ENCODING_PROVENANCE_FILENAME
    legacy = encoded_dir / "encoding_metadata.json"
    return canonical if canonical.is_file() or not legacy.is_file() else legacy


class ChunkMetadata(BaseModel):
    """Metadata for a single fixed-context training, validation, or test chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    split: Literal["train", "validation", "test"]
    family: str
    tokenizer: Literal["bpe", "character"]
    bin_path: str
    token_start: int
    valid_token_count: int
    raw_character_count: int
    document_ids: list[str]

    @property
    def target_token_count(self) -> int:
        """Number of valid target tokens in this chunk (excluding input 0 and padding)."""
        return max(0, self.valid_token_count - 1)


class EncodingProvenance(BaseModel):
    """Provenance metadata binding encoded binary streams to corpus, split, and tokenizer."""

    model_config = ConfigDict(extra="forbid")

    tokenizer_type: str
    context_length: int
    chunk_length: int
    corpus_fingerprint: str
    normalization_fingerprint: str
    split_manifest_hash: str
    tokenizer_artifact_sha256: str
    total_chunks: dict[str, int] = Field(default_factory=dict)
    natural_train_target_characters: int = 0
    files: dict[str, dict[str, Any]] = Field(default_factory=dict)


def build_chunks_from_stream(
    token_count: int,
    char_credits: np.ndarray,
    doc_spans: list[tuple[str, int, int]],
    context_length: int,
    split: Literal["train", "validation", "test"],
    family: str,
    tokenizer_type: Literal["bpe", "character"],
    bin_path: str,
    start_chunk_idx: int = 0,
) -> tuple[list[ChunkMetadata], int]:
    """Generate fixed-context chunks from an encoded token stream with stride L.

    Args:
        token_count: Total tokens in the stream.
        char_credits: Array of character credits for each token (length == token_count).
        doc_spans: List of (doc_id, start_token_idx, end_token_idx).
        context_length: Model context length L (e.g. 512 for BPE, 2048 for Character).
        split: "train", "validation", or "test".
        family: Scripture family name (e.g. "hebrew_bible").
        tokenizer_type: "bpe" or "character".
        bin_path: Relative path to the binary stream file.
        start_chunk_idx: Starting counter for chunk IDs.

    Returns:
        tuple of (chunks, next_chunk_idx)
    """
    if token_count < 2:
        return [], start_chunk_idx

    window_size = context_length + 1
    stride = context_length

    # Compute prefix sums of character credits for O(1) interval sum
    char_prefix = np.zeros(token_count + 1, dtype=np.int64)
    np.cumsum(char_credits, out=char_prefix[1:])

    chunks: list[ChunkMetadata] = []
    chunk_idx = start_chunk_idx

    s = 0
    while s < token_count:
        remaining = token_count - s
        if remaining <= 1 and s > 0:
            # The single remaining token s was already the final target in chunk [s-L, s+1).
            # No new target tokens exist.
            break

        valid_count = min(window_size, remaining)
        if valid_count < 2:
            break

        # Targets are in index range [s + 1, s + valid_count)
        target_start = s + 1
        target_end = s + valid_count
        raw_char_count = int(char_prefix[target_end] - char_prefix[target_start])

        # Determine document provenance overlapping this chunk [s, s + valid_count)
        chunk_window_end = s + valid_count
        overlapping_docs: list[str] = []
        for doc_id, doc_start, doc_end in doc_spans:
            if not (doc_end <= s or doc_start >= chunk_window_end):
                overlapping_docs.append(doc_id)

        chunk_id = f"{split}_{family}_{chunk_idx:06d}"
        chunks.append(
            ChunkMetadata(
                chunk_id=chunk_id,
                split=split,
                family=family,
                tokenizer=tokenizer_type,
                bin_path=bin_path,
                token_start=s,
                valid_token_count=valid_count,
                raw_character_count=raw_char_count,
                document_ids=overlapping_docs,
            )
        )
        chunk_idx += 1
        s += stride

    return chunks, chunk_idx


def save_chunk_index(chunks: list[ChunkMetadata], path: Path | str) -> None:
    """Save a list of chunk metadata to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [chunk.model_dump() for chunk in chunks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_chunk_index(path: Path | str) -> list[ChunkMetadata]:
    """Load a list of chunk metadata from JSON."""
    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Chunk index not found: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ChunkMetadata.model_validate(item) for item in data]
