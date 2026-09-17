"""Common tokenizer interface, special tokens, and provenance verification."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.corpus.normalize import CorpusLock
from scripture_lm.corpus.split import SplitManifest

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKEN_IDS = {
    PAD_TOKEN: PAD_ID,
    BOS_TOKEN: BOS_ID,
    EOS_TOKEN: EOS_ID,
    UNK_TOKEN: UNK_ID,
}


class TokenizerMetadata(BaseModel):
    """Provenance metadata binding a tokenizer artifact to the exact corpus and split."""

    model_config = ConfigDict(extra="forbid")

    tokenizer_type: str
    vocab_size: int
    corpus_fingerprint: str
    normalization_fingerprint: str
    split_manifest_hash: str
    training_document_ids: list[str]
    tokenizer_artifact_sha256: str
    tokenizers_library_version: str
    special_token_ids: dict[str, int] = Field(default_factory=lambda: SPECIAL_TOKEN_IDS)
    config: dict[str, Any] = Field(default_factory=dict)


class BaseTokenizer(ABC):
    """Abstract base class defining standard tokenizer interface for Scripture-LM."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""

    @property
    @abstractmethod
    def tokenizer_type(self) -> str:
        """Type of tokenizer ('bpe' or 'character')."""

    @property
    def pad_id(self) -> int:
        """Token ID for padding (<pad>)."""
        return PAD_ID

    @property
    def bos_id(self) -> int:
        """Token ID for beginning of sequence (<bos>)."""
        return BOS_ID

    @property
    def eos_id(self) -> int:
        """Token ID for end of sequence (<eos>)."""
        return EOS_ID

    @property
    def unk_id(self) -> int:
        """Token ID for unknown token (<unk>)."""
        return UNK_ID

    @abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text into token IDs with optional BOS/EOS boundary tokens."""

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to a text string."""

    @abstractmethod
    def id_to_token(self, token_id: int) -> str | None:
        """Convert integer ID to token string representation."""

    @abstractmethod
    def token_to_id(self, token: str) -> int | None:
        """Convert token string representation to integer ID."""

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist tokenizer vocabulary/merges to disk."""


def compute_manifest_sha256(path: Path) -> str:
    """Compute SHA-256 of split_manifest.json content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_corpus_and_split_integrity(
    split_manifest: SplitManifest,
    corpus_lock: CorpusLock,
    normalized_dir: Path = Path("data/normalized"),
) -> None:
    """Verify fingerprints and confirm every normalized file on disk matches corpus_lock."""
    # 1. Verify split fingerprints match corpus lock
    if split_manifest.corpus_fingerprint != corpus_lock.corpus_fingerprint:
        raise ValueError(
            f"Split manifest corpus fingerprint mismatch: "
            f"split has {split_manifest.corpus_fingerprint[:12]}..., "
            f"lock has {corpus_lock.corpus_fingerprint[:12]}.... "
            "Re-run 'scripture-lm corpus prepare'."
        )

    if split_manifest.normalization_fingerprint != corpus_lock.normalization_fingerprint:
        raise ValueError(
            f"Split manifest normalization fingerprint mismatch: "
            f"split has {split_manifest.normalization_fingerprint[:12]}..., "
            f"lock has {corpus_lock.normalization_fingerprint[:12]}.... "
            "Re-run 'scripture-lm corpus prepare'."
        )

    # 2. Build provenance lookup
    prov_map = {doc.document_id: doc for doc in corpus_lock.documents}

    # 3. Verify all split documents exist on disk and match normalized_sha256
    all_doc_ids = split_manifest.train + split_manifest.validation + split_manifest.test
    for doc_id in all_doc_ids:
        if doc_id not in prov_map:
            raise ValueError(
                f"Document '{doc_id}' in split manifest is not found in corpus_lock.json. "
                "Re-run 'scripture-lm corpus prepare'."
            )
        prov = prov_map[doc_id]
        file_path = normalized_dir / prov.family / f"{doc_id}.txt"
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Normalized corpus file missing for '{doc_id}': '{file_path}'. "
                "Re-run 'scripture-lm corpus prepare'."
            )

        actual_sha = compute_file_sha256(file_path)
        if actual_sha.lower() != prov.normalized_sha256.lower():
            raise ValueError(
                f"Normalized corpus file '{file_path}' does not match corpus_lock.json. "
                f"Expected SHA-256 {prov.normalized_sha256[:12]}..., got {actual_sha[:12]}.... "
                "Re-run 'scripture-lm corpus prepare'."
            )
