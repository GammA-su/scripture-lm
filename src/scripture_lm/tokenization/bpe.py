"""Byte-level BPE tokenizer implementation using Hugging Face tokenizers."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import tokenizers
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.corpus.normalize import CorpusLock
from scripture_lm.corpus.split import SplitManifest
from scripture_lm.tokenization.base import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    BaseTokenizer,
    TokenizerMetadata,
    compute_manifest_sha256,
    verify_corpus_and_split_integrity,
)


class BPETokenizer(BaseTokenizer):
    """Byte-level BPE tokenizer trained completely from scratch."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return self._tokenizer.get_vocab_size()

    @property
    def tokenizer_type(self) -> str:
        """Type identifier."""
        return "bpe"

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text into token IDs with NFC canonicalization."""
        # Always canonicalize to NFC prior to tokenization
        normalized = unicodedata.normalize("NFC", text)
        encoding = self._tokenizer.encode(normalized)
        ids = list(encoding.ids)

        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to a string."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def id_to_token(self, token_id: int) -> str | None:
        """Convert integer ID to token string."""
        return self._tokenizer.id_to_token(token_id)

    def token_to_id(self, token: str) -> int | None:
        """Convert token string to integer ID."""
        return self._tokenizer.token_to_id(token)

    def save(self, path: Path | str) -> None:
        """Save tokenizer model configuration and merges to JSON."""
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path | str) -> BPETokenizer:
        """Load trained BPE tokenizer from JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"BPE tokenizer artifact not found: {p}")
        return cls(Tokenizer.from_file(str(p)))


def train_bpe_tokenizer(
    split_manifest_path: Path = Path("data/splits/split_manifest.json"),
    corpus_lock_path: Path = Path("data/corpus_lock.json"),
    normalized_dir: Path = Path("data/normalized"),
    output_dir: Path = Path("artifacts/tokenizers"),
    vocab_size: int = 4096,
    config_dict: dict[str, Any] | None = None,
) -> tuple[BPETokenizer, TokenizerMetadata]:
    """Train a byte-level BPE tokenizer from scratch strictly on the TRAIN split."""
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    if not corpus_lock_path.is_file():
        raise FileNotFoundError(f"Corpus lock not found: {corpus_lock_path}")

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_manifest = SplitManifest.model_validate(json.load(f))
    with open(corpus_lock_path, "r", encoding="utf-8") as f:
        corpus_lock = CorpusLock.model_validate(json.load(f))

    # 1. Rigorous provenance & file hash verification
    verify_corpus_and_split_integrity(split_manifest, corpus_lock, normalized_dir)

    # 2. Collect training file paths in deterministic sorted order
    sorted_train_ids = sorted(split_manifest.train)
    prov_map = {doc.document_id: doc for doc in corpus_lock.documents}
    train_files: list[str] = [
        str(normalized_dir / prov_map[doc_id].family / f"{doc_id}.txt")
        for doc_id in sorted_train_ids
    ]

    # 3. Build empty tokenizer from scratch (no pretrained weights, models, or vocab)
    raw_tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS[UNK_ID]))
    raw_tokenizer.normalizer = NFC()
    raw_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = ByteLevelDecoder()

    # 4. Train BPE on train split files only
    trainer = BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    raw_tokenizer.train(train_files, trainer)

    # 5. Assert deterministic special token IDs
    assert raw_tokenizer.token_to_id(SPECIAL_TOKENS[PAD_ID]) == PAD_ID, "PAD token ID mismatch"
    assert raw_tokenizer.token_to_id(SPECIAL_TOKENS[BOS_ID]) == BOS_ID, "BOS token ID mismatch"
    assert raw_tokenizer.token_to_id(SPECIAL_TOKENS[EOS_ID]) == EOS_ID, "EOS token ID mismatch"
    assert raw_tokenizer.token_to_id(SPECIAL_TOKENS[UNK_ID]) == UNK_ID, "UNK token ID mismatch"

    # 6. Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    bpe_path = output_dir / "bpe.json"
    metadata_path = output_dir / "bpe_metadata.json"

    raw_tokenizer.save(str(bpe_path))
    bpe_sha = compute_file_sha256(bpe_path)
    split_hash = compute_manifest_sha256(split_manifest_path)

    metadata = TokenizerMetadata(
        tokenizer_type="bpe",
        vocab_size=raw_tokenizer.get_vocab_size(),
        corpus_fingerprint=corpus_lock.corpus_fingerprint,
        normalization_fingerprint=corpus_lock.normalization_fingerprint,
        split_manifest_hash=split_hash,
        training_document_ids=sorted_train_ids,
        tokenizer_artifact_sha256=bpe_sha,
        tokenizers_library_version=tokenizers.__version__,
        config=config_dict or {},
    )
    metadata_path.write_text(
        json.dumps(metadata.model_dump(), indent=2), encoding="utf-8", newline="\n"
    )

    return BPETokenizer(raw_tokenizer), metadata
