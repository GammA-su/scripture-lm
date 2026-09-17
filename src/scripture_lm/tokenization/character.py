"""Character-level tokenizer constructing vocabulary from training Unicode codepoints."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

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


class CharacterTokenizer(BaseTokenizer):
    """Character tokenizer mapping Unicode codepoints to integer IDs."""

    def __init__(self, vocab: list[str]) -> None:
        self._vocab = list(vocab)
        self._token_to_id = {tok: idx for idx, tok in enumerate(self._vocab)}
        self._id_to_token = {idx: tok for idx, tok in enumerate(self._vocab)}

        # Assert special token ordering
        assert self.token_to_id(SPECIAL_TOKENS[PAD_ID]) == PAD_ID, "PAD ID mismatch"
        assert self.token_to_id(SPECIAL_TOKENS[BOS_ID]) == BOS_ID, "BOS ID mismatch"
        assert self.token_to_id(SPECIAL_TOKENS[EOS_ID]) == EOS_ID, "EOS ID mismatch"
        assert self.token_to_id(SPECIAL_TOKENS[UNK_ID]) == UNK_ID, "UNK ID mismatch"

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return len(self._vocab)

    @property
    def tokenizer_type(self) -> str:
        """Type identifier."""
        return "character"

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text into character token IDs with NFC canonicalization."""
        normalized = unicodedata.normalize("NFC", text)
        ids = [self._token_to_id.get(c, self.unk_id) for c in normalized]

        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to a text string."""
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        chars: list[str] = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            chars.append(self._id_to_token.get(i, SPECIAL_TOKENS[self.unk_id]))
        return "".join(chars)

    def id_to_token(self, token_id: int) -> str | None:
        """Convert integer ID to token string."""
        return self._id_to_token.get(token_id)

    def token_to_id(self, token: str) -> int | None:
        """Convert token string to integer ID."""
        return self._token_to_id.get(token)

    def save(self, path: Path | str) -> None:
        """Save character vocabulary to JSON file."""
        data = {
            "version": "1.0",
            "type": "character",
            "vocab_size": self.vocab_size,
            "vocab": self._vocab,
        }
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, path: Path | str) -> CharacterTokenizer:
        """Load character tokenizer from JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Character vocabulary artifact not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"])


def build_character_tokenizer(
    split_manifest_path: Path = Path("data/splits/split_manifest.json"),
    corpus_lock_path: Path = Path("data/corpus_lock.json"),
    normalized_dir: Path = Path("data/normalized"),
    output_dir: Path = Path("artifacts/tokenizers"),
    config_dict: dict[str, Any] | None = None,
) -> tuple[CharacterTokenizer, TokenizerMetadata]:
    """Construct character vocabulary solely from Unicode codepoints in the TRAIN split."""
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

    # 2. Extract unique characters strictly from sorted training documents
    sorted_train_ids = sorted(split_manifest.train)
    prov_map = {doc.document_id: doc for doc in corpus_lock.documents}

    train_chars: set[str] = set()
    for doc_id in sorted_train_ids:
        fpath = normalized_dir / prov_map[doc_id].family / f"{doc_id}.txt"
        text = fpath.read_text(encoding="utf-8")
        train_chars.update(unicodedata.normalize("NFC", text))

    # Form deterministic vocabulary
    vocab = [
        SPECIAL_TOKENS[PAD_ID],
        SPECIAL_TOKENS[BOS_ID],
        SPECIAL_TOKENS[EOS_ID],
        SPECIAL_TOKENS[UNK_ID],
        *sorted(train_chars),
    ]

    tokenizer = CharacterTokenizer(vocab)

    # 3. Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "char_vocab.json"
    metadata_path = output_dir / "char_metadata.json"

    tokenizer.save(vocab_path)
    vocab_sha = compute_file_sha256(vocab_path)
    split_hash = compute_manifest_sha256(split_manifest_path)

    metadata = TokenizerMetadata(
        tokenizer_type="character",
        vocab_size=tokenizer.vocab_size,
        corpus_fingerprint=corpus_lock.corpus_fingerprint,
        normalization_fingerprint=corpus_lock.normalization_fingerprint,
        split_manifest_hash=split_hash,
        training_document_ids=sorted_train_ids,
        tokenizer_artifact_sha256=vocab_sha,
        tokenizers_library_version="custom-char-v1",
        config=config_dict or {},
    )
    metadata_path.write_text(
        json.dumps(metadata.model_dump(), indent=2), encoding="utf-8", newline="\n"
    )

    return tokenizer, metadata
