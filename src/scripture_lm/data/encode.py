"""Dataset encoding pipeline producing memory-mappable binary token streams and chunk indices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.corpus.normalize import CorpusLock
from scripture_lm.corpus.split import SplitManifest
from scripture_lm.data.chunk_index import (
    ENCODING_PROVENANCE_FILENAME,
    ChunkMetadata,
    EncodingProvenance,
    build_chunks_from_stream,
    save_chunk_index,
)
from scripture_lm.tokenization.base import (
    BOS_ID,
    EOS_ID,
    BaseTokenizer,
    compute_manifest_sha256,
    verify_corpus_and_split_integrity,
)
from scripture_lm.tokenization.bpe import BPETokenizer
from scripture_lm.tokenization.character import CharacterTokenizer


def compute_token_character_credits(
    text: str,
    tokenizer: BaseTokenizer,
) -> tuple[list[int], list[int]]:
    """Compute token IDs and exact cumulative character ownership credits for document text.

    Args:
        text: Normalized UTF-8 document string.
        tokenizer: BPETokenizer or CharacterTokenizer instance.

    Returns:
        tuple of (content_token_ids, character_credits) excluding BOS/EOS.
    """
    if isinstance(tokenizer, BPETokenizer):
        encoding = tokenizer._tokenizer.encode(text)
        token_ids: list[int] = encoding.ids
        offsets: list[tuple[int, int]] = encoding.offsets

        # Cumulative character ownership
        covered_end = 0
        char_credits: list[int] = []
        for _start, end in offsets:
            credit = max(0, end - covered_end)
            char_credits.append(credit)
            covered_end = max(covered_end, end)

        if sum(char_credits) != len(text):
            raise ValueError(
                f"BPE character credit mismatch: sum={sum(char_credits)}, len(text)={len(text)}"
            )
        return token_ids, char_credits

    elif isinstance(tokenizer, CharacterTokenizer):
        token_ids = tokenizer.encode(text)
        char_credits = [1] * len(token_ids)
        if sum(char_credits) != len(text):
            raise ValueError(
                f"Character credit mismatch: sum={sum(char_credits)}, len(text)={len(text)}"
            )
        return token_ids, char_credits

    else:
        # Fallback for any other BaseTokenizer
        token_ids = tokenizer.encode(text)
        char_credits = [1] * len(token_ids)
        return token_ids, char_credits


def encode_dataset(
    tokenizer: BaseTokenizer,
    split_manifest_path: Path | str = Path("data/splits/split_manifest.json"),
    corpus_lock_path: Path | str = Path("data/corpus_lock.json"),
    normalized_dir: Path | str = Path("data/normalized"),
    output_base_dir: Path | str = Path("data/encoded"),
    context_length: int | None = None,
) -> tuple[dict[str, list[ChunkMetadata]], EncodingProvenance]:
    """Encode the normalized corpus into memory-mappable binary streams and build chunk indices.

    Args:
        tokenizer: Trained BPETokenizer or CharacterTokenizer.
        split_manifest_path: Path to split_manifest.json.
        corpus_lock_path: Path to corpus_lock.json.
        normalized_dir: Path to directory containing normalized files.
        output_base_dir: Root directory to save encoded binary files and metadata.
        context_length: Context length L (defaults to 512 for BPE, 2048 for Character).

    Returns:
        tuple of (dict of split -> chunk list, EncodingProvenance metadata).
    """
    split_path = Path(split_manifest_path)
    lock_path = Path(corpus_lock_path)
    norm_path = Path(normalized_dir)
    out_base = Path(output_base_dir)

    split_manifest = SplitManifest.model_validate_json(split_path.read_text(encoding="utf-8"))
    corpus_lock = CorpusLock.model_validate_json(lock_path.read_text(encoding="utf-8"))

    # 1. Verify corpus and split integrity
    verify_corpus_and_split_integrity(split_manifest, corpus_lock, norm_path)

    # 2. Check storage safety bounds
    if tokenizer.vocab_size > 65536:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) exceeds uint16 bound (65536)."
        )

    tok_type: Literal["bpe", "character"] = (
        "bpe" if tokenizer.tokenizer_type == "bpe" else "character"
    )
    if context_length is None:
        ctx_len = 512 if tok_type == "bpe" else 2048
    else:
        ctx_len = context_length

    encoded_dir = out_base / tok_type
    encoded_dir.mkdir(parents=True, exist_ok=True)

    prov_map = {doc.document_id: doc for doc in corpus_lock.documents}
    splits_map: dict[Literal["train", "validation", "test"], list[str]] = {
        "train": split_manifest.train,
        "validation": split_manifest.validation,
        "test": split_manifest.test,
    }

    all_split_chunks: dict[str, list[ChunkMetadata]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    file_provenance: dict[str, dict[str, Any]] = {}
    chunk_counts: dict[str, int] = {}

    # 3. Process each split and family
    for split_name, doc_ids in splits_map.items():
        split_chunk_list: list[ChunkMetadata] = []
        chunk_idx = 0

        # Group document IDs by family
        family_docs: dict[str, list[str]] = {}
        for doc_id in doc_ids:
            fam = prov_map[doc_id].family
            family_docs.setdefault(fam, []).append(doc_id)

        # Sort families for determinism
        for family in sorted(family_docs.keys()):
            docs_in_family = sorted(family_docs[family])
            stream_tokens: list[int] = []
            stream_credits: list[int] = []
            doc_spans: list[tuple[str, int, int]] = []

            for doc_id in docs_in_family:
                doc_file = norm_path / family / f"{doc_id}.txt"
                text = doc_file.read_text(encoding="utf-8")

                content_tokens, content_credits = compute_token_character_credits(text, tokenizer)

                # Each complete document is framed by <bos> and <eos>
                # <bos> and <eos> receive 0 character credit
                doc_tokens = [BOS_ID] + content_tokens + [EOS_ID]
                doc_credits = [0] + content_credits + [0]

                start_idx = len(stream_tokens)
                stream_tokens.extend(doc_tokens)
                stream_credits.extend(doc_credits)
                end_idx = len(stream_tokens)
                doc_spans.append((doc_id, start_idx, end_idx))

            if not stream_tokens:
                continue

            # Safety check on token IDs
            max_token_id = max(stream_tokens)
            if max_token_id > 65535:
                raise ValueError(f"Token ID {max_token_id} exceeds uint16 storage maximum (65535).")

            # Write binary stream as uint16
            split_bin_dir = encoded_dir / split_name
            split_bin_dir.mkdir(parents=True, exist_ok=True)
            bin_file = split_bin_dir / f"{family}.bin"

            arr = np.array(stream_tokens, dtype=np.uint16)
            arr.tofile(bin_file)

            rel_bin_path = str(bin_file.relative_to(out_base)).replace("\\", "/")
            credits_arr = np.array(stream_credits, dtype=np.int64)

            # Build fixed-context chunks with stride L
            family_chunks, chunk_idx = build_chunks_from_stream(
                token_count=len(stream_tokens),
                char_credits=credits_arr,
                doc_spans=doc_spans,
                context_length=ctx_len,
                split=split_name,
                family=family,
                tokenizer_type=tok_type,
                bin_path=rel_bin_path,
                start_chunk_idx=chunk_idx,
            )
            split_chunk_list.extend(family_chunks)

            file_provenance[f"{split_name}/{family}"] = {
                "path": rel_bin_path,
                "token_count": len(stream_tokens),
                "byte_size": bin_file.stat().st_size,
                "sha256": compute_file_sha256(bin_file),
                "chunk_count": len(family_chunks),
            }

        all_split_chunks[split_name] = split_chunk_list
        chunk_counts[split_name] = len(split_chunk_list)

        # Save chunk index for this split
        index_file = encoded_dir / f"{split_name}_chunks.json"
        save_chunk_index(split_chunk_list, index_file)

    # 4. Strict Exposure Accounting Verification on TRAIN Split
    train_chunks = all_split_chunks["train"]
    n_chunks = sum(c.raw_character_count for c in train_chunks)
    n_manifest = sum(prov_map[doc_id].normalized_characters for doc_id in split_manifest.train)

    if n_chunks != n_manifest:
        raise ValueError(
            f"Exposure accounting mismatch: natural training chunks account for {n_chunks} "
            f"target characters, but training corpus has {n_manifest} normalized characters."
        )

    # 5. Build and save encoding provenance
    tok_artifact_path = Path("artifacts/tokenizers") / (
        "bpe.json" if tok_type == "bpe" else "char_vocab.json"
    )
    tok_sha256 = (
        compute_file_sha256(tok_artifact_path) if tok_artifact_path.is_file() else "unrecorded"
    )

    provenance = EncodingProvenance(
        tokenizer_type=tok_type,
        context_length=ctx_len,
        chunk_length=ctx_len + 1,
        corpus_fingerprint=corpus_lock.corpus_fingerprint,
        normalization_fingerprint=corpus_lock.normalization_fingerprint,
        split_manifest_hash=compute_manifest_sha256(split_path),
        tokenizer_artifact_sha256=tok_sha256,
        total_chunks=chunk_counts,
        natural_train_target_characters=n_chunks,
        files=file_provenance,
    )

    meta_file = encoded_dir / ENCODING_PROVENANCE_FILENAME
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write(provenance.model_dump_json(indent=2))

    return all_split_chunks, provenance
