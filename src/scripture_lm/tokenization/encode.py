"""Tokenization statistics calculation and cross-tokenizer comparison report."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scripture_lm.corpus.normalize import CorpusLock
from scripture_lm.corpus.split import SplitManifest
from scripture_lm.tokenization.base import (
    BaseTokenizer,
    TokenizerMetadata,
    verify_corpus_and_split_integrity,
)


def compute_tokenizer_split_stats(
    tokenizer: BaseTokenizer,
    metadata: TokenizerMetadata,
    split_manifest: SplitManifest,
    corpus_lock: CorpusLock,
    normalized_dir: Path = Path("data/normalized"),
) -> dict[str, Any]:
    """Compute detailed tokenization statistics across train, validation, and test splits."""
    prov_map = {doc.document_id: doc for doc in corpus_lock.documents}

    def load_split_text(doc_ids: list[str]) -> list[str]:
        texts: list[str] = []
        for d in sorted(doc_ids):
            fpath = normalized_dir / prov_map[d].family / f"{d}.txt"
            texts.append(fpath.read_text(encoding="utf-8"))
        return texts

    train_texts = load_split_text(split_manifest.train)
    val_texts = load_split_text(split_manifest.validation)
    test_texts = load_split_text(split_manifest.test)

    # 1. Training metrics
    total_train_chars = sum(len(t) for t in train_texts)
    total_train_bytes = sum(len(t.encode("utf-8")) for t in train_texts)

    train_token_counter: Counter[int] = Counter()
    total_train_tokens = 0

    for t in train_texts:
        ids = tokenizer.encode(t)
        total_train_tokens += len(ids)
        train_token_counter.update(ids)

    chars_per_token = (total_train_chars / total_train_tokens) if total_train_tokens else 0.0
    bytes_per_token = (total_train_bytes / total_train_tokens) if total_train_tokens else 0.0

    # Top 20 non-special tokens
    special_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id}
    non_special_counts = [
        (tid, count) for tid, count in train_token_counter.items() if tid not in special_ids
    ]
    non_special_counts.sort(key=lambda x: x[1], reverse=True)

    top_tokens: list[dict[str, Any]] = []
    for tid, count in non_special_counts[:20]:
        token_str = tokenizer.id_to_token(tid) or ""
        top_tokens.append({"token": token_str, "id": tid, "count": count})

    # 2. Unknown character analysis in validation and test
    def analyze_unknowns(texts: list[str]) -> dict[str, Any]:
        unknown_occurrences = 0
        unknown_chars: set[str] = set()
        for text in texts:
            if tokenizer.tokenizer_type == "character":
                for c in text:
                    if tokenizer.token_to_id(c) == tokenizer.unk_id:
                        unknown_occurrences += 1
                        unknown_chars.add(c)
            elif tokenizer.tokenizer_type == "bpe":
                # Check if BPE produces UNK token IDs
                ids = tokenizer.encode(text)
                unk_count = sum(1 for i in ids if i == tokenizer.unk_id)
                unknown_occurrences += unk_count

        return {
            "unknown_occurrences": unknown_occurrences,
            "unique_unknown_characters": len(unknown_chars),
            "characters": sorted(unknown_chars),
        }

    val_unknowns = analyze_unknowns(val_texts)
    test_unknowns = analyze_unknowns(test_texts)

    # 3. Learned merges for BPE
    learned_merges = None
    if tokenizer.tokenizer_type == "bpe":
        # Initial 256 bytes + 4 special tokens = 260 base symbols
        learned_merges = max(0, tokenizer.vocab_size - 260)

    return {
        "tokenizer_type": tokenizer.tokenizer_type,
        "tokenizer_artifact_sha256": metadata.tokenizer_artifact_sha256,
        "corpus_fingerprint": metadata.corpus_fingerprint,
        "normalization_fingerprint": metadata.normalization_fingerprint,
        "split_manifest_hash": metadata.split_manifest_hash,
        "vocab_size": tokenizer.vocab_size,
        "training_characters": total_train_chars,
        "training_bytes": total_train_bytes,
        "training_tokens": total_train_tokens,
        "characters_per_token": chars_per_token,
        "bytes_per_token": bytes_per_token,
        "learned_merge_count": learned_merges,
        "top_tokens": top_tokens,
        "unknown_characters": {
            "validation": val_unknowns,
            "test": test_unknowns,
        },
    }


def compute_and_update_tokenizer_stats(
    tokenizer: BaseTokenizer,
    metadata: TokenizerMetadata,
    split_manifest_path: Path = Path("data/splits/split_manifest.json"),
    corpus_lock_path: Path = Path("data/corpus_lock.json"),
    normalized_dir: Path = Path("data/normalized"),
    stats_path: Path = Path("artifacts/tokenizers/tokenizer_stats.json"),
) -> dict[str, Any]:
    """Compute stats for active tokenizer and update artifacts/tokenizers/tokenizer_stats.json."""
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_manifest = SplitManifest.model_validate(json.load(f))
    with open(corpus_lock_path, "r", encoding="utf-8") as f:
        corpus_lock = CorpusLock.model_validate(json.load(f))

    verify_corpus_and_split_integrity(split_manifest, corpus_lock, normalized_dir)

    new_stats = compute_tokenizer_split_stats(
        tokenizer=tokenizer,
        metadata=metadata,
        split_manifest=split_manifest,
        corpus_lock=corpus_lock,
        normalized_dir=normalized_dir,
    )

    # Load existing stats file if present
    existing_stats: dict[str, Any] = {"bpe": None, "character": None, "comparison": None}
    if stats_path.is_file():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_stats.update(loaded)
        except Exception:
            pass

    # Overwrite the active tokenizer's stats
    existing_stats[tokenizer.tokenizer_type] = new_stats

    # Check if cross-tokenizer comparison can be populated safely
    bpe_stats = existing_stats.get("bpe")
    char_stats = existing_stats.get("character")

    if bpe_stats and char_stats:
        bpe_corpus_fp = bpe_stats.get("corpus_fingerprint")
        char_corpus_fp = char_stats.get("corpus_fingerprint")
        bpe_norm_fp = bpe_stats.get("normalization_fingerprint")
        char_norm_fp = char_stats.get("normalization_fingerprint")
        bpe_split_hash = bpe_stats.get("split_manifest_hash")
        char_split_hash = char_stats.get("split_manifest_hash")

        if (
            bpe_corpus_fp == char_corpus_fp
            and bpe_norm_fp == char_norm_fp
            and bpe_split_hash == char_split_hash
        ):
            bpe_tokens = bpe_stats["training_tokens"]
            char_tokens = char_stats["training_tokens"]
            existing_stats["comparison"] = {
                "corpus_fingerprint": bpe_corpus_fp,
                "bpe_vocab_size": bpe_stats["vocab_size"],
                "char_vocab_size": char_stats["vocab_size"],
                "bpe_characters_per_token": bpe_stats["characters_per_token"],
                "char_characters_per_token": char_stats["characters_per_token"],
                "bpe_training_tokens": bpe_tokens,
                "char_training_tokens": char_tokens,
                "compression_ratio_char_over_bpe": (
                    (char_tokens / bpe_tokens) if bpe_tokens else 0.0
                ),
                "status": "Compatible: both tokenizers trained on identical corpus split.",
            }
        else:
            existing_stats["comparison"] = None
            existing_stats["comparison_status"] = (
                "Comparison unavailable: BPE and Character tokenizer artifacts were "
                "built from different corpus/split versions."
            )
    else:
        existing_stats["comparison"] = None

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(existing_stats, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    return existing_stats


def render_tokenizer_stats(stats_data: dict[str, Any]) -> None:
    """Render Rich summary tables for tokenizer statistics and cross-tokenizer comparison."""
    console = Console()

    # Per-tokenizer table
    table = Table(title="Tokenizer Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold yellow", width=28)
    table.add_column("BPE", style="white", justify="right")
    table.add_column("Character", style="white", justify="right")

    bpe = stats_data.get("bpe") or {}
    char = stats_data.get("character") or {}

    def fmt_val(d: dict[str, Any], key: str, is_float: bool = False) -> str:
        if key not in d or d[key] is None:
            return "N/A"
        return f"{d[key]:.2f}" if is_float else f"{d[key]:,}"

    table.add_row("Vocabulary Size", fmt_val(bpe, "vocab_size"), fmt_val(char, "vocab_size"))
    table.add_row(
        "Training Tokens",
        fmt_val(bpe, "training_tokens"),
        fmt_val(char, "training_tokens"),
    )
    table.add_row(
        "Characters / Token",
        fmt_val(bpe, "characters_per_token", True),
        fmt_val(char, "characters_per_token", True),
    )
    table.add_row(
        "Bytes / Token",
        fmt_val(bpe, "bytes_per_token", True),
        fmt_val(char, "bytes_per_token", True),
    )

    learned_merges_str = fmt_val(bpe, "learned_merge_count") if bpe else "N/A"
    table.add_row("Learned BPE Merges", learned_merges_str, "N/A")

    def get_unk_occ(d: dict[str, Any] | None, split: str) -> str:
        if not d:
            return "N/A"
        return str(d.get("unknown_characters", {}).get(split, {}).get("unknown_occurrences", "N/A"))

    val_unk_bpe = get_unk_occ(bpe, "validation")
    val_unk_char = get_unk_occ(char, "validation")
    table.add_row("Val Unknown Tokens", val_unk_bpe, val_unk_char)

    test_unk_bpe = get_unk_occ(bpe, "test")
    test_unk_char = get_unk_occ(char, "test")
    table.add_row("Test Unknown Tokens", test_unk_bpe, test_unk_char)

    console.print(table)

    # Comparison panel
    comp = stats_data.get("comparison")
    if comp:
        comp_panel = Panel(
            f"[bold green]Compression Ratio (CHAR tokens / BPE tokens):[/] "
            f"[bold white]{comp['compression_ratio_char_over_bpe']:.2f}x[/]\n"
            f"[green]Status:[/] {comp['status']}",
            title="Cross-Tokenizer Comparison",
            border_style="green",
        )
        console.print(comp_panel)
    elif stats_data.get("comparison_status"):
        comp_panel = Panel(
            f"[bold yellow]{stats_data['comparison_status']}[/]",
            title="Cross-Tokenizer Comparison",
            border_style="yellow",
        )
        console.print(comp_panel)
