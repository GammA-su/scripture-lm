"""Corpus statistics computation and Rich table rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scripture_lm.corpus.split import SplitManifest


def compute_corpus_stats(
    split_manifest: SplitManifest, normalized_dir: Path = Path("data/normalized")
) -> dict[str, Any]:
    """Compute document, character, byte, and word counts per split and scripture family."""
    # Build a lookup of doc_id -> (family, file_path)
    # Scan normalized_dir
    doc_paths: dict[str, tuple[str, Path]] = {}
    if normalized_dir.is_dir():
        for fpath in normalized_dir.rglob("*.txt"):
            if fpath.is_file():
                doc_id = fpath.stem
                family = fpath.parent.name
                doc_paths[doc_id] = (family, fpath)

    # Read stats for all documents
    doc_stats: dict[str, dict[str, int]] = {}
    total_chars = 0
    total_bytes = 0
    total_words = 0

    for doc_id, (family, path) in doc_paths.items():
        text = path.read_text(encoding="utf-8")
        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))
        n_words = len(text.split())
        doc_stats[doc_id] = {
            "characters": n_chars,
            "bytes": n_bytes,
            "words": n_words,
        }
        total_chars += n_chars
        total_bytes += n_bytes
        total_words += n_words

    splits_map = {
        "train": split_manifest.train,
        "validation": split_manifest.validation,
        "test": split_manifest.test,
    }

    results: dict[str, Any] = {
        "total": {
            "documents": len(doc_stats),
            "characters": total_chars,
            "bytes": total_bytes,
            "words": total_words,
        },
        "splits": {},
        "families": {},
    }

    for split_name, doc_ids in splits_map.items():
        s_docs = len(doc_ids)
        s_chars = sum(doc_stats.get(d, {}).get("characters", 0) for d in doc_ids)
        s_bytes = sum(doc_stats.get(d, {}).get("bytes", 0) for d in doc_ids)
        s_words = sum(doc_stats.get(d, {}).get("words", 0) for d in doc_ids)
        char_pct = (s_chars / total_chars * 100.0) if total_chars else 0.0

        results["splits"][split_name] = {
            "documents": s_docs,
            "characters": s_chars,
            "bytes": s_bytes,
            "words": s_words,
            "character_pct": char_pct,
            "families": {},
        }

        # Breakdown by family inside split
        fam_dict = split_manifest.family_breakdown
        for family, fam_splits in fam_dict.items():
            f_ids = fam_splits.get(split_name, [])
            f_docs = len(f_ids)
            f_chars = sum(doc_stats.get(d, {}).get("characters", 0) for d in f_ids)
            f_bytes = sum(doc_stats.get(d, {}).get("bytes", 0) for d in f_ids)
            f_words = sum(doc_stats.get(d, {}).get("words", 0) for d in f_ids)
            f_pct = (f_chars / total_chars * 100.0) if total_chars else 0.0

            results["splits"][split_name]["families"][family] = {
                "documents": f_docs,
                "characters": f_chars,
                "bytes": f_bytes,
                "words": f_words,
                "character_pct": f_pct,
            }

    # Aggregate by family across all splits
    for family, fam_splits in split_manifest.family_breakdown.items():
        all_fam_ids: list[str] = []
        for ids in fam_splits.values():
            all_fam_ids.extend(ids)
        f_docs = len(all_fam_ids)
        f_chars = sum(doc_stats.get(d, {}).get("characters", 0) for d in all_fam_ids)
        f_bytes = sum(doc_stats.get(d, {}).get("bytes", 0) for d in all_fam_ids)
        f_words = sum(doc_stats.get(d, {}).get("words", 0) for d in all_fam_ids)
        f_pct = (f_chars / total_chars * 100.0) if total_chars else 0.0

        results["families"][family] = {
            "documents": f_docs,
            "characters": f_chars,
            "bytes": f_bytes,
            "words": f_words,
            "character_pct": f_pct,
        }

    return results


def render_stats_table(stats: dict[str, Any]) -> None:
    """Render comprehensive corpus statistics tables using Rich."""
    console = Console()

    # Split Level Table
    split_table = Table(
        title="Corpus Statistics by Dataset Split", show_header=True, header_style="bold cyan"
    )
    split_table.add_column("Split", style="bold yellow")
    split_table.add_column("Documents", justify="right")
    split_table.add_column("Characters", justify="right")
    split_table.add_column("Bytes", justify="right")
    split_table.add_column("Words (approx)", justify="right")
    split_table.add_column("Corpus %", justify="right")

    for split_name in ["train", "validation", "test"]:
        s = stats["splits"].get(split_name, {})
        split_table.add_row(
            split_name,
            f"{s.get('documents', 0):,}",
            f"{s.get('characters', 0):,}",
            f"{s.get('bytes', 0):,}",
            f"{s.get('words', 0):,}",
            f"{s.get('character_pct', 0.0):.2f}%",
        )

    tot = stats["total"]
    split_table.add_section()
    split_table.add_row(
        "TOTAL",
        f"{tot['documents']:,}",
        f"{tot['characters']:,}",
        f"{tot['bytes']:,}",
        f"{tot['words']:,}",
        "100.00%",
        style="bold white",
    )
    console.print(split_table)

    # Family Breakdown Table
    if stats.get("families"):
        fam_table = Table(
            title="Corpus Statistics by Scripture Family",
            show_header=True,
            header_style="bold green",
        )
        fam_table.add_column("Family", style="bold white")
        fam_table.add_column("Documents", justify="right")
        fam_table.add_column("Characters", justify="right")
        fam_table.add_column("Bytes", justify="right")
        fam_table.add_column("Words (approx)", justify="right")
        fam_table.add_column("Corpus %", justify="right")

        for family, f in sorted(stats["families"].items()):
            fam_table.add_row(
                family,
                f"{f['documents']:,}",
                f"{f['characters']:,}",
                f"{f['bytes']:,}",
                f"{f['words']:,}",
                f"{f['character_pct']:.2f}%",
            )
        console.print(fam_table)

    if tot["documents"] == 0:
        console.print(
            Panel(
                "[bold yellow]Corpus contains 0 normalized documents. "
                "Populate corpus/raw and run 'scripture-lm corpus prepare'.[/]",
                title="Notice",
                border_style="yellow",
            )
        )
