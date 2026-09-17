"""Factual baseline reports with explicit status and benchmark provenance."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.evaluation.compare import compare_runs
from scripture_lm.evaluation.generation_suite import (
    get_canonical_generation_suite,
    load_generation_results,
)
from scripture_lm.evaluation.runner import evaluate_run
from scripture_lm.experiments.matrix import baseline_matrix, config_hash, scientific_config
from scripture_lm.experiments.storage import atomic_json, read_spec, read_status


def benchmark_metrics(run: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    """Only expose metrics tied to the complete, current standard_v1 benchmark."""
    unavailable: dict[str, Any] = {
        "generation_status": "unavailable",
        "longest_match_chars": None,
        "repetition_distinct_4": None,
    }
    root = run / "generations"
    metadata = root / "benchmark_provenance.json"
    if not metadata.is_file():
        return unavailable
    try:
        provenance = json.loads(metadata.read_text(encoding="utf-8"))
        suite = get_canonical_generation_suite()
        if provenance["suite"] != suite.model_dump(mode="json"):
            raise ValueError("Benchmark suite differs from standard_v1")
        for key in (
            "checkpoint_model_sha256",
            "tokenizer_artifact_sha256",
            "corpus_fingerprint",
            "split_manifest_hash",
        ):
            if not evaluation.get(key) or provenance.get(key) != evaluation[key]:
                raise ValueError(f"Benchmark {key} differs from evaluation")
        for name in ("samples.json", "repetition_report.json", "memorization_report.json"):
            if compute_file_sha256(root / name) != provenance["files"][name]:
                raise ValueError(f"Benchmark artifact hash mismatch: {name}")
        samples = load_generation_results(root / "samples.json")
        expected = Counter((p.prompt_text, seed) for p in suite.prompts for seed in suite.seeds)
        if Counter((s.prompt, s.settings.seed) for s in samples) != expected:
            raise ValueError("Incomplete or duplicate benchmark prompt/seed pairs")
        for sample in samples:
            settings = suite.canonical_settings.model_copy(update={"seed": sample.settings.seed})
            if sample.suite_id != suite.suite_id or sample.settings != settings:
                raise ValueError("Noncanonical benchmark settings")
        memo = json.loads((root / "memorization_report.json").read_text(encoding="utf-8"))
        repetition = json.loads((root / "repetition_report.json").read_text(encoding="utf-8"))
        return {
            "generation_status": "compatible",
            "longest_match_chars": memo["max_longest_match_chars"],
            "repetition_distinct_4": repetition["mean_distinct_4"],
        }
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return {**unavailable, "generation_status": "incompatible", "generation_reason": str(exc)}


def compare_baseline(
    *,
    runs_root: Path = Path("runs"),
    output_dir: Path | None = None,
    split: str = "validation",
    device: str | None = None,
    generate: bool = False,
) -> dict[str, Any]:
    """Evaluate completed runs and retain every baseline slot in fixed matrix order."""
    if split not in {"validation", "test"}:
        raise ValueError("Comparison split must be validation or test")
    rows: list[dict[str, Any]] = []
    completed: list[Path] = []
    evaluations: dict[str, dict[str, Any]] = {}
    for experiment in baseline_matrix():
        run = runs_root / experiment.name
        status = read_status(run)
        config = experiment.resolve()
        row: dict[str, Any] = {
            "name": experiment.name,
            "status": status["status"],
            "tokenizer": "char" if config.tokenizer.type == "character" else "bpe",
            "sampling_mode": config.data.sampling_mode,
            "sampling_alpha": config.data.sampling_alpha
            if config.data.sampling_mode == "temperature"
            else None,
            "macro_bpc": None,
            "micro_bpc": None,
            "family_bpc": None,
            "token_cross_entropy": None,
            "token_perplexity": None,
            "training_chars_exposed": None,
            "training_tokens_exposed": None,
            "parameter_count": None,
            "generation_status": "unavailable",
            "longest_match_chars": None,
            "repetition_distinct_4": None,
        }
        rows.append(row)
        if status["status"] == "missing":
            continue
        spec = read_spec(run)
        if spec["configuration"] != scientific_config(config):
            raise ValueError(f"Run {experiment.name} does not match its canonical baseline")
        row["experiment_config_hash"] = config_hash(spec)
        if status["status"] != "completed":
            row["error"] = status.get("error")
            continue
        # Always evaluate current best weights; stale reports are never silently reused.
        evaluate_run(run, split=split, checkpoint="best", device=device, generate=generate)
        evaluation = json.loads((run / "evaluation" / f"{split}_metrics.json").read_text("utf-8"))
        if evaluation.get("split") != split:
            raise ValueError(f"Wrong evaluation split for {run}")
        expected_checkpoint = compute_file_sha256(
            run / "checkpoints" / "best" / "model.safetensors"
        )
        if evaluation.get("checkpoint_model_sha256") != expected_checkpoint:
            raise ValueError(f"Stale evaluation checkpoint for {run}")
        for key in (
            "corpus_fingerprint",
            "normalization_fingerprint",
            "split_manifest_hash",
            "tokenizer_artifact_sha256",
        ):
            if not evaluation.get(key) or evaluation[key] != spec["provenance"].get(key):
                raise ValueError(f"Evaluation {key} does not match experiment specification: {run}")
        for key in ("macro_bpc", "micro_bpc", "token_cross_entropy", "token_perplexity"):
            if not math.isfinite(evaluation[key]):
                raise ValueError(f"Nonfinite evaluation metric {key}: {run}")
        evaluations[experiment.name] = evaluation
        completed.append(run)

    if completed:
        # Preserve Prompt 07's compatibility checks; no allow-incompatible escape hatch.
        report = compare_runs(completed, split=split)
        by_name = {summary.run_name: summary for summary in report.runs}
        for row in rows:
            if row["status"] != "completed":
                continue
            summary = by_name[row["name"]]
            evaluation = evaluations[row["name"]]
            for key in (
                "macro_bpc",
                "micro_bpc",
                "family_bpc",
                "token_cross_entropy",
                "token_perplexity",
            ):
                row[key] = evaluation[key]
            row.update(
                training_chars_exposed=summary.training_chars_exposed,
                training_tokens_exposed=summary.training_tokens_exposed,
                parameter_count=summary.parameter_count,
            )
            row.update(benchmark_metrics(runs_root / row["name"], evaluation))

    payload = {"schema_version": "baseline_comparison_v1", "split": split, "runs": rows}
    destination = output_dir if output_dir is not None else runs_root / "baseline_comparison"
    atomic_json(destination / "comparison.json", payload)
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "family_bpc"}
        for family in ("hebrew_bible", "new_testament", "quran"):
            flat[f"{family}_bpc"] = (row["family_bpc"] or {}).get(family)
        flat_rows.append(flat)
    columns = list(dict.fromkeys(key for row in flat_rows for key in row))
    with (destination / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)
    lines = [
        "# Baseline comparison",
        "",
        f"Evaluation split: **{split}**. Checkpoint: **best**.",
        "",
        "Runs appear in canonical matrix order. BPC is the cross-tokenizer metric; "
        "token perplexity uses different units across tokenizers.",
        "",
        "The canonical budget is max_effective_epochs = 20 with early stopping enabled "
        "and patience = 8. Early stopping is a valid completion.",
        "",
        "| Experiment | Status | Macro BPC | Micro BPC | HB BPC | NT BPC | Quran BPC | "
        "Token loss | Token PPL | Chars exposed | Benchmark | Longest match | Distinct-4 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in flat_rows:
        keys = (
            "name",
            "status",
            "macro_bpc",
            "micro_bpc",
            "hebrew_bible_bpc",
            "new_testament_bpc",
            "quran_bpc",
            "token_cross_entropy",
            "token_perplexity",
            "training_chars_exposed",
            "generation_status",
            "longest_match_chars",
            "repetition_distinct_4",
        )
        values = ["—" if row.get(key) is None else str(row[key]) for key in keys]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Unavailable metrics are not zero. Generation metrics require verified "
            "standard_v1 prompts, seeds 0–9, T=0.8, top-p=0.95, a 1024-character target, "
            "and the same checkpoint and corpus as evaluation.",
            "",
        ]
    )
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
