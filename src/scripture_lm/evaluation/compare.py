"""Cross-tokenizer experiment comparison and report generation.

Ranks compatible experiment runs strictly by the Macro BPC of the selected evaluation split
(default: validation).

Provenance verification:
    Before comparing runs, verifies that all runs share identical corpus_fingerprint,
    normalization_fingerprint, split_manifest_hash, evaluated character counts, and that
    evaluation reports match the current checkpoint hash.
    If --allow-incompatible is provided, performance ranking is disabled and input order
    is preserved with an explicit warning banner.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from scripture_lm.corpus.manifest import compute_file_sha256


class RunSummary(BaseModel):
    """Summary metrics and metadata for a single experiment run."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    run_path: str
    tokenizer_type: str
    sampling_mode: str
    sampling_alpha: float | None = None
    parameter_count: int = 0
    macro_bpc: float = float("inf")
    micro_bpc: float = float("inf")
    family_bpc: dict[str, float] = Field(default_factory=dict)
    token_perplexity: float = float("inf")
    token_cross_entropy: float = float("inf")
    longest_match_chars: int = 0
    longest_match_source_doc: str | None = None
    repetition_distinct_4: float | None = None
    training_chars_exposed: int = 0
    training_tokens_exposed: int = 0

    # Provenance fields
    corpus_fingerprint: str = ""
    normalization_fingerprint: str = ""
    split_manifest_hash: str = ""
    evaluation_checkpoint_hash: str = ""
    current_checkpoint_hash: str = ""
    evaluated_characters: int = 0
    evaluated_chunks: int = 0
    evaluation_split: str = "validation"


class ComparisonReport(BaseModel):
    """Comparison matrix across multiple runs."""

    model_config = ConfigDict(extra="forbid")

    split: str
    is_compatible: bool
    incompatibility_reasons: list[str] = Field(default_factory=list)
    ranking_metric: str = "macro_bpc"
    runs: list[RunSummary] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render comparison results as a GitHub Flavored Markdown table."""
        lines: list[str] = []
        if not self.is_compatible:
            lines.append("> [!WARNING]")
            lines.append(
                "> **INCOMPATIBLE** — Metrics are not directly comparable across these runs."
            )
            for reason in self.incompatibility_reasons:
                lines.append(f"> - {reason}")
            lines.append("")
        else:
            lines.append(
                f"*Runs ranked by {self.split.title()} Macro BPC (ascending). "
                "Token perplexity is per-token and not comparable across tokenizer families.*"
            )
            lines.append("")

        headers = [
            "Run",
            "Tokenizer",
            "Mode",
            "α",
            "Params",
            f"{self.split.title()} Macro BPC",
            f"{self.split.title()} Micro BPC",
            "Token PPL",
            "Longest Match",
            "Distinct-4",
            "Chars Exposed",
            "Tokens Exposed",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")

        for r in self.runs:
            alpha_str = f"{r.sampling_alpha:.2f}" if r.sampling_alpha is not None else "N/A"
            match_str = (
                f"{r.longest_match_chars} ({r.longest_match_source_doc})"
                if r.longest_match_chars > 0
                else "0"
            )
            dist_str = (
                f"{r.repetition_distinct_4:.3f}" if r.repetition_distinct_4 is not None else "N/A"
            )
            row = [
                f"**{r.run_name}**",
                r.tokenizer_type.upper(),
                r.sampling_mode,
                alpha_str,
                f"{r.parameter_count:,}",
                f"**{r.macro_bpc:.4f}**",
                f"{r.micro_bpc:.4f}",
                f"{r.token_perplexity:.2f}",
                match_str,
                dist_str,
                f"{r.training_chars_exposed:,}",
                f"{r.training_tokens_exposed:,}",
            ]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


def load_run_summary(run_dir: Path | str, split: str = "validation") -> RunSummary:
    """Load run metrics, configuration, training state, and evaluation results."""
    p = Path(run_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {p}")

    # 1. Load run configuration
    cfg_file = p / "config.toml"
    cfg_dict: dict[str, Any] = {}
    if cfg_file.is_file():
        import tomllib

        cfg_dict = tomllib.loads(cfg_file.read_text(encoding="utf-8"))

    tok_type = cfg_dict.get("tokenizer", {}).get("type", "unknown")
    sampling_mode = cfg_dict.get("data", {}).get("strategy", "unknown")
    sampling_alpha = cfg_dict.get("data", {}).get("temperature_alpha", None)
    if sampling_mode == "natural":
        sampling_alpha = None

    # 2. Checkpoint inspection
    ckpt_dir = p / "checkpoints" / "best"
    if not ckpt_dir.is_dir():
        # Fallback to latest
        ckpt_dir = p / "checkpoints" / "latest"

    curr_ckpt_hash = ""
    param_count = 0
    training_chars = 0
    training_tokens = 0

    if ckpt_dir.is_dir():
        model_file = ckpt_dir / "model.safetensors"
        if model_file.is_file():
            curr_ckpt_hash = compute_file_sha256(model_file)
        meta_file = ckpt_dir / "metadata.json"
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            training_chars = int(meta.get("cumulative_raw_chars", 0))
            training_tokens = int(meta.get("cumulative_model_tokens", 0))
            param_count = int(meta.get("trainable_parameter_count", 0))

    # 3. Environment & Split Metadata
    corpus_fp = ""
    norm_fp = ""
    split_hash = ""
    lock_file = p / "corpus_lock.json"
    if lock_file.is_file():
        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        corpus_fp = lock_data.get("corpus_fingerprint", "")
        norm_fp = lock_data.get("normalization_fingerprint", "")

    split_file = p / "split_manifest.json"
    if split_file.is_file():
        split_hash = compute_file_sha256(split_file)

    # 4. Evaluation Metrics
    eval_file = p / "evaluation" / f"{split}_metrics.json"
    macro_bpc = float("inf")
    micro_bpc = float("inf")
    token_ppl = float("inf")
    token_ce = float("inf")
    family_bpcs: dict[str, float] = {}
    eval_ckpt_hash = ""
    evaluated_chars = 0
    evaluated_chunks = 0

    if eval_file.is_file():
        eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
        macro_bpc = float(eval_data.get("macro_bpc", float("inf")))
        micro_bpc = float(eval_data.get("micro_bpc", float("inf")))
        token_ppl = float(eval_data.get("token_perplexity", float("inf")))
        token_ce = float(eval_data.get("token_cross_entropy", float("inf")))
        family_bpcs = {k: float(v) for k, v in eval_data.get("family_bpc", {}).items()}
        eval_ckpt_hash = str(eval_data.get("checkpoint_model_sha256", ""))
        evaluated_chars = int(eval_data.get("evaluated_characters", 0))
        evaluated_chunks = int(eval_data.get("evaluated_chunks", 0))
        if not corpus_fp:
            corpus_fp = str(eval_data.get("corpus_fingerprint", ""))
        if not norm_fp:
            norm_fp = str(eval_data.get("normalization_fingerprint", ""))
        if not split_hash:
            split_hash = str(eval_data.get("split_manifest_hash", ""))

    # 5. Memorization & Repetition
    memo_file = p / "evaluation" / "memorization_report.json"
    longest_match = 0
    longest_doc: str | None = None
    if memo_file.is_file():
        memo_data = json.loads(memo_file.read_text(encoding="utf-8"))
        longest_match = int(memo_data.get("max_longest_match_chars", 0))
        # Find sample with longest match
        for s in memo_data.get("samples", []):
            if int(s.get("longest_exact_match_chars", 0)) == longest_match:
                longest_doc = s.get("longest_match_source_doc")
                break

    rep_file = p / "evaluation" / "repetition_report.json"
    dist_4: float | None = None
    if rep_file.is_file():
        rep_data = json.loads(rep_file.read_text(encoding="utf-8"))
        dist_4 = float(rep_data.get("mean_distinct_4", 0.0))

    return RunSummary(
        run_name=p.name,
        run_path=str(p),
        tokenizer_type=tok_type,
        sampling_mode=sampling_mode,
        sampling_alpha=sampling_alpha,
        parameter_count=param_count,
        macro_bpc=macro_bpc,
        micro_bpc=micro_bpc,
        family_bpc=family_bpcs,
        token_perplexity=token_ppl,
        token_cross_entropy=token_ce,
        longest_match_chars=longest_match,
        longest_match_source_doc=longest_doc,
        repetition_distinct_4=dist_4,
        training_chars_exposed=training_chars,
        training_tokens_exposed=training_tokens,
        corpus_fingerprint=corpus_fp,
        normalization_fingerprint=norm_fp,
        split_manifest_hash=split_hash,
        evaluation_checkpoint_hash=eval_ckpt_hash,
        current_checkpoint_hash=curr_ckpt_hash,
        evaluated_characters=evaluated_chars,
        evaluated_chunks=evaluated_chunks,
        evaluation_split=split,
    )


def compare_runs(
    run_dirs: Sequence[Path | str],
    split: str = "validation",
    allow_incompatible: bool = False,
) -> ComparisonReport:
    """Compare multiple runs, enforcing strict cryptographic provenance.

    Args:
        run_dirs: Paths to run directories to compare.
        split: Evaluation split ('validation' or 'test'). Defaults to 'validation'.
        allow_incompatible: If True, bypasses refusal on mismatched provenance,
            disables performance ranking, preserves input order, and marks table.

    Returns:
        ComparisonReport containing summary for each run and compatibility verdict.
    """
    if not run_dirs:
        raise ValueError("At least one run directory must be provided for comparison.")

    summaries = [load_run_summary(d, split=split) for d in run_dirs]
    incompatibility_reasons: list[str] = []

    # 1. Stale Evaluation Report Check
    for s in summaries:
        if (
            s.current_checkpoint_hash
            and s.evaluation_checkpoint_hash
            and s.current_checkpoint_hash != s.evaluation_checkpoint_hash
        ):
            incompatibility_reasons.append(
                f"Run '{s.run_name}' evaluation report is stale: model.safetensors hash "
                f"({s.current_checkpoint_hash[:10]}...) does not match evaluation hash "
                f"({s.evaluation_checkpoint_hash[:10]}...). Re-run 'scripture-lm evaluate'."
            )

    # 2. Cross-Run Provenance Compatibility
    ref = summaries[0]
    for s in summaries[1:]:
        if (
            ref.corpus_fingerprint
            and s.corpus_fingerprint
            and ref.corpus_fingerprint != s.corpus_fingerprint
        ):
            incompatibility_reasons.append(
                f"corpus_fingerprint differs between '{ref.run_name}' ({ref.corpus_fingerprint}) "
                f"and '{s.run_name}' ({s.corpus_fingerprint})."
            )
        if (
            ref.normalization_fingerprint
            and s.normalization_fingerprint
            and ref.normalization_fingerprint != s.normalization_fingerprint
        ):
            incompatibility_reasons.append(
                f"normalization_fingerprint differs between '{ref.run_name}' and '{s.run_name}'."
            )
        if (
            ref.split_manifest_hash
            and s.split_manifest_hash
            and ref.split_manifest_hash != s.split_manifest_hash
        ):
            incompatibility_reasons.append(
                f"split_manifest_hash differs between '{ref.run_name}' "
                f"({ref.split_manifest_hash[:10]}...) and '{s.run_name}' "
                f"({s.split_manifest_hash[:10]}...)."
            )
        if (
            ref.evaluated_characters > 0
            and s.evaluated_characters > 0
            and ref.evaluated_characters != s.evaluated_characters
        ):
            incompatibility_reasons.append(
                f"Evaluated character count differs for split '{split}' between '{ref.run_name}' "
                f"({ref.evaluated_characters:,}) and '{s.run_name}' ({s.evaluated_characters:,})."
            )

    is_compatible = len(incompatibility_reasons) == 0

    if not is_compatible and not allow_incompatible:
        err_msg = (
            "Cannot directly compare runs due to incompatible provenance:\n"
            + "\n".join(f"  - {r}" for r in incompatibility_reasons)
            + "\n\nUse --allow-incompatible to force display "
            "(performance ranking will be disabled)."
        )
        raise ValueError(err_msg)

    # Ranking behavior:
    # If compatible: sort by selected split's Macro BPC (ascending).
    # If incompatible: preserve original input order, DO NOT rank.
    if is_compatible:
        sorted_summaries = sorted(summaries, key=lambda s: s.macro_bpc)
    else:
        sorted_summaries = summaries

    return ComparisonReport(
        split=split,
        is_compatible=is_compatible,
        incompatibility_reasons=incompatibility_reasons,
        ranking_metric="macro_bpc" if is_compatible else "none",
        runs=sorted_summaries,
    )


def render_comparison_table(report: ComparisonReport, console: Console | None = None) -> Table:
    """Render a Rich Table comparing experiment runs."""
    title = f"Cross-Tokenizer Experiment Comparison ({report.split.title()} Split)"
    table = Table(title=title, show_header=True, header_style="bold cyan")

    table.add_column("Run", style="bold yellow")
    table.add_column("Tokenizer", style="bold green", justify="center")
    table.add_column("Sampling", justify="center")
    table.add_column("α", justify="center")
    table.add_column("Params", justify="right")
    table.add_column(f"{report.split.title()} Macro BPC", style="bold magenta", justify="right")
    table.add_column(f"{report.split.title()} Micro BPC", justify="right")
    table.add_column("Token PPL", justify="right")
    table.add_column("Longest Match", justify="center")
    table.add_column("Distinct-4", justify="right")
    table.add_column("Chars Exposed", justify="right")
    table.add_column("Tokens Exposed", justify="right")

    for r in report.runs:
        alpha_str = f"{r.sampling_alpha:.2f}" if r.sampling_alpha is not None else "N/A"
        match_str = (
            f"{r.longest_match_chars} ({r.longest_match_source_doc})"
            if r.longest_match_chars > 0
            else "0"
        )
        dist_str = (
            f"{r.repetition_distinct_4:.3f}" if r.repetition_distinct_4 is not None else "N/A"
        )
        bpc_style = "bold green" if r.macro_bpc < 2.0 else "white"

        table.add_row(
            r.run_name,
            r.tokenizer_type.upper(),
            r.sampling_mode,
            alpha_str,
            f"{r.parameter_count:,}" if r.parameter_count > 0 else "N/A",
            f"[{bpc_style}]{r.macro_bpc:.4f}[/]" if r.macro_bpc < float("inf") else "N/A",
            f"{r.micro_bpc:.4f}" if r.micro_bpc < float("inf") else "N/A",
            f"{r.token_perplexity:.2f}" if r.token_perplexity < float("inf") else "N/A",
            match_str,
            dist_str,
            f"{r.training_chars_exposed:,}" if r.training_chars_exposed > 0 else "0",
            f"{r.training_tokens_exposed:,}" if r.training_tokens_exposed > 0 else "0",
        )

    return table
