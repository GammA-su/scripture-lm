"""Command-line interface for Scripture-LM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from safetensors.torch import load_model

from scripture_lm.config import BPETokenizerConfig, ScriptureLMConfig, load_config
from scripture_lm.corpus import (
    CorpusLock,
    SplitManifest,
    audit_corpus,
    compute_corpus_stats,
    load_manifest,
    load_or_generate_splits,
    normalize_corpus,
    render_audit_report,
    render_stats_table,
)
from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.data import (
    calculate_temperature_parameters,
    encode_dataset,
    load_chunk_index,
    simulate_sampling,
)
from scripture_lm.evaluation import (
    compare_runs,
    render_comparison_table,
)
from scripture_lm.experiments.cli import app as experiment_app
from scripture_lm.model import TransformerConfig, TransformerLM
from scripture_lm.tokenization import (
    BaseTokenizer,
    BPETokenizer,
    CharacterTokenizer,
    build_character_tokenizer,
    compute_and_update_tokenizer_stats,
    render_tokenizer_stats,
    train_bpe_tokenizer,
)

app = typer.Typer(
    name="scripture-lm",
    help="Scripture-LM: Language models trained from scratch on Abrahamic scripture.",
    no_args_is_help=True,
    add_completion=False,
)

corpus_app = typer.Typer(
    name="corpus",
    help="Audit, prepare, and analyze the scripture corpus.",
    no_args_is_help=True,
)
app.add_typer(corpus_app)

tokenizer_app = typer.Typer(
    name="tokenizer",
    help="Train BPE tokenizer or build character vocabulary.",
    no_args_is_help=True,
)
app.add_typer(tokenizer_app)

config_app = typer.Typer(
    name="config",
    help="Inspect and validate configurations.",
    no_args_is_help=True,
)
app.add_typer(config_app)

app.add_typer(experiment_app)

console = Console()


def display_config_table(config: ScriptureLMConfig, title: str = "Resolved Configuration") -> None:
    """Render configuration sections in a clean Rich table."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Section", style="bold yellow", width=14)
    table.add_column("Parameter", style="bold green", width=26)
    table.add_column("Value", style="white")

    # Model
    table.add_row("model", "layers", str(config.model.layers))
    table.add_row("model", "d_model", str(config.model.d_model))
    table.add_row("model", "heads", str(config.model.heads))
    table.add_row("model", "head_dim (derived)", str(config.model.head_dim))
    table.add_row("model", "mlp_hidden", str(config.model.mlp_hidden))
    table.add_row("model", "rope_theta", str(config.model.rope_theta))
    table.add_row("model", "dropout", str(config.model.dropout))

    # Tokenizer
    table.add_row("tokenizer", "type", config.tokenizer.type)
    if config.tokenizer.type == "bpe":
        table.add_row("tokenizer", "bpe_vocab_size", str(config.tokenizer.bpe_vocab_size))
    table.add_row("tokenizer", "context_length", str(config.tokenizer.context_length))

    # Data
    table.add_row("data", "sampling_mode", config.data.sampling_mode)
    alpha_desc = (
        f"{config.data.sampling_alpha} (active)"
        if config.data.sampling_mode == "temperature"
        else f"{config.data.sampling_alpha} (ignored in natural mode)"
    )
    table.add_row("data", "sampling_alpha", alpha_desc)

    # Training
    table.add_row("training", "seed", str(config.training.seed))
    table.add_row("training", "learning_rate", str(config.training.learning_rate))
    table.add_row("training", "precision", config.training.precision)
    table.add_row("training", "compile", str(config.training.compile))
    table.add_row("training", "device", config.training.device)
    table.add_row("training", "max_effective_epochs", str(config.training.max_effective_epochs))

    console.print(table)


# =========================================================================
# Config Commands
# =========================================================================


@config_app.command(name="show")
def config_show(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to model configuration TOML (e.g. configs/bpe.toml, configs/char.toml)",
        ),
    ] = Path("configs/bpe.toml"),
) -> None:
    """Inspect the fully resolved configuration after applying base.toml and overrides."""
    try:
        cfg = load_config(config_path=config)
        display_config_table(cfg, title=f"Resolved Configuration ({config})")
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/] {e}")
        raise typer.Exit(code=1) from e


# =========================================================================
# Corpus Commands
# =========================================================================


@corpus_app.command(name="audit")
def corpus_audit(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", "-m", help="Path to corpus_manifest.toml"),
    ] = Path("corpus/corpus_manifest.toml"),
    corpus_root: Annotated[
        Path,
        typer.Option("--corpus-root", help="Path to corpus directory"),
    ] = Path("corpus"),
) -> None:
    """Audit corpus directory against corpus_manifest.toml for purity and hash verification."""
    result = audit_corpus(manifest_path=manifest, corpus_root=corpus_root)
    render_audit_report(result)
    if not result.is_clean:
        raise typer.Exit(code=1)


@corpus_app.command(name="prepare")
def corpus_prepare(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", "-m", help="Path to corpus_manifest.toml"),
    ] = Path("corpus/corpus_manifest.toml"),
    corpus_root: Annotated[
        Path,
        typer.Option("--corpus-root", help="Path to corpus directory"),
    ] = Path("corpus"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Destination directory for normalized text"),
    ] = Path("data/normalized"),
    split_path: Annotated[
        Path,
        typer.Option("--split-manifest", help="Path to save or verify split_manifest.json"),
    ] = Path("data/splits/split_manifest.json"),
    lock_path: Annotated[
        Path,
        typer.Option("--lock-path", help="Path to save data/corpus_lock.json"),
    ] = Path("data/corpus_lock.json"),
    force_split: Annotated[
        bool,
        typer.Option("--force-split", help="Force regeneration of train/val/test splits"),
    ] = False,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic random seed for splitting"),
    ] = 1337,
) -> None:
    """Normalize scripture text and create deterministic train/val/test splits."""
    console.print("[bold cyan]Step 1: Auditing corpus for purity and integrity...[/]")
    audit_res = audit_corpus(manifest_path=manifest, corpus_root=corpus_root)
    if not audit_res.is_clean:
        render_audit_report(audit_res)
        console.print("[bold red]Corpus audit failed. Halting preparation.[/]")
        raise typer.Exit(code=1)

    if audit_res.num_documents == 0:
        console.print(
            "[bold yellow]Corpus manifest is empty. "
            "Register documents in corpus/corpus_manifest.toml first.[/]"
        )
        raise typer.Exit(code=0)

    console.print("[bold cyan]Step 2: Performing conservative normalization...[/]")
    manifest_obj = load_manifest(manifest)
    lock = normalize_corpus(
        manifest=manifest_obj,
        corpus_root=corpus_root,
        output_dir=output_dir,
        lock_path=lock_path,
    )
    console.print(
        f"[green]Normalized {len(lock.documents)} documents. Lock saved to {lock_path}[/]"
    )

    console.print("[bold cyan]Step 3: Generating/validating document splits...[/]")
    try:
        split_res, was_reused = load_or_generate_splits(
            manifest=manifest_obj,
            corpus_lock=lock,
            split_path=split_path,
            seed=seed,
            force_split=force_split,
        )
    except ValueError as e:
        console.print(f"[bold red]Split error:[/] {e}")
        raise typer.Exit(code=1) from e

    status_msg = "Reused matching split manifest" if was_reused else "Generated new split manifest"
    tr_pct = split_res.actual_proportions.get("train", 0.0)
    val_pct = split_res.actual_proportions.get("validation", 0.0)
    te_pct = split_res.actual_proportions.get("test", 0.0)

    console.print(
        Panel(
            f"[bold green]Corpus preparation complete![/]\n"
            f"{status_msg}: {split_path}\n"
            f"Train: {len(split_res.train)} docs ({tr_pct:.1%})\n"
            f"Val:   {len(split_res.validation)} docs ({val_pct:.1%})\n"
            f"Test:  {len(split_res.test)} docs ({te_pct:.1%})\n"
            f"Corpus Fingerprint: {lock.corpus_fingerprint[:16]}...\n"
            f"Normalization Fingerprint: {lock.normalization_fingerprint[:16]}...",
            title="Preparation Summary",
            border_style="green",
        )
    )


@corpus_app.command(name="stats")
def corpus_stats(
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", help="Path to split_manifest.json"),
    ] = Path("data/splits/split_manifest.json"),
    normalized_dir: Annotated[
        Path,
        typer.Option("--normalized-dir", help="Path to data/normalized directory"),
    ] = Path("data/normalized"),
) -> None:
    """Compute and display character and document counts by scripture family."""
    if not split_manifest.is_file():
        console.print(
            f"[bold red]Split manifest not found:[/] {split_manifest}\n"
            "Run 'uv run scripture-lm corpus prepare' first."
        )
        raise typer.Exit(code=1)

    import json

    with open(split_manifest, "r", encoding="utf-8") as f:
        split_obj = SplitManifest.model_validate(json.load(f))

    stats_data = compute_corpus_stats(split_obj, normalized_dir=normalized_dir)
    render_stats_table(stats_data)


@corpus_app.command(name="sampling-preview")
def corpus_sampling_preview(
    sampling_mode: Annotated[
        str,
        typer.Option("--sampling-mode", help="Sampling strategy ('natural' or 'temperature')"),
    ] = "temperature",
    sampling_alpha: Annotated[
        float,
        typer.Option("--sampling-alpha", help="Temperature alpha exponent [0.0, 1.0]"),
    ] = 0.5,
    draws: Annotated[
        int,
        typer.Option("--draws", help="Number of simulated draws for preview"),
    ] = 100_000,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for simulated draws"),
    ] = 1337,
    chunks_index: Annotated[
        Path | None,
        typer.Option("--chunks-index", help="Optional path to chunks JSON index"),
    ] = None,
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", help="Path to split_manifest.json"),
    ] = Path("data/splits/split_manifest.json"),
    corpus_lock: Annotated[
        Path,
        typer.Option("--corpus-lock", help="Path to data/corpus_lock.json"),
    ] = Path("data/corpus_lock.json"),
) -> None:
    """Preview theoretical and simulated family exposure for natural and temperature modes."""
    # 1. Gather family chunk counts and raw target characters
    family_chunk_counts: dict[str, int] = {}
    family_raw_chars: dict[str, int] = {}

    target_chunk_file = chunks_index
    if target_chunk_file is None:
        default_bpe = Path("data/encoded/bpe/train_chunks.json")
        default_char = Path("data/encoded/character/train_chunks.json")
        if default_bpe.is_file():
            target_chunk_file = default_bpe
        elif default_char.is_file():
            target_chunk_file = default_char

    if target_chunk_file and target_chunk_file.is_file():
        chunks = load_chunk_index(target_chunk_file)
        train_chunks = [c for c in chunks if c.split == "train"] or chunks
        for c in train_chunks:
            family_chunk_counts[c.family] = family_chunk_counts.get(c.family, 0) + 1
            family_raw_chars[c.family] = family_raw_chars.get(c.family, 0) + c.raw_character_count
    else:
        # Fall back to split_manifest.json and corpus_lock.json
        if not split_manifest.is_file() or not corpus_lock.is_file():
            console.print(
                "[bold red]Cannot run sampling preview:[/] neither encoded chunks nor "
                f"corpus split files found at {split_manifest} and {corpus_lock}."
            )
            raise typer.Exit(code=1)

        split_obj = SplitManifest.model_validate_json(split_manifest.read_text(encoding="utf-8"))
        lock_obj = CorpusLock.model_validate_json(corpus_lock.read_text(encoding="utf-8"))
        prov_map = {doc.document_id: doc for doc in lock_obj.documents}

        for doc_id in split_obj.train:
            fam = prov_map[doc_id].family
            chars = prov_map[doc_id].normalized_characters
            family_raw_chars[fam] = family_raw_chars.get(fam, 0) + chars
            # Approximate chunk count assuming mean context window of ~1500 chars
            family_chunk_counts[fam] = family_chunk_counts.get(fam, 0) + max(1, round(chars / 1500))

    if not family_raw_chars:
        console.print("[bold yellow]No training documents or chunks available for preview.[/]")
        return

    mode = sampling_mode.lower()
    alpha = float(sampling_alpha)

    if mode == "natural":
        params = calculate_temperature_parameters(family_chunk_counts, family_raw_chars, alpha=1.0)
        table = Table(title="Sampling Preview: NATURAL Mode (Single pass without replacement)")
        table.add_column("Family", style="bold cyan")
        table.add_column("Natural raw %", justify="right")
        table.add_column("Target raw %", justify="right")
        table.add_column("Draw probability", justify="right")
        table.add_column("Observed draws %", justify="right")
        table.add_column("Observed raw %", justify="right")

        for fam in sorted(family_raw_chars.keys()):
            p = params[fam]
            table.add_row(
                fam,
                f"{p['natural_raw_share']:.2%}",
                f"{p['natural_raw_share']:.2%}",
                "100% natural",
                f"{p['natural_raw_share']:.2%}",
                f"{p['natural_raw_share']:.2%}",
            )
        console.print(table)
        return

    # Temperature mode
    params = calculate_temperature_parameters(family_chunk_counts, family_raw_chars, alpha=alpha)
    sim = simulate_sampling(
        family_chunk_counts, family_raw_chars, alpha=alpha, draws=draws, seed=seed
    )

    table = Table(
        title=f"Sampling Preview: TEMPERATURE Mode (alpha={alpha:.2f}, {draws:,} simulated draws)"
    )
    table.add_column("Family", style="bold cyan")
    table.add_column("Natural raw %", justify="right")
    table.add_column("Target raw %", justify="right")
    table.add_column("Draw probability", justify="right")
    table.add_column("Observed draws %", justify="right")
    table.add_column("Observed raw %", justify="right")

    for fam in sorted(family_raw_chars.keys()):
        p = params[fam]
        s = sim[fam]
        table.add_row(
            fam,
            f"{p['natural_raw_share']:.2%}",
            f"{p['target_exposure_share']:.2%}",
            f"{p['family_draw_probability']:.2%}",
            f"{s['observed_draw_pct']:.2%}",
            f"{s['observed_raw_pct']:.2%}",
        )
    console.print(table)


# =========================================================================
# Tokenizer Commands
# =========================================================================


@tokenizer_app.command(name="train-bpe")
def tokenizer_train_bpe(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to BPE model config TOML"),
    ] = Path("configs/bpe.toml"),
    vocab_size: Annotated[
        int | None,
        typer.Option("--vocab-size", help="Optional override for BPE vocabulary size"),
    ] = None,
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", help="Path to split_manifest.json"),
    ] = Path("data/splits/split_manifest.json"),
    corpus_lock: Annotated[
        Path,
        typer.Option("--corpus-lock", help="Path to data/corpus_lock.json"),
    ] = Path("data/corpus_lock.json"),
    normalized_dir: Annotated[
        Path,
        typer.Option("--normalized-dir", help="Path to normalized corpus directory"),
    ] = Path("data/normalized"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Path to save tokenizer artifacts"),
    ] = Path("artifacts/tokenizers"),
) -> None:
    """Train a byte-level BPE tokenizer from scratch on the training split only."""
    try:
        cfg = load_config(config_path=config)
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/] {e}")
        raise typer.Exit(code=1) from e

    v_size: int = (
        vocab_size
        if vocab_size is not None
        else int(getattr(cfg.tokenizer, "bpe_vocab_size", 4096))
    )
    console.print(f"[bold cyan]Training BPE tokenizer (vocab_size={v_size}) from scratch...[/]")

    try:
        tok, meta = train_bpe_tokenizer(
            split_manifest_path=split_manifest,
            corpus_lock_path=corpus_lock,
            normalized_dir=normalized_dir,
            output_dir=output_dir,
            vocab_size=v_size,
            config_dict=cfg.model_dump(),
        )
        console.print(f"[green]Saved BPE tokenizer to {output_dir / 'bpe.json'}[/]")
        stats_data = compute_and_update_tokenizer_stats(
            tokenizer=tok,
            metadata=meta,
            split_manifest_path=split_manifest,
            corpus_lock_path=corpus_lock,
            normalized_dir=normalized_dir,
            stats_path=output_dir / "tokenizer_stats.json",
        )
        render_tokenizer_stats(stats_data)
    except Exception as e:
        console.print(f"[bold red]BPE training failed:[/] {e}")
        raise typer.Exit(code=1) from e


@tokenizer_app.command(name="build-char")
def tokenizer_build_char(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to Character model config TOML"),
    ] = Path("configs/char.toml"),
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", help="Path to split_manifest.json"),
    ] = Path("data/splits/split_manifest.json"),
    corpus_lock: Annotated[
        Path,
        typer.Option("--corpus-lock", help="Path to data/corpus_lock.json"),
    ] = Path("data/corpus_lock.json"),
    normalized_dir: Annotated[
        Path,
        typer.Option("--normalized-dir", help="Path to normalized corpus directory"),
    ] = Path("data/normalized"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Path to save tokenizer artifacts"),
    ] = Path("artifacts/tokenizers"),
) -> None:
    """Construct character tokenizer vocabulary directly from training split."""
    try:
        cfg = load_config(config_path=config)
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/] {e}")
        raise typer.Exit(code=1) from e

    console.print("[bold cyan]Building character tokenizer from train split codepoints...[/]")

    try:
        tok, meta = build_character_tokenizer(
            split_manifest_path=split_manifest,
            corpus_lock_path=corpus_lock,
            normalized_dir=normalized_dir,
            output_dir=output_dir,
            config_dict=cfg.model_dump(),
        )
        console.print(f"[green]Saved character vocabulary to {output_dir / 'char_vocab.json'}[/]")
        stats_data = compute_and_update_tokenizer_stats(
            tokenizer=tok,
            metadata=meta,
            split_manifest_path=split_manifest,
            corpus_lock_path=corpus_lock,
            normalized_dir=normalized_dir,
            stats_path=output_dir / "tokenizer_stats.json",
        )
        render_tokenizer_stats(stats_data)
    except Exception as e:
        console.print(f"[bold red]Character vocabulary build failed:[/] {e}")
        raise typer.Exit(code=1) from e


@tokenizer_app.command(name="stats")
def tokenizer_stats(
    stats_path: Annotated[
        Path,
        typer.Option("--stats-path", help="Path to artifacts/tokenizers/tokenizer_stats.json"),
    ] = Path("artifacts/tokenizers/tokenizer_stats.json"),
) -> None:
    """Display comparative tokenizer statistics from artifacts/tokenizers/tokenizer_stats.json."""
    if not stats_path.is_file():
        console.print(
            f"[bold red]Tokenizer stats file not found:[/] {stats_path}\n"
            "Train BPE or build character vocabulary first."
        )
        raise typer.Exit(code=1)

    import json

    with open(stats_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    render_tokenizer_stats(stats_data)


# =========================================================================
# Data Encoding Command
# =========================================================================


@app.command(name="encode")
def encode(
    tokenizer: Annotated[
        str,
        typer.Option(
            "--tokenizer",
            "-t",
            help="Tokenizer type to encode with ('bpe', 'char', or 'character')",
        ),
    ] = "bpe",
    split_manifest: Annotated[
        Path,
        typer.Option("--split-manifest", help="Path to split_manifest.json"),
    ] = Path("data/splits/split_manifest.json"),
    corpus_lock: Annotated[
        Path,
        typer.Option("--corpus-lock", help="Path to data/corpus_lock.json"),
    ] = Path("data/corpus_lock.json"),
    normalized_dir: Annotated[
        Path,
        typer.Option("--normalized-dir", help="Path to normalized corpus directory"),
    ] = Path("data/normalized"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Path to output encoded streams"),
    ] = Path("data/encoded"),
    tokenizer_dir: Annotated[
        Path,
        typer.Option("--tokenizer-dir", help="Path to tokenizer artifacts directory"),
    ] = Path("artifacts/tokenizers"),
    context_length: Annotated[
        int | None,
        typer.Option("--context-length", help="Optional override for context length L"),
    ] = None,
) -> None:
    """Encode train/val/test splits into uint16 memory-mapped binary token streams and chunks."""
    norm_type: Literal["bpe", "character"] = (
        "character" if tokenizer.lower() in ("char", "character") else "bpe"
    )

    console.print(f"[bold cyan]Loading {norm_type.upper()} tokenizer...[/]")
    try:
        if norm_type == "bpe":
            tok_path = tokenizer_dir / "bpe.json"
            if not tok_path.is_file():
                console.print(
                    f"[bold red]BPE tokenizer artifact not found:[/] {tok_path}\n"
                    "Train BPE first: 'uv run scripture-lm tokenizer train-bpe'."
                )
                raise typer.Exit(code=1)
            tok_instance: BaseTokenizer = BPETokenizer.load(tok_path)
        else:
            tok_path = tokenizer_dir / "char_vocab.json"
            if not tok_path.is_file():
                console.print(
                    f"[bold red]Character vocabulary artifact not found:[/] {tok_path}\n"
                    "Build character vocabulary first: 'uv run scripture-lm tokenizer build-char'."
                )
                raise typer.Exit(code=1)
            tok_instance = CharacterTokenizer.load(tok_path)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]Failed to load tokenizer:[/] {e}")
        raise typer.Exit(code=1) from e

    console.print(f"[bold cyan]Encoding corpus with {norm_type.upper()} tokenizer...[/]")
    try:
        chunks_by_split, prov = encode_dataset(
            tokenizer=tok_instance,
            split_manifest_path=split_manifest,
            corpus_lock_path=corpus_lock,
            normalized_dir=normalized_dir,
            output_base_dir=output_dir,
            context_length=context_length,
        )
    except Exception as e:
        console.print(f"[bold red]Encoding failed:[/] {e}")
        raise typer.Exit(code=1) from e

    # Render summary table
    table = Table(title=f"Encoded Dataset Summary ({norm_type.upper()})")
    table.add_column("Split", style="bold cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Target Characters", justify="right")

    for s_name in ["train", "validation", "test"]:
        s_chunks = chunks_by_split.get(s_name, [])
        chars = sum(c.raw_character_count for c in s_chunks)
        table.add_row(s_name, f"{len(s_chunks):,}", f"{chars:,}")

    console.print(table)
    console.print(
        Panel(
            f"[bold green]Encoding complete![/]\n"
            f"Context Length L: {prov.context_length} (Window: {prov.chunk_length})\n"
            f"Natural Training Exposure N: "
            f"{prov.natural_train_target_characters:,} target characters\n"
            f"Binary Streams & Chunks: {output_dir / norm_type}",
            title="Success",
            border_style="green",
        )
    )


# =========================================================================
# Training Command
# =========================================================================


@app.command(name="train")
def train(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to model config TOML (e.g. configs/bpe.toml, configs/char.toml)",
        ),
    ] = Path("configs/bpe.toml"),
    sampling_mode: Annotated[
        str | None,
        typer.Option(
            "--sampling-mode",
            help="Sampling mode ('natural' or 'temperature')",
        ),
    ] = None,
    sampling_alpha: Annotated[
        float | None,
        typer.Option(
            "--sampling-alpha",
            help="Sampling temperature alpha (0.0 to 1.0)",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Compute device (e.g. 'cuda', 'cpu')",
        ),
    ] = None,
    compile_model: Annotated[
        bool | None,
        typer.Option(
            "--compile/--no-compile",
            help="Enable or disable torch.compile",
        ),
    ] = None,
    effective_epochs: Annotated[
        int | None,
        typer.Option(
            "--effective-epochs",
            help="Maximum effective epochs",
        ),
    ] = None,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            help="Destination run directory for logs, checkpoints, and metadata",
        ),
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help="Path to checkpoint directory to resume training from",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate configuration and data provenance without training",
        ),
    ] = False,
) -> None:
    """Train a Scripture-LM language model."""
    # Assemble CLI overrides
    overrides: dict[str, Any] = {}
    if sampling_mode is not None:
        overrides["data.sampling_mode"] = sampling_mode
    if sampling_alpha is not None:
        overrides["data.sampling_alpha"] = sampling_alpha
    if seed is not None:
        overrides["training.seed"] = seed
    if device is not None:
        overrides["training.device"] = device
    if compile_model is not None:
        overrides["training.compile"] = compile_model
    if effective_epochs is not None:
        overrides["training.max_effective_epochs"] = effective_epochs

    try:
        resolved_config = load_config(config_path=config, cli_overrides=overrides)
    except Exception as e:
        console.print(f"[bold red]Configuration validation error:[/] {e}")
        raise typer.Exit(code=1) from e

    display_config_table(resolved_config, title=f"Training Configuration ({config or 'default'})")

    if dry_run:
        console.print(
            Panel(
                "[bold green]Configuration successfully validated![/]\n"
                "[dim]Dry run complete; training was not started.[/]",
                title="Dry Run Verification",
                border_style="cyan",
            )
        )
        return

    # Check for encoded dataset before starting
    tok_type = resolved_config.tokenizer.type
    encoding_meta_file = Path("data/encoded") / tok_type / "encoding_metadata.json"
    if not encoding_meta_file.is_file():
        console.print(
            f"[bold red]ERROR: encoded {tok_type.upper()} dataset not found.[/]\n"
            f"Run:\n  uv run scripture-lm encode --tokenizer {tok_type}"
        )
        raise typer.Exit(code=1)

    try:
        from scripture_lm.training.trainer import Trainer

        trainer = Trainer(
            config=resolved_config,
            run_dir=run_dir,
            resume_checkpoint_dir=resume,
        )
        trainer.train()
    except Exception as e:
        console.print(f"[bold red]Training error:[/] {e}")
        raise typer.Exit(code=1) from e


# =========================================================================
# Generation Command
# =========================================================================


@app.command(name="generate")
def generate(
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", "-c", help="Path to checkpoint directory or safetensors file"),
    ] = Path("runs/bpe-natural/checkpoints/best"),
    prompt: Annotated[
        str,
        typer.Option("--prompt", "-p", help="Text prompt for continuation (empty for BOS-only)"),
    ] = "And the prophet said",
    temperature: Annotated[float, typer.Option(help="Sampling temperature (0.0 for greedy)")] = 0.8,
    top_p: Annotated[float, typer.Option(help="Nucleus sampling top-p threshold")] = 0.95,
    top_k: Annotated[int | None, typer.Option(help="Optional top-k filtering bound")] = None,
    max_new_tokens: Annotated[
        int | None,
        typer.Option(help="Maximum tokens to generate (default: 256 for BPE, 1024 for CHAR)"),
    ] = None,
    max_new_characters: Annotated[
        int | None,
        typer.Option(help="Optional maximum character length for generated continuation"),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Random sampling seed for reproducibility")] = 42,
    device: Annotated[str | None, typer.Option(help="Compute device ('cuda', 'cpu')")] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to save generation artifact JSON"),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable KV caching (for benchmarking/testing)"),
    ] = False,
) -> None:
    """Generate scriptural continuation from a trained checkpoint using KV caching."""
    from datetime import datetime

    from scripture_lm.evaluation.generation_suite import GenerationSettings
    from scripture_lm.generation.generate import (
        TextGenerator,
        create_generation_artifact,
    )

    # 1. Resolve model checkpoint path
    if checkpoint.is_file():
        model_file = checkpoint
        parent_name = checkpoint.parent.name
        if parent_name in ("best", "latest"):
            run_dir = checkpoint.parent.parent
        else:
            run_dir = checkpoint.parent
    else:
        if (checkpoint / "model.safetensors").is_file():
            model_file = checkpoint / "model.safetensors"
            run_dir = checkpoint.parent if checkpoint.name in ("best", "latest") else checkpoint
            if run_dir.name == "checkpoints":
                run_dir = run_dir.parent
        elif (checkpoint / "checkpoints" / "best" / "model.safetensors").is_file():
            model_file = checkpoint / "checkpoints" / "best" / "model.safetensors"
            run_dir = checkpoint
        else:
            console.print(f"[bold red]Checkpoint model not found in:[/] {checkpoint}")
            raise typer.Exit(code=1)

    # 2. Resolve device
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load config and provenance
    config_path = run_dir / "config.toml"
    if config_path.is_file():
        config = load_config(config_path)
    else:
        config = ScriptureLMConfig(tokenizer=BPETokenizerConfig())

    tok_type = getattr(config.tokenizer, "type", "bpe")
    if tok_type not in ("bpe", "character", "char"):
        tok_type = "bpe"
    if tok_type == "char":
        tok_type = "character"

    # 4. Load tokenizer
    tok_dir = Path("artifacts/tokenizers")
    tokenizer_inst: BaseTokenizer
    if tok_type == "bpe":
        bpe_path = run_dir / "bpe.json"
        if not bpe_path.is_file():
            bpe_path = tok_dir / "bpe.json"
        if not bpe_path.is_file():
            console.print(f"[bold red]BPE tokenizer artifact not found:[/] {bpe_path}")
            raise typer.Exit(code=1)
        tokenizer_inst = BPETokenizer.load(bpe_path)
    else:
        char_path = run_dir / "char_vocab.json"
        if not char_path.is_file():
            char_path = tok_dir / "char_vocab.json"
        if not char_path.is_file():
            console.print(f"[bold red]Character vocabulary artifact not found:[/] {char_path}")
            raise typer.Exit(code=1)
        tokenizer_inst = CharacterTokenizer.load(char_path)

    # 5. Instantiate Model
    vocab_size = tokenizer_inst.vocab_size
    model_cfg = TransformerConfig.from_app_config(config, vocab_size=vocab_size)
    model = TransformerLM(model_cfg)
    load_model(model, str(model_file))
    model.to(target_device)
    model.eval()

    # 6. Build GenerationSettings
    effective_max_tokens = (
        max_new_tokens if max_new_tokens is not None else (256 if tok_type == "bpe" else 1024)
    )
    settings = GenerationSettings(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=effective_max_tokens,
        max_new_characters=max_new_characters,
        seed=seed if seed is not None else 42,
    )

    # 7. Generate Continuation
    generator = TextGenerator(model, tokenizer_inst, device=target_device)
    result = generator.generate(prompt, settings, use_cache=not no_cache)

    # 8. Create GenerationArtifact
    model_sha256 = compute_file_sha256(model_file)
    sampling_mode = getattr(config.training, "sampling_strategy", None)
    temp_alpha = getattr(config.training, "temperature_alpha", None)

    artifact = create_generation_artifact(
        result=result,
        checkpoint_path=model_file,
        checkpoint_model_sha256=model_sha256,
        tokenizer_type=tok_type,
        settings=settings,
        training_sampling_mode=sampling_mode,
        training_temperature_alpha=temp_alpha,
    )

    # 9. Save Artifact if requested or into run_dir/generations
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact.model_dump(), indent=2), encoding="utf-8")
        console.print(f"[green]Saved generation artifact to {output}[/]")
    elif run_dir.is_dir():
        gen_dir = run_dir / "generations"
        gen_dir.mkdir(parents=True, exist_ok=True)
        default_out = gen_dir / f"generation_{int(datetime.now().timestamp())}.json"
        default_out.write_text(json.dumps(artifact.model_dump(), indent=2), encoding="utf-8")

    # 10. Display Visual Panel
    prompt_display = prompt if prompt else "[italic dim]<unprompted BOS>[/]"
    footer_info = (
        f"[dim]Finish: {result.finish_reason} | Tokens: {len(result.generated_token_ids)} | "
        f"Chars: {result.characters_generated} | Seed: {settings.seed} | "
        f"T: {settings.temperature} | p: {settings.top_p} | "
        f"KV-Cache: {'off' if no_cache else 'on'}[/]"
    )
    console.print(
        Panel(
            f"[bold magenta]SYNTHETIC MODEL OUTPUT[/]\n"
            f"[dim]Generated autoregressively by Scripture-LM. Not authentic scripture.[/]\n\n"
            f"[bold cyan]Prompt:[/] {prompt_display}\n"
            f"[bold green]Continuation:[/] {result.continuation}\n\n"
            f"{footer_info}",
            title="Scripture-LM Generation",
            border_style="cyan",
        )
    )


# =========================================================================
# Evaluation Commands
# =========================================================================


@app.command(name="evaluate")
def evaluate(
    run: Annotated[
        Path,
        typer.Option("--run", "-r", help="Path to experiment run directory"),
    ] = Path("runs/bpe"),
    split: Annotated[
        str,
        typer.Option("--split", "-s", help="Evaluation split ('validation' or 'test')"),
    ] = "validation",
    checkpoint: Annotated[
        str,
        typer.Option(
            "--checkpoint",
            "-c",
            help="Checkpoint to evaluate ('best', 'latest', or directory path)",
        ),
    ] = "best",
    device: Annotated[
        str | None,
        typer.Option("--device", help="Compute device ('cuda', 'cpu')"),
    ] = None,
    generate: Annotated[
        bool,
        typer.Option(
            "--generate/--no-generate", help="Run generation suite and memorization analysis"
        ),
    ] = False,
    data_root: Annotated[
        Path,
        typer.Option("--data-root", help="Path to data directory"),
    ] = Path("data"),
) -> None:
    """Evaluate a checkpoint on held-out data and optionally the canonical benchmark."""
    from scripture_lm.evaluation.runner import evaluate_run

    try:
        evaluate_run(run, split, checkpoint, device, generate, data_root)
    except (ValueError, OSError) as exc:
        console.print(f"[bold red]Evaluation error:[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="compare")
def compare(
    runs: Annotated[
        list[Path],
        typer.Argument(help="List of experiment run directories to compare (at least 2)"),
    ],
    split: Annotated[
        str,
        typer.Option("--split", "-s", help="Evaluation split to compare ('validation' or 'test')"),
    ] = "validation",
    allow_incompatible: Annotated[
        bool,
        typer.Option(
            "--allow-incompatible",
            help="Allow comparing runs with different corpus/split provenance (disables ranking)",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Optional path to save comparison report (.json or .md)"
        ),
    ] = None,
) -> None:
    """Compare cross-tokenizer performance between experiment runs.

    Defaults to --split validation to prevent test set snooping during development.
    Compatible runs are ranked strictly by Macro BPC of the selected split.
    """
    if len(runs) < 2:
        console.print("[bold red]At least 2 run directories are required for comparison.[/]")
        raise typer.Exit(code=1)

    try:
        report = compare_runs(runs, split=split, allow_incompatible=allow_incompatible)
    except ValueError as e:
        console.print(f"[bold red]Comparison Error:[/]\n{e}")
        raise typer.Exit(code=1)

    if not report.is_compatible:
        console.print(
            Panel(
                "[bold red]WARNING: INCOMPATIBLE RUNS[/]\n"
                "Runs do not share identical corpus/split provenance.\n"
                "Performance ranking has been disabled; runs are shown in input order.",
                style="bold red",
            )
        )

    table = render_comparison_table(report, console=console)
    console.print(table)
    console.print(
        "[dim]* Runs ranked by Macro BPC. "
        "Token perplexity is per-token and not comparable across tokenizer families.[/dim]"
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".json":
            output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        else:
            output.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"[green]Saved comparison report to {output}[/]")


# =========================================================================
# Experiment Matrix Command
# =========================================================================


def main() -> None:
    """Entry point for scripture-lm CLI."""
    app()


if __name__ == "__main__":
    main()
