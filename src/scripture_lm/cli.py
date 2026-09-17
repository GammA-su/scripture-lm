"""Command-line interface for Scripture-LM."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scripture_lm.config import ScriptureLMConfig, load_config
from scripture_lm.corpus import (
    SplitManifest,
    audit_corpus,
    compute_corpus_stats,
    load_manifest,
    load_or_generate_splits,
    normalize_corpus,
    render_audit_report,
    render_stats_table,
)
from scripture_lm.tokenization import (
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

experiment_app = typer.Typer(
    name="experiment",
    help="Experiment matrix execution and management.",
    no_args_is_help=True,
)
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
) -> None:
    """Encode train/val/test splits into uint16 memory-mapped binary token streams."""
    norm_tokenizer = "character" if tokenizer in ("char", "character") else "bpe"
    console.print(
        Panel(
            f"[bold yellow]Encoding Stub[/]\n"
            f"Will encode splits using {norm_tokenizer} tokenizer into uint16 bin files. "
            "Implementation will be completed in Prompt 04.",
            title="Encode Dataset",
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
) -> None:
    """Train a Scripture-LM model. (Phase 01 resolves and verifies configuration)."""
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

    console.print(
        Panel(
            "[bold green]Configuration successfully validated![/]\n"
            "[bold yellow]Note:[/] Training backend will be implemented in Prompt 06. "
            "In Phase 01, configuration loading and constraints are verified.",
            title="Phase 01 Verification",
            border_style="cyan",
        )
    )


# =========================================================================
# Generation Command
# =========================================================================


@app.command(name="generate")
def generate(
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", help="Path to checkpoint directory or safetensors file"),
    ] = Path("runs/bpe/checkpoints/best"),
    prompt: Annotated[
        str,
        typer.Option("--prompt", help="Text prompt continuation"),
    ] = "And the Lord said",
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.8,
    top_p: Annotated[float, typer.Option(help="Nucleus sampling top-p")] = 0.95,
    seed: Annotated[int | None, typer.Option(help="Sampling seed")] = 1337,
) -> None:
    """Generate scriptural continuation from a trained checkpoint."""
    console.print(
        Panel(
            f"[bold yellow]Generation Stub[/]\n"
            f"Checkpoint: {checkpoint}\n"
            f"Prompt: {prompt}\n"
            f"Temp: {temperature}, Top-p: {top_p}, Seed: {seed}\n"
            "Implementation will be completed in Prompt 07.",
            title="Generate Text",
        )
    )


# =========================================================================
# Evaluation Commands
# =========================================================================


@app.command(name="evaluate")
def evaluate(
    run: Annotated[
        Path,
        typer.Option("--run", help="Path to experiment run directory"),
    ] = Path("runs/bpe"),
) -> None:
    """Evaluate test BPC, perplexity, and memorization on an experiment run."""
    console.print(
        Panel(
            f"[bold yellow]Evaluation Stub[/]\n"
            f"Run: {run}\n"
            "Will evaluate BPC and memorization in Prompt 08.",
            title="Evaluate Run",
        )
    )


@app.command(name="compare")
def compare(
    run_a: Annotated[Path, typer.Argument(help="First run directory (e.g. runs/bpe)")],
    run_b: Annotated[Path, typer.Argument(help="Second run directory (e.g. runs/char)")],
) -> None:
    """Compare cross-tokenizer performance (BPC, memorization, speed) between two runs."""
    console.print(
        Panel(
            f"[bold yellow]Comparison Stub[/]\n"
            f"Comparing {run_a} vs {run_b}\n"
            "Implementation will be completed in Prompt 08.",
            title="Cross-Tokenizer Comparison",
        )
    )


# =========================================================================
# Experiment Matrix Command
# =========================================================================


@experiment_app.command(name="run-matrix")
def experiment_run_matrix() -> None:
    """Run the 2x2 baseline experimental matrix (BPE/CHAR x natural/temperature)."""
    console.print(
        Panel(
            "[bold yellow]Experiment Matrix Stub[/]\n"
            "Runs:\n"
            "  1. BPE + natural\n"
            "  2. BPE + temperature(alpha=0.5)\n"
            "  3. CHAR + natural\n"
            "  4. CHAR + temperature(alpha=0.5)\n"
            "Implementation will be completed in Prompt 08.",
            title="Experiment Matrix",
        )
    )


def main() -> None:
    """Entry point for scripture-lm CLI."""
    app()


if __name__ == "__main__":
    main()
