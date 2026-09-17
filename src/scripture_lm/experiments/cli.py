"""Typer commands for canonical and custom experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.table import Table

from scripture_lm.config import load_config
from scripture_lm.experiments.matrix import baseline_matrix, custom_experiment, get_baseline
from scripture_lm.experiments.reporting import compare_baseline
from scripture_lm.experiments.runner import run_baseline, run_experiment

app = typer.Typer(
    name="experiment", help="Plan, run, and compare experiments.", no_args_is_help=True
)
console = Console()


def display_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        console.print(
            f"{result['name']}: {result['status']}"
            + (" (skipped)" if result.get("skipped") else "")
        )
        if "config" in result:
            console.print_json(data=result["config"])


@app.command(name="matrix")
@app.command(name="run-matrix", hidden=True)
def matrix() -> None:
    """Display the canonical baseline without creating files or starting training."""
    table = Table("NAME", "TOKENIZER", "SAMPLING", "ALPHA")
    for experiment in baseline_matrix():
        cfg = experiment.resolve()
        table.add_row(
            experiment.name,
            "bpe" if cfg.tokenizer.type == "bpe" else "char",
            cfg.data.sampling_mode,
            "0.5" if cfg.data.sampling_mode == "temperature" else "-",
        )
    console.print(table)


@app.command(name="run")
def run(
    name: Annotated[str, typer.Option("--name", help="Canonical baseline name")],
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    device: Annotated[str | None, typer.Option("--device")] = None,
    compile_model: Annotated[bool | None, typer.Option("--compile/--no-compile")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Run a fixed baseline; scientific overrides require run-custom."""
    try:
        result = run_experiment(
            get_baseline(name),
            runs_root=runs_root,
            device=device,
            compile_model=compile_model,
            resume=resume,
            dry_run=dry_run,
        )
        display_results([result])
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"[red]Experiment error:[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="run-baseline")
def baseline(
    names: Annotated[
        list[str] | None, typer.Option("--name", help="Repeat to select a subset")
    ] = None,
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    device: Annotated[str | None, typer.Option("--device")] = None,
    compile_model: Annotated[bool | None, typer.Option("--compile/--no-compile")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Run selected baselines sequentially; skip completed and retain failed runs."""
    try:
        if names:
            for name in names:
                get_baseline(name)
        selected = [e for e in baseline_matrix() if not names or e.name in names]
        display_results(
            run_baseline(
                selected,
                runs_root=runs_root,
                device=device,
                compile_model=compile_model,
                resume=resume,
                dry_run=dry_run,
            )
        )
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"[red]Baseline stopped:[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="run-custom")
def custom(
    tokenizer: Annotated[Literal["bpe", "char"], typer.Option("--tokenizer")],
    sampling_mode: Annotated[
        Literal["natural", "temperature"], typer.Option("--sampling-mode")
    ] = "natural",
    sampling_alpha: Annotated[str, typer.Option("--sampling-alpha")] = "0.5",
    effective_epochs: Annotated[int | None, typer.Option("--effective-epochs", min=1)] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    device: Annotated[str | None, typer.Option("--device")] = None,
    compile_model: Annotated[bool | None, typer.Option("--compile/--no-compile")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Create a distinct scientific identity, optionally from a custom TOML config."""
    try:
        cfg = load_config(config) if config else get_baseline(f"{tokenizer}-natural").resolve()
        experiment = custom_experiment(
            tokenizer,
            sampling_mode,
            sampling_alpha,
            effective_epochs if effective_epochs is not None else cfg.training.max_effective_epochs,
            seed if seed is not None else cfg.training.seed,
            config=cfg,
        )
        display_results(
            [
                run_experiment(
                    experiment,
                    runs_root=runs_root,
                    device=device,
                    compile_model=compile_model,
                    resume=resume,
                    dry_run=dry_run,
                )
            ]
        )
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"[red]Custom experiment error:[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="compare-baseline")
def comparison(
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    split: Annotated[Literal["validation", "test"], typer.Option("--split")] = "validation",
    device: Annotated[str | None, typer.Option("--device")] = None,
    generate: Annotated[bool, typer.Option("--generate/--no-generate")] = False,
) -> None:
    """Evaluate completed best checkpoints and export factual baseline reports."""
    try:
        payload = compare_baseline(
            runs_root=runs_root,
            output_dir=output_dir,
            split=split,
            device=device,
            generate=generate,
        )
        table = Table("Experiment", "Status", "Macro BPC", "Benchmark")
        for row in payload["runs"]:
            table.add_row(
                row["name"],
                row["status"],
                str(row["macro_bpc"]) if row["macro_bpc"] is not None else "—",
                row["generation_status"],
            )
        console.print(table)
        console.print(f"Reports: {output_dir or runs_root / 'baseline_comparison'}")
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"[red]Comparison error:[/] {exc}")
        raise typer.Exit(code=1) from exc
