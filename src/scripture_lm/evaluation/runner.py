"""Reusable checkpoint evaluation and canonical generation benchmark execution."""

from __future__ import annotations

import contextlib
import json
import tomllib
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_model
from torch.utils.data import SequentialSampler

from scripture_lm.config import ScriptureLMConfig
from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.data import load_chunk_index
from scripture_lm.data.batching import create_dataloader
from scripture_lm.data.dataset import ScriptureChunkDataset
from scripture_lm.evaluation import compute_bpc, compute_perplexity
from scripture_lm.model import TransformerConfig, TransformerLM
from scripture_lm.tokenization import BaseTokenizer, BPETokenizer, CharacterTokenizer
from scripture_lm.tokenization.base import tokenizer_metadata_path

console = Console()


def evaluate_run(
    run: Path,
    split: str = "validation",
    checkpoint: str = "best",
    device: str | None = None,
    generate: bool = False,
    data_root: Path = Path("data"),
) -> None:
    """Evaluate a trained model checkpoint on held-out data (BPC and perplexity)."""
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    if not run.is_dir():
        console.print(f"[bold red]Run directory not found:[/] {run}")
        raise ValueError("Evaluation prerequisites are missing; see details above")

    # 1. Load config
    cfg_file = run / "config.toml"
    if not cfg_file.is_file():
        console.print(f"[bold red]Missing config.toml in run directory:[/] {cfg_file}")
        raise ValueError("Evaluation prerequisites are missing; see details above")

    cfg_dict = tomllib.loads(cfg_file.read_text(encoding="utf-8"))
    config = ScriptureLMConfig.model_validate(cfg_dict)
    tok_type = config.tokenizer.type

    # 2. Resolve checkpoint path
    if checkpoint in ("best", "latest"):
        ckpt_dir = run / "checkpoints" / checkpoint
    else:
        ckpt_dir = Path(checkpoint)

    model_file = ckpt_dir / "model.safetensors"
    if not model_file.is_file():
        console.print(f"[bold red]Model checkpoint not found:[/] {model_file}")
        raise ValueError("Evaluation prerequisites are missing; see details above")

    model_sha256 = compute_file_sha256(model_file)

    # 3. Determine device and precision autocast
    if device is not None:
        target_device = torch.device(device)
    else:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autocast_ctx: Any = contextlib.nullcontext()
    if target_device.type == "cuda":
        if config.training.precision == "bf16" and torch.cuda.is_bf16_supported():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif config.training.precision == "fp16":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)

    # Bind actual evaluation data to the immutable run before loading model weights.
    if (run / "experiment_config.toml").is_file():
        from scripture_lm.experiments.matrix import config_hash
        from scripture_lm.experiments.storage import read_spec
        from scripture_lm.training.trainer import verify_encoding_provenance

        _, _, encoded, _ = verify_encoding_provenance(config, data_root=data_root)
        specification = read_spec(run)
        if specification["provenance"] != encoded.model_dump(mode="json"):
            raise ValueError("Evaluation data differs from immutable run provenance")
        checkpoint_metadata = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
        if checkpoint_metadata.get("experiment_config_hash") != config_hash(specification):
            raise ValueError("Checkpoint belongs to a different experiment specification")

    # 4. Instantiate model and load weights
    console.print(f"[cyan]Loading model weights from {model_file}...[/]")
    vocab_size = 0
    if (run / "tokenizer_metadata.json").is_file():
        tok_m = json.loads((run / "tokenizer_metadata.json").read_text(encoding="utf-8"))
        vocab_size = int(tok_m.get("vocab_size", 0))
    elif tokenizer_metadata_path(Path("artifacts/tokenizers"), tok_type).is_file():
        tok_m = json.loads(
            tokenizer_metadata_path(Path("artifacts/tokenizers"), tok_type).read_text(
                encoding="utf-8"
            )
        )
        vocab_size = int(tok_m.get("vocab_size", 0))

    if vocab_size <= 0:
        if tok_type == "bpe":
            vocab_size = int(getattr(config.tokenizer, "bpe_vocab_size", 4096))
        else:
            from safetensors import safe_open

            with safe_open(str(model_file), framework="pt") as f:
                vocab_size = int(f.get_slice("tok_embeddings.weight").get_shape()[0])

    model_cfg = TransformerConfig.from_app_config(config, vocab_size=vocab_size)
    model = TransformerLM(model_cfg)
    load_model(model, str(model_file))
    model.to(target_device)
    model.eval()

    # 5. Load dataset chunks for requested split
    encoded_dir = data_root / "encoded" / tok_type
    chunk_file = encoded_dir / f"{split}_chunks.json"
    if not chunk_file.is_file():
        console.print(
            f"[bold red]Encoded chunks not found for split '{split}': {chunk_file}[/]\n"
            f"Please run `scripture-lm encode` first."
        )
        raise ValueError("Evaluation prerequisites are missing; see details above")

    chunks = load_chunk_index(chunk_file)
    if not chunks:
        console.print(f"[bold red]Split '{split}' contains 0 chunks in {chunk_file}[/]")
        raise ValueError("Evaluation prerequisites are missing; see details above")

    dataset = ScriptureChunkDataset(
        chunks,
        base_dir=data_root / "encoded",
        context_length=config.tokenizer.context_length,
    )
    dataloader = create_dataloader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=config.training.microbatch_size,
        drop_last=False,
    )

    console.print(
        f"[cyan]Evaluating {len(chunks):,} chunks ({split} split, {tok_type.upper()} tokenizer) "
        f"on {target_device}...[/]"
    )

    # 6. Compute evaluation metrics
    bpc_res = compute_bpc(model, dataloader, target_device, autocast_context=autocast_ctx)
    ppl_res = compute_perplexity(model, dataloader, target_device, autocast_context=autocast_ctx)

    # 7. Collect Provenance Metadata
    corpus_fp = ""
    norm_fp = ""
    split_hash = ""
    tok_hash = ""

    lock_path = run / "corpus_lock.json"
    if not lock_path.is_file():
        lock_path = data_root / "corpus_lock.json"
    if lock_path.is_file():
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        corpus_fp = lock_data.get("corpus_fingerprint", "")
        norm_fp = lock_data.get("normalization_fingerprint", "")

    split_path = run / "split_manifest.json"
    if not split_path.is_file():
        split_path = data_root / "splits" / "split_manifest.json"
    if split_path.is_file():
        split_hash = compute_file_sha256(split_path)

    tok_meta_path = run / "tokenizer_metadata.json"
    if not tok_meta_path.is_file():
        tok_meta_path = tokenizer_metadata_path(Path("artifacts/tokenizers"), tok_type)
    if tok_meta_path.is_file():
        tok_meta = json.loads(tok_meta_path.read_text(encoding="utf-8"))
        tok_hash = tok_meta.get("tokenizer_artifact_sha256", "")

    # 8. Save JSON Report
    eval_dir = run / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_file = eval_dir / f"{split}_metrics.json"

    eval_report = {
        "evaluation_schema_version": "1.0",
        "evaluation_algorithm_version": "evaluation_v1",
        "split": split,
        "checkpoint": checkpoint,
        "checkpoint_model_sha256": model_sha256,
        "corpus_fingerprint": corpus_fp,
        "normalization_fingerprint": norm_fp,
        "split_manifest_hash": split_hash,
        "tokenizer_artifact_sha256": tok_hash,
        "evaluated_characters": bpc_res.total_characters,
        "evaluated_chunks": len(chunks),
        "macro_bpc": bpc_res.macro_bpc,
        "micro_bpc": bpc_res.micro_bpc,
        "family_bpc": bpc_res.family_bpc,
        "total_bits": bpc_res.total_bits,
        "total_non_special_tokens": bpc_res.total_non_special_tokens,
        "token_cross_entropy": ppl_res.cross_entropy_per_token,
        "token_perplexity": ppl_res.perplexity,
        "family_token_perplexity": ppl_res.family_perplexity,
        "total_valid_targets": ppl_res.total_valid_targets,
    }

    report_file.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    console.print(f"[green]Saved evaluation report to {report_file}[/]")

    if generate:
        console.print("[bold cyan]Running canonical generation benchmark suite (standard_v1)...[/]")
        from scripture_lm.evaluation.generation_suite import (
            GenerationSample,
            GenerationSettings,
            get_canonical_generation_suite,
            sample_from_result,
            save_generation_results,
        )
        from scripture_lm.evaluation.repetition import analyze_generation_repetition
        from scripture_lm.generation.generate import TextGenerator

        tok_dir = Path("artifacts/tokenizers")
        tokenizer_inst: BaseTokenizer
        if tok_type == "bpe":
            bpe_path = run / "bpe.json" if (run / "bpe.json").is_file() else tok_dir / "bpe.json"
            if compute_file_sha256(bpe_path) != tok_hash:
                raise ValueError("Generation tokenizer differs from evaluation provenance")
            tokenizer_inst = BPETokenizer.load(bpe_path)
        else:
            char_path = (
                run / "char_vocab.json"
                if (run / "char_vocab.json").is_file()
                else tok_dir / "char_vocab.json"
            )
            if compute_file_sha256(char_path) != tok_hash:
                raise ValueError("Generation tokenizer differs from evaluation provenance")
            tokenizer_inst = CharacterTokenizer.load(char_path)

        suite = get_canonical_generation_suite()
        text_gen = TextGenerator(model, tokenizer_inst, device=target_device)
        samples: list[GenerationSample] = []
        for prompt_def in suite.prompts:
            for seed_val in suite.seeds:
                gen_settings = GenerationSettings(
                    temperature=suite.canonical_settings.temperature,
                    top_p=suite.canonical_settings.top_p,
                    top_k=suite.canonical_settings.top_k,
                    max_new_tokens=suite.canonical_settings.max_new_tokens,
                    max_new_characters=suite.canonical_settings.max_new_characters,
                    seed=seed_val,
                )
                sample_id = f"{prompt_def.prompt_id}_s{seed_val}"
                res = text_gen.generate(prompt_def.prompt_text, gen_settings)
                sample = sample_from_result(
                    sample_id=sample_id,
                    result=res,
                    settings=gen_settings,
                    family=prompt_def.family,
                    suite_id=suite.suite_id,
                )
                samples.append(sample)

        gen_dir = run / "generations"
        gen_dir.mkdir(parents=True, exist_ok=True)
        save_generation_results(gen_dir / "samples.json", samples)
        console.print(
            f"[green]Saved {len(samples)} generation samples to {gen_dir / 'samples.json'}[/]"
        )

        rep_report = analyze_generation_repetition(samples)
        (gen_dir / "repetition_report.json").write_text(
            json.dumps(rep_report.model_dump(), indent=2), encoding="utf-8"
        )
        console.print(
            f"[dim]Repetition: distinct-1={rep_report.mean_distinct_1:.3f}, "
            f"distinct-4={rep_report.mean_distinct_4:.3f}, "
            f"cycles={rep_report.samples_with_degenerate_cycle}[/]"
        )

        from scripture_lm.corpus.normalize import CorpusLock
        from scripture_lm.corpus.split import SplitManifest
        from scripture_lm.evaluation.memorization import TrainingCorpusMatcher
        from scripture_lm.tokenization.base import verify_corpus_and_split_integrity

        lock = CorpusLock.model_validate_json(
            (data_root / "corpus_lock.json").read_text(encoding="utf-8")
        )
        split_manifest = SplitManifest.model_validate_json(
            (data_root / "splits" / "split_manifest.json").read_text(encoding="utf-8")
        )
        verify_corpus_and_split_integrity(split_manifest, lock, data_root / "normalized")
        matcher = TrainingCorpusMatcher.from_corpus_and_split(data_root)
        memo_report = matcher.analyze_generation_samples(samples)
        memo_path = gen_dir / "memorization_report.json"
        memo_path.write_text(memo_report.model_dump_json(indent=2), encoding="utf-8")
        provenance = {
            "suite": suite.model_dump(mode="json"),
            "checkpoint_model_sha256": model_sha256,
            "tokenizer_artifact_sha256": tok_hash,
            "corpus_fingerprint": corpus_fp,
            "split_manifest_hash": split_hash,
            "files": {
                name: compute_file_sha256(gen_dir / name)
                for name in ("samples.json", "repetition_report.json", "memorization_report.json")
            },
        }
        (gen_dir / "benchmark_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

    # 9. Render Summary Table
    table = Table(title=f"Evaluation Results: {run.name} ({split.title()} Split)")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="bold green")

    table.add_row("Macro BPC (cross-tokenizer)", f"{bpc_res.macro_bpc:.4f}")
    table.add_row("Micro BPC", f"{bpc_res.micro_bpc:.4f}")
    for fam, bpc in bpc_res.family_bpc.items():
        table.add_row(f"  {fam} BPC", f"{bpc:.4f}")
    table.add_row("Token Perplexity", f"{ppl_res.perplexity:.2f}")
    table.add_row("Token Cross-Entropy (nats)", f"{ppl_res.cross_entropy_per_token:.4f}")
    table.add_row("Evaluated Characters", f"{bpc_res.total_characters:,}")
    table.add_row("Evaluated Target Tokens", f"{ppl_res.total_valid_targets:,}")
    console.print(table)
