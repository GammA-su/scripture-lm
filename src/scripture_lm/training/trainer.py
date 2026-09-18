"""Single-GPU training engine for Scripture-LM with target-weighted gradient accumulation.

Builds a robust trainer intended for RTX 4090 under strict closed-world conditions.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from scripture_lm.config import ScriptureLMConfig
from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.corpus.normalize import CorpusLock
from scripture_lm.corpus.split import SplitManifest
from scripture_lm.data.batching import create_dataloader
from scripture_lm.data.chunk_index import (
    EncodingProvenance,
    encoding_provenance_path,
    load_chunk_index,
)
from scripture_lm.data.dataset import ScriptureChunkDataset
from scripture_lm.data.sampler import NaturalSampler, SequentialSampler, TemperatureSampler
from scripture_lm.experiments.matrix import config_hash, scientific_config
from scripture_lm.experiments.storage import (
    atomic_json,
    prepare_run,
    read_status,
    timestamp,
    update_status,
)
from scripture_lm.experiments.storage import serialize_toml_dict as serialize_toml_dict
from scripture_lm.model.config import TransformerConfig
from scripture_lm.model.transformer import TransformerLM
from scripture_lm.tokenization.base import compute_manifest_sha256, tokenizer_metadata_path
from scripture_lm.tokenization.character import CharacterTokenizer
from scripture_lm.training.checkpoint import load_checkpoint, save_checkpoint
from scripture_lm.training.metrics import (
    EarlyStopping,
    MetricsLogger,
    ValidationMetrics,
    compute_validation_metrics,
)
from scripture_lm.training.optimizer import clip_gradients, configure_optimizer
from scripture_lm.training.scheduler import ExposureCosineScheduler

console = Console()


def verify_encoding_provenance(
    config: ScriptureLMConfig,
    data_root: Path = Path("data"),
    corpus_root: Path = Path("corpus"),
    artifacts_root: Path = Path("artifacts"),
) -> tuple[CorpusLock, SplitManifest, EncodingProvenance, int]:
    """Verify that encoded dataset matches current corpus lock, split, and tokenizer.

    Fails fast before creating run directory if any provenance mismatch is detected.

    Returns:
        tuple of (corpus_lock, split_manifest, encoding_provenance, total_natural_training_chars).
    """
    tok_type = config.tokenizer.type
    lock_file = data_root / "corpus_lock.json"
    split_file = data_root / "splits" / "split_manifest.json"
    encoded_dir = data_root / "encoded" / tok_type
    encoding_meta_file = encoding_provenance_path(encoded_dir)
    tok_meta_file = tokenizer_metadata_path(artifacts_root / "tokenizers", tok_type)
    tok_artifact = (
        artifacts_root / "tokenizers" / ("bpe.json" if tok_type == "bpe" else "char_vocab.json")
    )

    if not lock_file.is_file():
        raise FileNotFoundError(
            f"Corpus lock not found at {lock_file}. Run 'scripture-lm corpus prepare' first."
        )
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at {split_file}. Run 'scripture-lm corpus prepare' first."
        )
    if not encoding_meta_file.is_file():
        raise FileNotFoundError(
            f"Encoded dataset metadata not found at {encoding_meta_file}. "
            f"Run 'scripture-lm encode --tokenizer {tok_type}' first."
        )
    if not tok_meta_file.is_file() or not tok_artifact.is_file():
        raise FileNotFoundError(
            "Tokenizer artifacts not found. Run 'scripture-lm tokenizer train' first."
        )

    corpus_lock = CorpusLock.model_validate_json(lock_file.read_text(encoding="utf-8"))
    split_manifest = SplitManifest.model_validate_json(split_file.read_text(encoding="utf-8"))
    enc_prov = EncodingProvenance.model_validate_json(
        encoding_meta_file.read_text(encoding="utf-8")
    )

    # 1. Corpus fingerprints
    if enc_prov.corpus_fingerprint != corpus_lock.corpus_fingerprint:
        raise ValueError(
            f"Encoded dataset corpus_fingerprint ({enc_prov.corpus_fingerprint}) does not match "
            f"corpus_lock.json ({corpus_lock.corpus_fingerprint}). Re-run scripture-lm encode."
        )
    if enc_prov.normalization_fingerprint != corpus_lock.normalization_fingerprint:
        raise ValueError(
            "Encoded dataset normalization_fingerprint mismatch. Re-run scripture-lm encode."
        )

    # 2. Split hash
    current_split_hash = compute_manifest_sha256(split_file)
    if enc_prov.split_manifest_hash != current_split_hash:
        raise ValueError(
            "Encoded dataset was generated from a different split manifest. "
            "Re-run scripture-lm encode."
        )

    # 3. Tokenizer artifact SHA256
    current_tok_hash = compute_file_sha256(tok_artifact)
    if enc_prov.tokenizer_artifact_sha256 != current_tok_hash:
        raise ValueError(
            f"Encoded dataset tokenizer artifact hash mismatch "
            f"({enc_prov.tokenizer_artifact_sha256} vs {current_tok_hash}). "
            f"Re-run scripture-lm encode --tokenizer {tok_type}."
        )

    # 4. Context length and tokenizer type
    if enc_prov.tokenizer_type != tok_type:
        raise ValueError(
            f"Configured tokenizer type ({tok_type}) does not match "
            f"encoded dataset ({enc_prov.tokenizer_type})."
        )
    if enc_prov.context_length != config.tokenizer.context_length:
        raise ValueError(
            f"Configured context_length ({config.tokenizer.context_length}) does not match "
            f"encoded dataset ({enc_prov.context_length}). Re-run scripture-lm encode."
        )

    # Total natural training characters N
    doc_map = {doc.document_id: doc for doc in corpus_lock.documents}
    n_train = sum(
        doc_map[doc_id].normalized_characters
        for doc_id in split_manifest.train
        if doc_id in doc_map
    )

    return corpus_lock, split_manifest, enc_prov, n_train


class Trainer:
    """Single-GPU training engine with exposure-based scheduling and target accumulation."""

    def __init__(
        self,
        config: ScriptureLMConfig,
        run_dir: Path | str | None = None,
        data_root: Path = Path("data"),
        corpus_root: Path = Path("corpus"),
        artifacts_root: Path = Path("artifacts"),
        resume_checkpoint_dir: Path | str | None = None,
    ) -> None:
        """Initialize trainer, datasets, model, and metadata."""
        self.config = config
        self.data_root = Path(data_root)
        self.corpus_root = Path(corpus_root)
        self.artifacts_root = Path(artifacts_root)

        # 1. Verify provenance before creating any run files
        self.corpus_lock, self.split_manifest, self.enc_prov, self.N = verify_encoding_provenance(
            config, self.data_root, self.corpus_root, self.artifacts_root
        )

        # 2. Setup run directory
        tok_type = config.tokenizer.type
        sampling_mode = config.data.sampling_mode
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            self.run_dir = Path("runs") / f"{tok_type}_{sampling_mode}"

        self.experiment_spec = prepare_run(
            self.run_dir,
            config,
            self.enc_prov.model_dump(mode="json"),
            resume=resume_checkpoint_dir is not None,
        )
        if resume_checkpoint_dir is None:
            update_status(self.run_dir, "planned")
        else:
            if Path(resume_checkpoint_dir).resolve().parent.parent != self.run_dir.resolve():
                raise ValueError("Resume checkpoint must belong to this run directory")
            saved_state = torch.load(
                Path(resume_checkpoint_dir) / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            if saved_state.get("experiment_config_hash") != config_hash(self.experiment_spec):
                raise ValueError("Resume checkpoint experiment provenance mismatch")
            saved_config = ScriptureLMConfig.model_validate(saved_state["config"])
            if scientific_config(saved_config) != scientific_config(config):
                raise ValueError("Resume checkpoint scientific configuration mismatch")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_dir = self.checkpoints_dir / "best"
        self.latest_checkpoint_dir = self.checkpoints_dir / "latest"
        (self.run_dir / "generations").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "evaluation").mkdir(parents=True, exist_ok=True)

        # Re-seed each fresh experiment, independent of earlier matrix runs.
        random.seed(config.training.seed)
        np.random.seed(config.training.seed % (2**32))
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)

        # 3. Setup device
        requested_device = config.training.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            console.print("[yellow]CUDA requested but unavailable. Falling back to CPU.[/yellow]")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(requested_device)

        # 4. Setup precision and autocast
        self.precision = config.training.precision
        self.scaler: torch.amp.GradScaler | None = None
        self.autocast_context: Any = contextlib.nullcontext()

        if self.device.type == "cuda":
            if self.precision == "bf16":
                if torch.cuda.is_bf16_supported():
                    self.autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                else:
                    console.print(
                        "[yellow]BF16 unsupported on CUDA device. Falling back to FP32.[/yellow]"
                    )
                    self.precision = "fp32"
                    self.autocast_context = contextlib.nullcontext()
            elif self.precision == "fp16":
                self.autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
                self.scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
            else:
                self.autocast_context = contextlib.nullcontext()
        else:
            self.precision = "fp32"
            self.autocast_context = contextlib.nullcontext()

        # 5. Load chunk datasets
        encoded_dir = self.data_root / "encoded" / tok_type
        train_index_file = encoded_dir / "train_chunks.json"
        val_index_file = encoded_dir / "validation_chunks.json"

        if not train_index_file.is_file():
            raise FileNotFoundError(f"Missing train chunk index: {train_index_file}")
        if not val_index_file.is_file():
            raise FileNotFoundError(f"Missing validation chunk index: {val_index_file}")

        self.train_chunks = load_chunk_index(train_index_file)
        self.val_chunks = load_chunk_index(val_index_file)

        self.train_dataset = ScriptureChunkDataset(
            self.train_chunks,
            base_dir=self.data_root / "encoded",
            context_length=config.tokenizer.context_length,
        )
        self.val_dataset = ScriptureChunkDataset(
            self.val_chunks,
            base_dir=self.data_root / "encoded",
            context_length=config.tokenizer.context_length,
        )

        # 6. Setup samplers and loaders
        if sampling_mode == "natural":
            self.train_sampler = NaturalSampler(
                self.train_chunks,
                seed=config.training.seed,
            )
        elif sampling_mode == "temperature":
            self.train_sampler = TemperatureSampler(  # type: ignore[assignment]
                self.train_chunks,
                alpha=config.data.sampling_alpha,
                seed=config.training.seed,
            )
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        self.val_sampler = SequentialSampler(self.val_chunks)

        self.train_loader = create_dataloader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=config.training.microbatch_size,
            drop_last=False,
            num_workers=0,
        )
        self.val_loader = create_dataloader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=config.training.microbatch_size,
            drop_last=False,
            num_workers=0,
        )

        # 7. Model instantiation
        if hasattr(config.tokenizer, "bpe_vocab_size"):
            vocab_size = getattr(config.tokenizer, "bpe_vocab_size")
        else:
            char_vocab_file = self.artifacts_root / "tokenizers" / "char_vocab.json"
            vocab_size = CharacterTokenizer.load(char_vocab_file).vocab_size

        self.model_config = TransformerConfig.from_app_config(config, vocab_size=vocab_size)
        self.raw_model = TransformerLM(self.model_config).to(self.device)

        # Assert tied weights identity
        if self.model_config.embedding_tying:
            assert self.raw_model.lm_head.weight is self.raw_model.tok_embeddings.weight, (
                "Embedding tying failed on model initialization"
            )

        # 8. Compilation management
        self.compile_requested = config.training.compile
        self.compile_active = False
        self.compile_failure_reason: str | None = None
        self.train_model: nn.Module = self.raw_model

        if self.compile_requested and self.device.type == "cuda":
            try:
                self.train_model = cast(nn.Module, torch.compile(self.raw_model))
                self.compile_active = True
            except Exception as e:
                self.compile_failure_reason = str(e)
                self.compile_active = False
                self.train_model = self.raw_model

        # 9. Optimizer and Scheduler setup
        self.optimizer = configure_optimizer(self.raw_model, config.training)

        self.total_target_exposure = self.N * config.training.max_effective_epochs
        self.scheduler = ExposureCosineScheduler(
            self.optimizer,
            total_target_exposure=self.total_target_exposure,
            warmup_ratio=config.training.warmup_ratio,
            min_lr=config.training.min_learning_rate,
            base_lr=config.training.learning_rate,
        )

        # 10. Metrics, Early Stopping, and Logging
        self.early_stopping = EarlyStopping(patience=config.training.early_stopping_patience)
        self.metrics_logger = MetricsLogger(self.run_dir)

        # 11. State counters
        self.global_step = 0
        self.micro_step = 0
        self.cumulative_raw_chars = 0
        self.cumulative_model_tokens = 0
        self.best_val_bpc = float("inf")

        # Evaluation frequency: 4 times per effective epoch -> every 0.25 * N
        self.eval_interval_chars = max(1, round(self.N / config.training.eval_frequency_per_epoch))
        self.next_eval_chars = self.eval_interval_chars

        # Resume if requested
        if resume_checkpoint_dir is not None:
            self._resume(resume_checkpoint_dir)

        # 12. Snapshot metadata files
        self._write_run_snapshots()

    def _write_run_snapshots(self) -> None:
        """Snapshot all configuration, environment, and provenance files into run directory."""
        # config.toml
        # Scientific and initial resolved configurations were written once by prepare_run.

        # environment.json
        git_commit = "unknown"
        git_branch = "unknown"
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        env_info = {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "os": platform.platform(),
            "hostname": platform.node(),
            "device_requested": self.config.training.device,
            "device_active": str(self.device),
            "seed": self.config.training.seed,
            "sampling_mode": self.config.data.sampling_mode,
            "sampling_alpha": self.config.data.sampling_alpha,
            "compile_requested": self.compile_requested,
            "compile_active": self.compile_active,
            "compile_failure_reason": self.compile_failure_reason,
            "model_parameters_total": self.raw_model.count_parameters(trainable_only=False),
            "model_parameters_trainable": self.raw_model.count_parameters(trainable_only=True),
            "tokenizer_type": self.config.tokenizer.type,
            "tokenizer_artifact_sha256": self.enc_prov.tokenizer_artifact_sha256,
            "corpus_fingerprint": self.corpus_lock.corpus_fingerprint,
            "normalization_fingerprint": self.corpus_lock.normalization_fingerprint,
            "split_manifest_hash": self.enc_prov.split_manifest_hash,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        environment_path = self.run_dir / "environment.json"
        previous_environment = (
            json.loads(environment_path.read_text(encoding="utf-8"))
            if environment_path.is_file()
            else {}
        )
        history = previous_environment.get("execution_history", [])
        history.append(dict(env_info))
        env_info["execution_history"] = history
        atomic_json(environment_path, env_info)

        sources = {
            "corpus_manifest.toml": self.corpus_root / "corpus_manifest.toml",
            "corpus_lock.json": self.data_root / "corpus_lock.json",
            "split_manifest.json": self.data_root / "splits" / "split_manifest.json",
            "tokenizer_metadata.json": tokenizer_metadata_path(
                self.artifacts_root / "tokenizers", self.config.tokenizer.type
            ),
            "encoding_metadata.json": encoding_provenance_path(
                self.data_root / "encoded" / self.config.tokenizer.type
            ),
        }
        artifact = "bpe.json" if self.config.tokenizer.type == "bpe" else "char_vocab.json"
        sources[artifact] = self.artifacts_root / "tokenizers" / artifact
        for name, source in sources.items():
            if source.is_file():
                # Preserve exact source bytes for provenance hashes (including newlines).
                destination = self.run_dir / name
                if destination.exists():
                    if destination.read_bytes() != source.read_bytes():
                        raise ValueError(f"Immutable provenance snapshot differs: {destination}")
                else:
                    with destination.open("xb") as handle:
                        handle.write(source.read_bytes())

    def _resume(self, checkpoint_dir: Path | str) -> None:
        """Restore all state from an existing checkpoint directory."""
        console.print(f"[bold cyan]Resuming training from checkpoint:[/] {checkpoint_dir}")
        state = load_checkpoint(
            checkpoint_dir,
            raw_model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=self.train_sampler,
            early_stopping=self.early_stopping,
        )

        self.global_step = int(state.get("global_step", 0))
        self.micro_step = int(state.get("micro_step", 0))
        self.cumulative_raw_chars = int(state.get("cumulative_raw_chars", 0))
        self.cumulative_model_tokens = int(state.get("cumulative_model_tokens", 0))
        self.best_val_bpc = float(state.get("best_val_bpc", float("inf")))

        # Advance next evaluation threshold past resumed exposure
        while self.next_eval_chars <= self.cumulative_raw_chars:
            self.next_eval_chars += self.eval_interval_chars

    def evaluate(self) -> ValidationMetrics:
        """Run validation evaluation and calculate macro BPC."""
        metrics = compute_validation_metrics(
            model=self.raw_model,
            val_loader=self.val_loader,
            device=self.device,
            autocast_context=self.autocast_context,
        )
        effective_epoch = self.cumulative_raw_chars / self.N if self.N > 0 else 0.0
        self.metrics_logger.log_validation(
            global_step=self.global_step,
            cumulative_raw_chars=self.cumulative_raw_chars,
            effective_epoch=effective_epoch,
            metrics=metrics,
        )
        return metrics

    def _execute_optimizer_step(
        self,
        accumulated_targets: int,
        accumulated_loss_sum: float,
        window_raw_chars: int,
        window_target_tokens: int,
    ) -> tuple[float, float, float]:
        """Perform normalized gradient update, clipping, exposure accounting, and scheduler step."""
        # 1. Normalize gradients by total valid target count across accumulation window
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

            if accumulated_targets > 0:
                for p in self.raw_model.parameters():
                    if p.grad is not None:
                        p.grad.div_(accumulated_targets)

            grad_norm = clip_gradients(self.raw_model, max_norm=self.config.training.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if accumulated_targets > 0:
                for p in self.raw_model.parameters():
                    if p.grad is not None:
                        p.grad.div_(accumulated_targets)

            grad_norm = clip_gradients(self.raw_model, max_norm=self.config.training.gradient_clip)
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        # 2. Update step and exposure metrics
        self.global_step += 1
        self.cumulative_raw_chars += window_raw_chars
        self.cumulative_model_tokens += window_target_tokens

        effective_epoch = self.cumulative_raw_chars / self.N if self.N > 0 else 0.0
        step_loss = accumulated_loss_sum / accumulated_targets if accumulated_targets > 0 else 0.0

        # 3. Step exposure-based scheduler
        active_lr = self.scheduler.step(self.cumulative_raw_chars)

        # 4. Log training step
        self.metrics_logger.log_train_step(
            global_step=self.global_step,
            loss=step_loss,
            lr=active_lr,
            grad_norm=grad_norm,
            cumulative_raw_chars=self.cumulative_raw_chars,
            cumulative_model_tokens=self.cumulative_model_tokens,
            effective_epoch=effective_epoch,
        )

        return step_loss, active_lr, grad_norm

    def train(self) -> dict[str, Any]:
        """Track execution status and clean up writers on every exit path."""
        previous = read_status(self.run_dir)
        history = previous.get("resume_history", [])
        if previous["status"] != "planned":
            history.append({"resumed_at": timestamp(), "previous_status": previous["status"]})
        update_status(self.run_dir, "running", started_at=timestamp(), resume_history=history)
        try:
            summary = self._train()
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            update_status(
                self.run_dir,
                status,
                error=str(exc),
                last_checkpoint=(
                    "checkpoints/latest" if self.latest_checkpoint_dir.is_dir() else None
                ),
                effective_epochs_completed=self.cumulative_raw_chars / self.N if self.N else 0,
            )
            raise
        else:
            update_status(
                self.run_dir,
                "completed",
                completed_at=timestamp(),
                best_checkpoint=("checkpoints/best" if self.best_checkpoint_dir.is_dir() else None),
                last_checkpoint="checkpoints/latest",
                final_raw_exposure=self.cumulative_raw_chars,
                effective_epochs_completed=summary["effective_epoch"],
                completion_reason=(
                    "early_stopping"
                    if self.config.training.early_stopping_enabled
                    and self.early_stopping.should_stop
                    else "exposure_budget"
                ),
                error=None,
            )
            return summary
        finally:
            self.metrics_logger.close()

    def _train(self) -> dict[str, Any]:
        """Execute full training loop until total target exposure or early stopping."""
        console.print(
            f"[bold green]Starting Scripture-LM Training[/]\n"
            f"  Tokenizer: [cyan]{self.config.tokenizer.type}[/]\n"
            f"  Sampling:  [cyan]{self.config.data.sampling_mode}[/]\n"
            f"  Max Effective Epochs: [cyan]{self.config.training.max_effective_epochs}[/]\n"
            f"  Target Exposure:      [cyan]{self.total_target_exposure:,} characters[/]\n"
            f"  Device:    [cyan]{self.device}[/] (precision: {self.precision})\n"
        )

        start_time = time.time()
        grad_accum_steps = self.config.training.gradient_accumulation_steps

        # Pending accumulation buffers across microbatches
        accumulated_targets = 0
        accumulated_loss_sum = 0.0
        microbatches_in_window = 0
        window_raw_chars = 0
        window_target_tokens = 0

        latest_val_bpc: float | None = None

        self.raw_model.train()

        while self.cumulative_raw_chars < self.total_target_exposure and not (
            self.config.training.early_stopping_enabled and self.early_stopping.should_stop
        ):
            for batch in self.train_loader:
                self.micro_step += 1

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                target_ids = batch["target_ids"].to(self.device, non_blocking=True)
                b_raw_chars = int(batch["raw_characters"])
                b_target_tokens = int(batch["target_tokens"])

                # Forward pass in autocast
                try:
                    with self.autocast_context:
                        out = self.train_model(input_ids)
                        logits_flat = out.logits.view(-1, out.logits.shape[-1]).float()
                        targets_flat = target_ids.view(-1)
                        # Summed NLL across all valid targets
                        loss_sum = F.cross_entropy(
                            logits_flat, targets_flat, reduction="sum", ignore_index=-100
                        )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    err_msg = str(e).lower()
                    if "out of memory" in err_msg:
                        ctx = self.config.tokenizer.context_length
                        bs = self.config.training.microbatch_size
                        raise RuntimeError(
                            f"CUDA Out Of Memory encountered during training "
                            f"(context length: {ctx}, microbatch_size: {bs}, "
                            f"precision: {self.precision}). "
                            "Please reduce microbatch_size or check GPU memory headroom."
                        ) from e
                    if self.compile_active and (
                        "inductor" in err_msg
                        or "cl is not found" in err_msg
                        or "compile" in err_msg
                    ):
                        console.print(
                            f"[yellow]Compilation failed during execution ({e}). "
                            "Falling back to uncompiled model.[/yellow]"
                        )
                        self.train_model = self.raw_model
                        self.compile_active = False
                        self.compile_failure_reason = str(e)
                        # Retry uncompiled
                        with self.autocast_context:
                            out = self.train_model(input_ids)
                            logits_flat = out.logits.view(-1, out.logits.shape[-1]).float()
                            targets_flat = target_ids.view(-1)
                            loss_sum = F.cross_entropy(
                                logits_flat, targets_flat, reduction="sum", ignore_index=-100
                            )
                    else:
                        raise

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss_sum).backward()  # type: ignore[no-untyped-call]
                else:
                    loss_sum.backward()  # type: ignore[no-untyped-call]

                # Accumulate targets, loss, and exposure for this window
                valid_count = int((target_ids != -100).sum().item())
                accumulated_targets += valid_count
                accumulated_loss_sum += float(loss_sum.item())
                window_raw_chars += b_raw_chars
                window_target_tokens += b_target_tokens
                microbatches_in_window += 1

                # Step optimizer when accumulation window is complete
                if microbatches_in_window >= grad_accum_steps:
                    step_loss, active_lr, grad_norm = self._execute_optimizer_step(
                        accumulated_targets=accumulated_targets,
                        accumulated_loss_sum=accumulated_loss_sum,
                        window_raw_chars=window_raw_chars,
                        window_target_tokens=window_target_tokens,
                    )

                    # Reset accumulation window
                    accumulated_targets = 0
                    accumulated_loss_sum = 0.0
                    microbatches_in_window = 0
                    window_raw_chars = 0
                    window_target_tokens = 0

                    effective_epoch = self.cumulative_raw_chars / self.N

                    # Console progress update
                    if self.global_step % 10 == 0 or self.global_step == 1:
                        bpc_str = f"{latest_val_bpc:.4f}" if latest_val_bpc is not None else "N/A"
                        max_ep = self.config.training.max_effective_epochs
                        console.print(
                            f"Step {self.global_step:5d} | "
                            f"Epoch {effective_epoch:5.2f}/{max_ep} | "
                            f"Loss {step_loss:6.4f} | "
                            f"LR {active_lr:.2e} | "
                            f"Chars {self.cumulative_raw_chars:10,d} | "
                            f"Tokens {self.cumulative_model_tokens:8,d} | "
                            f"Val BPC {bpc_str}"
                        )

                    # Evaluation checkpoint trigger based on raw character exposure
                    if self.cumulative_raw_chars >= self.next_eval_chars:
                        console.print(
                            f"\n[bold yellow]Triggering evaluation at "
                            f"{self.cumulative_raw_chars:,} chars "
                            f"(threshold {self.next_eval_chars:,})[/]"
                        )
                        val_metrics = self.evaluate()
                        latest_val_bpc = val_metrics.macro_val_bpc
                        self.raw_model.train()

                        console.print(
                            f"  [bold]Validation Loss:[/] {val_metrics.val_loss:.4f} | "
                            f"[bold]PPL:[/] {val_metrics.val_perplexity:.2f} | "
                            f"[bold]Macro BPC:[/] {val_metrics.macro_val_bpc:.4f} | "
                            f"[bold]Micro BPC:[/] {val_metrics.micro_val_bpc:.4f}"
                        )

                        # Advance next evaluation threshold
                        while self.next_eval_chars <= self.cumulative_raw_chars:
                            self.next_eval_chars += self.eval_interval_chars

                        # Update stopping state before serializing the resume checkpoint.
                        is_best = self.early_stopping.step(val_metrics.macro_val_bpc)
                        if is_best:
                            self.best_val_bpc = val_metrics.macro_val_bpc

                        # Save latest checkpoint
                        save_checkpoint(
                            checkpoint_dir=self.latest_checkpoint_dir,
                            raw_model=self.raw_model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            sampler=self.train_sampler,
                            early_stopping=self.early_stopping,
                            global_step=self.global_step,
                            micro_step=self.micro_step,
                            cumulative_raw_chars=self.cumulative_raw_chars,
                            cumulative_model_tokens=self.cumulative_model_tokens,
                            effective_epoch=effective_epoch,
                            accumulated_targets=accumulated_targets,
                            best_val_bpc=self.best_val_bpc,
                            config_dict=self.config.model_dump(),
                            experiment_config_hash=config_hash(self.experiment_spec),
                        )

                        update_status(
                            self.run_dir,
                            "running",
                            last_checkpoint="checkpoints/latest",
                            effective_epochs_completed=effective_epoch,
                        )

                        # Check if new best achieved
                        if is_best:
                            console.print(
                                f"  [bold green]New best validation macro BPC: "
                                f"{self.best_val_bpc:.4f}![/] "
                                f"Saving best checkpoint."
                            )
                            save_checkpoint(
                                checkpoint_dir=self.best_checkpoint_dir,
                                raw_model=self.raw_model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                sampler=self.train_sampler,
                                early_stopping=self.early_stopping,
                                global_step=self.global_step,
                                micro_step=self.micro_step,
                                cumulative_raw_chars=self.cumulative_raw_chars,
                                cumulative_model_tokens=self.cumulative_model_tokens,
                                effective_epoch=effective_epoch,
                                accumulated_targets=accumulated_targets,
                                best_val_bpc=self.best_val_bpc,
                                config_dict=self.config.model_dump(),
                                experiment_config_hash=config_hash(self.experiment_spec),
                            )

                        if (
                            self.config.training.early_stopping_enabled
                            and self.early_stopping.should_stop
                        ):
                            console.print(
                                f"\n[bold red]Early stopping triggered after "
                                f"{self.early_stopping.patience} "
                                f"evaluations without improvement.[/]"
                            )
                            break

                # Check if total target exposure reached
                if self.cumulative_raw_chars >= self.total_target_exposure:
                    break

            if (
                self.config.training.early_stopping_enabled and self.early_stopping.should_stop
            ) or self.cumulative_raw_chars >= self.total_target_exposure:
                break

        # Flush any partial accumulation window remaining upon training completion
        if microbatches_in_window > 0 and accumulated_targets > 0:
            console.print(
                "[cyan]Flushing final partial accumulation window at training end...[/cyan]"
            )
            self._execute_optimizer_step(
                accumulated_targets=accumulated_targets,
                accumulated_loss_sum=accumulated_loss_sum,
                window_raw_chars=window_raw_chars,
                window_target_tokens=window_target_tokens,
            )

        # Final evaluation
        final_metrics = self.evaluate()
        total_time = time.time() - start_time
        effective_epoch = self.cumulative_raw_chars / self.N if self.N > 0 else 0.0

        # Always retain a best checkpoint, including a final partial accumulation update.
        if final_metrics.macro_val_bpc < self.best_val_bpc or not self.best_checkpoint_dir.is_dir():
            self.best_val_bpc = final_metrics.macro_val_bpc
            self.early_stopping.best_metric = self.best_val_bpc
            save_checkpoint(
                checkpoint_dir=self.best_checkpoint_dir,
                raw_model=self.raw_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                sampler=self.train_sampler,
                early_stopping=self.early_stopping,
                global_step=self.global_step,
                micro_step=self.micro_step,
                cumulative_raw_chars=self.cumulative_raw_chars,
                cumulative_model_tokens=self.cumulative_model_tokens,
                effective_epoch=effective_epoch,
                accumulated_targets=0,
                best_val_bpc=self.best_val_bpc,
                config_dict=self.config.model_dump(),
                experiment_config_hash=config_hash(self.experiment_spec),
            )

        # Final checkpoint save
        save_checkpoint(
            checkpoint_dir=self.latest_checkpoint_dir,
            raw_model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=self.train_sampler,
            early_stopping=self.early_stopping,
            global_step=self.global_step,
            micro_step=self.micro_step,
            cumulative_raw_chars=self.cumulative_raw_chars,
            cumulative_model_tokens=self.cumulative_model_tokens,
            effective_epoch=effective_epoch,
            accumulated_targets=0,
            best_val_bpc=self.best_val_bpc,
            config_dict=self.config.model_dump(),
            experiment_config_hash=config_hash(self.experiment_spec),
        )

        self.metrics_logger.close()

        summary: dict[str, Any] = {
            "total_time_seconds": total_time,
            "global_steps": self.global_step,
            "effective_epoch": effective_epoch,
            "cumulative_raw_chars": self.cumulative_raw_chars,
            "cumulative_model_tokens": self.cumulative_model_tokens,
            "best_val_bpc": self.best_val_bpc,
            "final_val_loss": final_metrics.val_loss,
            "final_macro_bpc": final_metrics.macro_val_bpc,
            "final_micro_bpc": final_metrics.micro_val_bpc,
        }

        # Save run summary
        (self.run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        console.print(
            f"\n[bold green]Training Completed Successfully![/]\n"
            f"  Total Time:          {total_time:.2f}s\n"
            f"  Global Steps:        {self.global_step}\n"
            f"  Effective Epochs:    {effective_epoch:.2f}\n"
            f"  Raw Chars Exposed:   {self.cumulative_raw_chars:,}\n"
            f"  Tokens Exposed:      {self.cumulative_model_tokens:,}\n"
            f"  Best Validation BPC: {self.best_val_bpc:.4f}\n"
        )

        return summary
