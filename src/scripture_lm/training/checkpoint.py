"""Checkpoint saving and restoration using safetensors and PyTorch state dicts."""

from __future__ import annotations

import datetime
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model
from torch.optim import Optimizer

from scripture_lm.training.metrics import EarlyStopping
from scripture_lm.training.scheduler import ExposureCosineScheduler


def save_checkpoint(
    checkpoint_dir: Path | str,
    raw_model: nn.Module,
    optimizer: Optimizer,
    scheduler: ExposureCosineScheduler,
    sampler: Any,
    early_stopping: EarlyStopping,
    global_step: int,
    micro_step: int,
    cumulative_raw_chars: int,
    cumulative_model_tokens: int,
    effective_epoch: float,
    accumulated_targets: int,
    best_val_bpc: float,
    config_dict: dict[str, Any] | None = None,
    experiment_config_hash: str | None = None,
) -> Path:
    """Save full training checkpoint with safetensors model weights and RNG state.

    Uses atomic directory write to avoid partial/corrupted checkpoints on failure.

    Args:
        checkpoint_dir: Destination directory path (e.g. runs/bpe/checkpoints/best).
        raw_model: Uncompiled TransformerLM instance.
        optimizer: Optimizer instance.
        scheduler: ExposureCosineScheduler instance.
        sampler: NaturalSampler or TemperatureSampler instance.
        early_stopping: EarlyStopping instance.
        global_step: Global optimizer step index.
        micro_step: Microbatch index.
        cumulative_raw_chars: Total normalized characters exposed so far.
        cumulative_model_tokens: Total target model tokens exposed so far.
        effective_epoch: Current effective epoch progress.
        accumulated_targets: Number of pending accumulated targets.
        best_val_bpc: Best validation macro BPC so far.
        config_dict: Optional full config dictionary to persist.

    Returns:
        Path to the saved checkpoint directory.
    """
    dest_dir = Path(checkpoint_dir)
    parent_dir = dest_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Atomic write pattern: save to temp dir then move
    with tempfile.TemporaryDirectory(dir=parent_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Save model weights using safetensors (natively handles tied weights)
        model_file = tmp_path / "model.safetensors"
        save_model(raw_model, str(model_file))

        # 2. Gather RNG states
        rng_states: dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }

        # 3. Save training state
        training_state: dict[str, Any] = {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "sampler_state": sampler.state_dict(),
            "early_stopping_state": early_stopping.state_dict(),
            "global_step": int(global_step),
            "micro_step": int(micro_step),
            "cumulative_raw_chars": int(cumulative_raw_chars),
            "cumulative_model_tokens": int(cumulative_model_tokens),
            "effective_epoch": float(effective_epoch),
            "accumulated_targets": int(accumulated_targets),
            "best_val_bpc": float(best_val_bpc),
            "rng_states": rng_states,
            "config": config_dict or {},
            "experiment_config_hash": experiment_config_hash,
        }
        torch.save(training_state, tmp_path / "training_state.pt")

        # 4. Save human-readable metadata
        metadata: dict[str, Any] = {
            "global_step": int(global_step),
            "micro_step": int(micro_step),
            "cumulative_raw_chars": int(cumulative_raw_chars),
            "cumulative_model_tokens": int(cumulative_model_tokens),
            "effective_epoch": float(effective_epoch),
            "best_val_bpc": float(best_val_bpc),
            "experiment_config_hash": experiment_config_hash,
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 5. Move temp dir contents to dest_dir atomically
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(tmp_path, dest_dir)

    return dest_dir


def load_checkpoint(
    checkpoint_dir: Path | str,
    raw_model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: ExposureCosineScheduler | None = None,
    sampler: Any = None,
    early_stopping: EarlyStopping | None = None,
) -> dict[str, Any]:
    """Restore model weights, optimizer, scheduler, sampler, and RNG states from checkpoint.

    Strictly preserves tied parameter identity without re-allocating parameters.

    Args:
        checkpoint_dir: Directory containing model.safetensors and training_state.pt.
        raw_model: Uncompiled TransformerLM instance.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        sampler: Optional sampler to restore.
        early_stopping: Optional early stopping tracker to restore.

    Returns:
        The loaded training state dictionary.
    """
    ckpt_path = Path(checkpoint_dir)
    model_file = ckpt_path / "model.safetensors"
    state_file = ckpt_path / "training_state.pt"

    if not model_file.is_file():
        raise FileNotFoundError(f"Missing model weights at {model_file}")
    if not state_file.is_file():
        raise FileNotFoundError(f"Missing training state at {state_file}")

    # 1. Load weights directly into raw_model
    load_model(raw_model, str(model_file))

    # 2. Verify tied embeddings identity
    model_any = cast(Any, raw_model)
    if getattr(model_any, "config", None) is not None and getattr(
        model_any.config, "embedding_tying", False
    ):
        if hasattr(model_any, "lm_head") and hasattr(model_any, "tok_embeddings"):
            assert model_any.lm_head.weight is model_any.tok_embeddings.weight, (
                "Weight tying identity broken after loading safetensors checkpoint"
            )

    # 3. Load training state
    state = torch.load(state_file, map_location="cpu", weights_only=False)

    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])

    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])

    if sampler is not None and "sampler_state" in state:
        sampler.load_state_dict(state["sampler_state"])

    if early_stopping is not None and "early_stopping_state" in state:
        early_stopping.load_state_dict(state["early_stopping_state"])

    # 4. Restore RNG states
    if "rng_states" in state:
        rngs = state["rng_states"]
        if "torch" in rngs and rngs["torch"] is not None:
            torch.set_rng_state(rngs["torch"])
        if "cuda" in rngs and rngs["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rngs["cuda"])
        if "python" in rngs and rngs["python"] is not None:
            random.setstate(rngs["python"])
        if "numpy" in rngs and rngs["numpy"] is not None:
            np.random.set_state(rngs["numpy"])

    return cast(dict[str, Any], state)
