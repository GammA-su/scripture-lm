"""Validation metrics, bits-per-character (BPC) calculation, logging, and early stopping."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scripture_lm.tokenization.base import BOS_ID, EOS_ID, PAD_ID


@dataclass
class ValidationMetrics:
    """Aggregated validation evaluation metrics."""

    val_loss: float
    val_perplexity: float
    macro_val_bpc: float
    micro_val_bpc: float
    family_bpc: dict[str, float]
    total_raw_characters: int
    total_valid_targets: int
    total_non_special_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_validation_metrics(
    model: nn.Module,
    val_loader: DataLoader[dict[str, Any]],
    device: torch.device | str,
    autocast_context: Any = None,
) -> ValidationMetrics:
    """Compute validation metrics with strict target-only cross-tokenizer BPC.

    BPC numerator: Sum of -log2 P(t) across all non-special text tokens (excluding PAD, BOS, EOS).
    BPC denominator: Sum of original normalized scripture characters only.
    Macro BPC: Unweighted mean of family BPCs.

    Args:
        model: TransformerLM model in eval mode.
        val_loader: DataLoader over validation chunks (SequentialSampler).
        device: Device to transfer tensors to.
        autocast_context: Optional autocast context manager (e.g. torch.autocast(...)).

    Returns:
        ValidationMetrics containing loss, perplexity, macro BPC, micro BPC, and per-family BPCs.
    """
    model.eval()

    total_nll_nats = 0.0
    total_valid_targets = 0

    family_non_special_bits: dict[str, float] = {}
    family_raw_chars: dict[str, int] = {}
    total_non_special_tokens = 0

    log2_e = 1.0 / math.log(2.0)

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            families = batch["families"]
            batch_size, seq_len = input_ids.shape

            # Forward pass
            if autocast_context is not None:
                with autocast_context:
                    out = model(input_ids)
            else:
                out = model(input_ids)

            logits = out.logits.float()
            vocab_size = logits.shape[-1]

            # Element-wise NLL for each token position
            token_nlls = F.cross_entropy(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(batch_size, seq_len)

            valid_mask = target_ids != -100
            total_nll_nats += float(token_nlls[valid_mask].sum().item())
            total_valid_targets += int(valid_mask.sum().item())

            # Non-special tokens for cross-tokenizer BPC (exclude PAD, BOS, EOS)
            non_special_mask = (
                valid_mask
                & (target_ids != PAD_ID)
                & (target_ids != BOS_ID)
                & (target_ids != EOS_ID)
            )

            chunk_non_special_nlls = (token_nlls * non_special_mask).sum(dim=1)
            chunk_non_special_bits = chunk_non_special_nlls * log2_e
            chunk_non_special_counts = non_special_mask.sum(dim=1)

            # Retrieve raw characters per chunk from dataset items or batch metadata
            # In batch collator, raw characters can be extracted per item
            for i in range(batch_size):
                fam = str(families[i])
                bits = float(chunk_non_special_bits[i].item())
                n_non_special = int(chunk_non_special_counts[i].item())

                # Recover chunk raw_character_count from batch or dataset chunk metadata
                if "raw_characters_per_chunk" in batch:
                    raw_chars = int(batch["raw_characters_per_chunk"][i])
                elif "chunk_ids" in batch and hasattr(val_loader.dataset, "chunks"):
                    c_id = batch["chunk_ids"][i]
                    if isinstance(val_loader.dataset.chunks, list) and isinstance(c_id, int):
                        raw_chars = val_loader.dataset.chunks[c_id].raw_character_count
                    else:
                        raw_chars = max(1, int(batch["raw_characters"]) // batch_size)
                else:
                    raw_chars = max(1, int(batch["raw_characters"]) // batch_size)

                family_non_special_bits[fam] = family_non_special_bits.get(fam, 0.0) + bits
                family_raw_chars[fam] = family_raw_chars.get(fam, 0) + raw_chars
                total_non_special_tokens += n_non_special

    # Overall token-weighted loss and perplexity
    overall_val_loss = (
        total_nll_nats / total_valid_targets if total_valid_targets > 0 else float("inf")
    )
    val_perplexity = math.exp(min(overall_val_loss, 100.0))

    # Per-family BPC
    family_bpc: dict[str, float] = {}
    for fam in sorted(family_non_special_bits.keys()):
        bits = family_non_special_bits[fam]
        chars = family_raw_chars.get(fam, 0)
        family_bpc[fam] = bits / chars if chars > 0 else float("inf")

    # Macro validation BPC: unweighted average of scripture family BPCs
    if family_bpc:
        macro_val_bpc = sum(family_bpc.values()) / len(family_bpc)
    else:
        macro_val_bpc = float("inf")

    total_chars = sum(family_raw_chars.values())
    total_bits = sum(family_non_special_bits.values())
    micro_val_bpc = total_bits / total_chars if total_chars > 0 else float("inf")

    return ValidationMetrics(
        val_loss=overall_val_loss,
        val_perplexity=val_perplexity,
        macro_val_bpc=macro_val_bpc,
        micro_val_bpc=micro_val_bpc,
        family_bpc=family_bpc,
        total_raw_characters=total_chars,
        total_valid_targets=total_valid_targets,
        total_non_special_tokens=total_non_special_tokens,
    )


class EarlyStopping:
    """Patience-based early stopping monitoring validation metric (e.g. macro_val_bpc)."""

    def __init__(self, patience: int = 8, min_delta: float = 0.0) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of validation evaluations without improvement before stopping.
            min_delta: Minimum required decrease to qualify as an improvement.
        """
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_metric = float("inf")
        self.should_stop = False

    def step(self, current_metric: float) -> bool:
        """Evaluate metric against best recorded so far.

        Args:
            current_metric: Latest validation metric (lower is better).

        Returns:
            True if a new best metric was achieved, False otherwise.
        """
        if current_metric < (self.best_metric - self.min_delta):
            self.best_metric = current_metric
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def state_dict(self) -> dict[str, Any]:
        """Serialize early stopping state for exact resumption."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best_metric": self.best_metric,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore early stopping state from checkpoint."""
        self.patience = int(state_dict["patience"])
        self.min_delta = float(state_dict["min_delta"])
        self.counter = int(state_dict["counter"])
        self.best_metric = float(state_dict["best_metric"])
        self.should_stop = bool(state_dict["should_stop"])


class MetricsLogger:
    """Manages metrics logging to jsonl and TensorBoard."""

    def __init__(self, run_dir: Path | str) -> None:
        """Initialize logger with run directory.

        Args:
            run_dir: Directory where metrics.jsonl and tensorboard/ will be stored.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.tb_dir = self.run_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir))

    def log_train_step(
        self,
        global_step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        cumulative_raw_chars: int,
        cumulative_model_tokens: int,
        effective_epoch: float,
    ) -> None:
        """Log training step metrics."""
        record = {
            "type": "train",
            "global_step": global_step,
            "loss": loss,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "cumulative_raw_chars": cumulative_raw_chars,
            "cumulative_model_tokens": cumulative_model_tokens,
            "effective_epoch": effective_epoch,
        }
        self._append_jsonl(record)

        self.tb_writer.add_scalar("train/loss", loss, global_step)
        self.tb_writer.add_scalar("train/learning_rate", lr, global_step)
        self.tb_writer.add_scalar("train/grad_norm", grad_norm, global_step)
        self.tb_writer.add_scalar("train/raw_characters_exposed", cumulative_raw_chars, global_step)
        self.tb_writer.add_scalar("train/tokens_exposed", cumulative_model_tokens, global_step)
        self.tb_writer.add_scalar("train/effective_epoch", effective_epoch, global_step)

    def log_validation(
        self,
        global_step: int,
        cumulative_raw_chars: int,
        effective_epoch: float,
        metrics: ValidationMetrics,
    ) -> None:
        """Log validation evaluation metrics."""
        record = {
            "type": "val",
            "global_step": global_step,
            "cumulative_raw_chars": cumulative_raw_chars,
            "effective_epoch": effective_epoch,
            **metrics.to_dict(),
        }
        self._append_jsonl(record)

        self.tb_writer.add_scalar("val/loss", metrics.val_loss, global_step)
        self.tb_writer.add_scalar("val/perplexity", metrics.val_perplexity, global_step)
        self.tb_writer.add_scalar("val/macro_bpc", metrics.macro_val_bpc, global_step)
        self.tb_writer.add_scalar("val/micro_bpc", metrics.micro_val_bpc, global_step)

        for fam, bpc in metrics.family_bpc.items():
            self.tb_writer.add_scalar(f"val/family_bpc/{fam}", bpc, global_step)

        self.tb_writer.flush()

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        """Flush and close TensorBoard writer."""
        self.tb_writer.flush()
        self.tb_writer.close()
