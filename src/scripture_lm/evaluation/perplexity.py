"""Token-level cross-entropy and perplexity evaluation for Scripture-LM.

Token Perplexity Definition:
    Evaluated over all valid prediction targets except PAD (i.e. target != -100 and
    target != PAD_ID). This includes EOS, BOS (where applicable as targets), and UNK.
    It measures how well the model performs on its native autoregressive token
    prediction task.

IMPORTANT SCIENTIFIC NOTE:
    Token perplexity is strictly a token-level metric and CANNOT be used to compare
    or rank models with different tokenizers (e.g. BPE vocab 4096 vs Character vocab ~100).
    A character model predicting single codepoints naturally achieves lower token perplexity
    because each step chooses among fewer alternatives, while BPE tokens carry multiple
    characters. Cross-tokenizer comparison MUST be conducted using Bits Per Character (BPC).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from scripture_lm.tokenization.base import PAD_ID


class PerplexityResult(BaseModel):
    """Container for token-level cross-entropy and perplexity metrics."""

    model_config = ConfigDict(extra="forbid")

    cross_entropy_per_token: float = Field(
        ..., description="Average cross-entropy loss per target token (nats/token)"
    )
    perplexity: float = Field(
        ..., description="Token-level perplexity: exp(cross_entropy_per_token)"
    )
    family_cross_entropy: dict[str, float] = Field(
        default_factory=dict, description="Cross-entropy per token broken down by scripture family"
    )
    family_perplexity: dict[str, float] = Field(
        default_factory=dict, description="Token perplexity broken down by scripture family"
    )
    total_valid_targets: int = Field(
        ..., description="Total valid target tokens evaluated (excluding padding)"
    )
    total_nll_nats: float = Field(
        ..., description="Sum of negative log-likelihoods in nats across all valid targets"
    )
    family_valid_targets: dict[str, int] = Field(
        default_factory=dict, description="Valid target token count per scripture family"
    )
    family_nll_nats: dict[str, float] = Field(
        default_factory=dict, description="Total NLL in nats per scripture family"
    )


def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device | str,
    autocast_context: Any = None,
) -> PerplexityResult:
    """Compute token-level cross-entropy loss and perplexity across a split.

    Executes under model.eval() and torch.inference_mode().
    Valid targets include all non-PAD tokens (including UNK, EOS, and BOS where applicable).

    Args:
        model: Autoregressive language model.
        dataloader: DataLoader traversing the split sequentially.
        device: Device to transfer tensors to.
        autocast_context: Optional context manager (e.g. torch.autocast) for mixed precision.

    Returns:
        PerplexityResult containing overall and per-family cross-entropy and perplexity.
    """
    model.eval()

    family_nlls: dict[str, float] = {}
    family_targets: dict[str, int] = {}

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            families = batch["families"]
            batch_size, seq_len = input_ids.shape

            if autocast_context is not None:
                with autocast_context:
                    out = model(input_ids)
            else:
                out = model(input_ids)

            logits = out.logits.float()
            vocab_size = logits.shape[-1]

            # Token-level cross-entropy in nats: shape (batch_size, seq_len)
            token_nlls = F.cross_entropy(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(batch_size, seq_len)

            # Valid targets: not -100 and not PAD_ID
            valid_mask = (target_ids != -100) & (target_ids != PAD_ID)

            chunk_nlls = (token_nlls * valid_mask).sum(dim=1)
            chunk_target_counts = valid_mask.sum(dim=1)

            for i in range(batch_size):
                fam = str(families[i])
                nll = float(chunk_nlls[i].item())
                n_targets = int(chunk_target_counts[i].item())

                family_nlls[fam] = family_nlls.get(fam, 0.0) + nll
                family_targets[fam] = family_targets.get(fam, 0) + n_targets

    total_nll = sum(family_nlls.values())
    total_targets = sum(family_targets.values())

    if total_targets > 0:
        overall_ce = total_nll / total_targets
        overall_ppl = math.exp(min(overall_ce, 100.0))
    else:
        overall_ce = float("inf")
        overall_ppl = float("inf")

    family_ce: dict[str, float] = {}
    family_ppl: dict[str, float] = {}
    for fam in sorted(family_nlls.keys()):
        fam_nll = family_nlls[fam]
        fam_t = family_targets.get(fam, 0)
        if fam_t > 0:
            ce = fam_nll / fam_t
            family_ce[fam] = ce
            family_ppl[fam] = math.exp(min(ce, 100.0))
        else:
            family_ce[fam] = float("inf")
            family_ppl[fam] = float("inf")

    return PerplexityResult(
        cross_entropy_per_token=overall_ce,
        perplexity=overall_ppl,
        family_cross_entropy=family_ce,
        family_perplexity=family_ppl,
        total_valid_targets=total_targets,
        total_nll_nats=total_nll,
        family_valid_targets=family_targets,
        family_nll_nats=family_nlls,
    )
