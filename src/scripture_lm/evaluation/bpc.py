"""Cross-tokenizer Bits-Per-Character (BPC) evaluation for Scripture-LM.

BPC is the primary scientific metric for comparing models across different tokenizers
(e.g. byte-level BPE vs character-level).

Definition:
    BPC = -log2 P(held-out text) / number of original normalized characters

Tokens included in likelihood numerator:
    - All non-special text tokens.
    - UNK tokens (which represent authentic characters from the original text).
    - Multi-byte BPE subword and byte tokens (regardless of single-token character credit).

Tokens excluded from likelihood numerator:
    - PAD (<pad>=0)
    - BOS (<bos>=1)
    - EOS (<eos>=2)
    - Masked targets (-100)

Denominator:
    - Sum of original normalized scripture characters only.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from scripture_lm.tokenization.base import BOS_ID, EOS_ID, PAD_ID

# Structural tokens excluded from BPC likelihood computation.
# Note: UNK_ID is strictly NOT in this set because it represents genuine corpus characters.
BPC_EXCLUDED_IDS: frozenset[int] = frozenset({PAD_ID, BOS_ID, EOS_ID})


class BPCResult(BaseModel):
    """Container for bits-per-character evaluation results."""

    model_config = ConfigDict(extra="forbid")

    macro_bpc: float = Field(..., description="Unweighted arithmetic mean of family BPC values")
    micro_bpc: float = Field(..., description="Total non-special bits divided by total characters")
    family_bpc: dict[str, float] = Field(
        default_factory=dict, description="BPC calculated per scripture family"
    )
    total_bits: float = Field(..., description="Total negative log2 likelihood in bits")
    total_characters: int = Field(..., description="Total original normalized characters evaluated")
    total_non_special_tokens: int = Field(
        ..., description="Total prediction tokens contributing to BPC"
    )
    family_bits: dict[str, float] = Field(
        default_factory=dict, description="Bits sum per scripture family"
    )
    family_characters: dict[str, int] = Field(
        default_factory=dict, description="Character sum per scripture family"
    )


def compute_bpc(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device | str,
    autocast_context: Any = None,
) -> BPCResult:
    """Compute Bits Per Character (BPC) over a complete dataset split.

    Executes under model.eval() and torch.inference_mode() for determinism,
    zero autograd overhead, and disabled dropout.

    Args:
        model: Autoregressive language model.
        dataloader: DataLoader traversing the split sequentially.
        device: Device to transfer batch tensors to.
        autocast_context: Optional context manager (e.g. torch.autocast) for mixed precision.

    Returns:
        BPCResult containing macro BPC, micro BPC, and per-family BPCs.
    """
    model.eval()

    family_bits: dict[str, float] = {}
    family_characters: dict[str, int] = {}
    total_non_special_tokens = 0
    log2_e = 1.0 / math.log(2.0)

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

            # Cross-entropy in nats: shape (batch_size, seq_len)
            token_nlls = F.cross_entropy(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(batch_size, seq_len)

            # Valid targets (excluding padding ignore_index -100)
            valid_mask = target_ids != -100

            # BPC mask: exclude structural boundary tokens PAD, BOS, EOS.
            # Retain all text tokens, byte pieces, and UNK_ID.
            bpc_mask = valid_mask.clone()
            for exc_id in BPC_EXCLUDED_IDS:
                bpc_mask &= target_ids != exc_id

            chunk_bits = (token_nlls * bpc_mask).sum(dim=1) * log2_e
            chunk_token_counts = bpc_mask.sum(dim=1)

            # Aggregate per chunk and family
            for i in range(batch_size):
                fam = str(families[i])
                bits = float(chunk_bits[i].item())
                n_tokens = int(chunk_token_counts[i].item())

                # Extract chunk raw_character_count from batch metadata
                if "raw_characters_per_chunk" in batch:
                    raw_chars = int(batch["raw_characters_per_chunk"][i])
                elif "chunk_ids" in batch and hasattr(dataloader.dataset, "chunks"):
                    c_id = batch["chunk_ids"][i]
                    if isinstance(dataloader.dataset.chunks, list) and isinstance(c_id, int):
                        raw_chars = dataloader.dataset.chunks[c_id].raw_character_count
                    else:
                        raw_chars = max(1, int(batch["raw_characters"]) // batch_size)
                else:
                    raw_chars = max(1, int(batch["raw_characters"]) // batch_size)

                family_bits[fam] = family_bits.get(fam, 0.0) + bits
                family_characters[fam] = family_characters.get(fam, 0) + raw_chars
                total_non_special_tokens += n_tokens

    # Compute family BPCs
    family_bpc: dict[str, float] = {}
    for fam in sorted(family_bits.keys()):
        bits = family_bits[fam]
        chars = family_characters.get(fam, 0)
        family_bpc[fam] = bits / chars if chars > 0 else float("inf")

    # Macro BPC: unweighted mean across families
    if family_bpc:
        macro_bpc = sum(family_bpc.values()) / len(family_bpc)
    else:
        macro_bpc = float("inf")

    # Micro BPC: total bits / total characters
    total_bits = sum(family_bits.values())
    total_chars = sum(family_characters.values())
    micro_bpc = total_bits / total_chars if total_chars > 0 else float("inf")

    return BPCResult(
        macro_bpc=macro_bpc,
        micro_bpc=micro_bpc,
        family_bpc=family_bpc,
        total_bits=total_bits,
        total_characters=total_chars,
        total_non_special_tokens=total_non_special_tokens,
        family_bits=family_bits,
        family_characters=family_characters,
    )
