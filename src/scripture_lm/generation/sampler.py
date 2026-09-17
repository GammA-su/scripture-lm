"""Next-token sampling algorithms with greedy, temperature, top-p, and top-k strategies."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripture_lm.tokenization.base import PAD_ID


def validate_sampling_parameters(
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> None:
    """Validate sampling hyperparameters, raising ValueError on invalid configurations."""
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0.0, got {temperature}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0.0, 1.0], got {top_p}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the next token ID from an unnormalized logit vector.

    Args:
        logits: Unnormalized token scores of shape (vocab_size,) or (batch_size, vocab_size).
        temperature: Sampling temperature. Exactly 0.0 triggers greedy argmax decoding.
        top_p: Nucleus sampling probability threshold in (0.0, 1.0].
        top_k: Optional top-k filter bound >= 1.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Tensor containing sampled token ID(s) shaped (batch_size, 1) or scalar tensor.
    """
    validate_sampling_parameters(temperature, top_p, top_k)

    is_1d = logits.ndim == 1
    if is_1d:
        logits = logits.unsqueeze(0)

    # Clone logits in FP32 for numerical stability
    logits = logits.float().clone()

    # Invariant: <pad> is never a valid autoregressively generated token
    logits[..., PAD_ID] = -float("inf")

    # 1. Greedy Decoding (strictly when temperature == 0.0)
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token.squeeze(0) if is_1d else next_token

    # 2. Temperature scaling
    logits = logits / temperature

    # 3. Top-K filtering
    if top_k is not None:
        k_val = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k_val, dim=-1)
        kth_val = topk_vals[..., [-1]]
        logits[logits < kth_val] = -float("inf")

    # 4. Top-P (nucleus) filtering & sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Mask out tokens beyond the cumulative top-p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right so the first token that crosses threshold is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_logits[sorted_indices_to_remove] = -float("inf")
        probs = F.softmax(sorted_logits, dim=-1)
        sample_idx = torch.multinomial(probs, num_samples=1, generator=generator)
        next_token = torch.gather(sorted_indices, -1, sample_idx)
    else:
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)

    return next_token.squeeze(0) if is_1d else next_token
