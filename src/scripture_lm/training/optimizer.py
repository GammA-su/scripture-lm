"""Optimizer configuration with parameter grouping and gradient clipping for Scripture-LM."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from scripture_lm.config import TrainingConfig


def configure_optimizer(
    raw_model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """Construct AdamW optimizer with strict parameter grouping.

    - Excludes 1D parameters (RMSNorm scales, biases) from weight decay.
    - Applies weight decay (config.weight_decay) to 2D+ parameters (linear weights, embeddings).
    - Deduplicates shared/tied parameters (e.g. lm_head.weight is tok_embeddings.weight).

    Args:
        raw_model: Uncompiled TransformerLM model instance.
        config: TrainingConfig containing optimizer hyperparameters.

    Returns:
        Configured torch.optim.AdamW optimizer.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue

        # Prevent duplicate parameter registration for tied weights
        param_id = id(param)
        if param_id in seen_param_ids:
            continue
        seen_param_ids.add(param_id)

        # 1D normalization scales and biases receive 0.0 weight decay
        if param.ndim < 2 or "scale" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups: list[dict[str, Any]] = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )

    return optimizer


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """Clip global gradient norm across all model parameters.

    Args:
        model: Model whose gradients should be clipped.
        max_norm: Maximum allowed gradient norm.

    Returns:
        The total computed gradient norm before clipping.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    return float(total_norm)
