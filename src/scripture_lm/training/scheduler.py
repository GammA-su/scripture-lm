"""Exposure-based learning rate scheduler for Scripture-LM.

Schedules learning rate based on cumulative raw scripture character exposure
rather than arbitrary optimizer step numbers, guaranteeing that all four models
(BPE natural, BPE temperature, CHAR natural, CHAR temperature) experience identical
learning rate schedules at identical scripture exposure.
"""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer


class ExposureCosineScheduler:
    """Cosine learning rate scheduler with linear warmup paced by raw character exposure.

    Progress P is defined as:
        P = cumulative_raw_chars / total_target_exposure
        where total_target_exposure = N * max_effective_epochs.

    - Warmup (0% to 2% of total_target_exposure):
        Linear ramp from 0.0 to base_lr.
    - Cosine Decay (2% to 100% of total_target_exposure):
        Cosine decay from base_lr down to min_lr.
    - Tail (>= 100% of total_target_exposure):
        Clamped at min_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_target_exposure: int,
        warmup_ratio: float = 0.02,
        min_lr: float = 3e-5,
        base_lr: float = 3e-4,
    ) -> None:
        """Initialize exposure scheduler.

        Args:
            optimizer: PyTorch optimizer whose learning rates are managed.
            total_target_exposure: Total target characters (N * max_effective_epochs).
            warmup_ratio: Fraction of exposure dedicated to linear warmup (default 0.02).
            min_lr: Minimum learning rate floor after decay (default 3e-5).
            base_lr: Peak learning rate achieved after warmup (default 3e-4).
        """
        if total_target_exposure <= 0:
            raise ValueError(f"total_target_exposure must be positive, got {total_target_exposure}")
        if min_lr < 0 or base_lr <= 0 or min_lr > base_lr:
            raise ValueError(f"Invalid LR bounds: min_lr={min_lr}, base_lr={base_lr}")
        if not (0.0 <= warmup_ratio < 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")

        self.optimizer = optimizer
        self.total_target_exposure = int(total_target_exposure)
        self.warmup_ratio = float(warmup_ratio)
        self.min_lr = float(min_lr)
        self.base_lr = float(base_lr)

        self.warmup_exposure = int(round(self.total_target_exposure * self.warmup_ratio))
        self.cumulative_raw_chars = 0
        self.step_count = 0
        self.current_lr = 0.0 if self.warmup_exposure > 0 else self.base_lr

        # Set initial LR in optimizer
        self._apply_lr(self.current_lr)

    def _calculate_lr(self, cumulative_raw_chars: int) -> float:
        """Compute target learning rate for given cumulative raw character exposure."""
        if self.warmup_exposure > 0 and cumulative_raw_chars < self.warmup_exposure:
            # Linear warmup from 0.0 to base_lr
            alpha = max(0.0, float(cumulative_raw_chars)) / float(self.warmup_exposure)
            return self.base_lr * alpha

        if cumulative_raw_chars >= self.total_target_exposure:
            # Clamped at floor
            return self.min_lr

        # Cosine decay between warmup_exposure and total_target_exposure
        decay_denom = max(1, self.total_target_exposure - self.warmup_exposure)
        decay_ratio = float(cumulative_raw_chars - self.warmup_exposure) / float(decay_denom)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.base_lr - self.min_lr)

    def _apply_lr(self, lr: float) -> None:
        """Apply learning rate across all optimizer parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, cumulative_raw_chars: int) -> float:
        """Advance scheduler with latest cumulative raw character count.

        Args:
            cumulative_raw_chars: Total cumulative raw characters exposed so far.

        Returns:
            The active learning rate after this update.
        """
        self.cumulative_raw_chars = int(cumulative_raw_chars)
        self.step_count += 1
        self.current_lr = self._calculate_lr(self.cumulative_raw_chars)
        self._apply_lr(self.current_lr)
        return self.current_lr

    def get_lr(self) -> float:
        """Get the current learning rate."""
        return self.current_lr

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for exact checkpoint resumption."""
        return {
            "total_target_exposure": self.total_target_exposure,
            "warmup_ratio": self.warmup_ratio,
            "warmup_exposure": self.warmup_exposure,
            "min_lr": self.min_lr,
            "base_lr": self.base_lr,
            "cumulative_raw_chars": self.cumulative_raw_chars,
            "step_count": self.step_count,
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self.total_target_exposure = int(state_dict["total_target_exposure"])
        self.warmup_ratio = float(state_dict["warmup_ratio"])
        self.warmup_exposure = int(state_dict["warmup_exposure"])
        self.min_lr = float(state_dict["min_lr"])
        self.base_lr = float(state_dict["base_lr"])
        self.cumulative_raw_chars = int(state_dict["cumulative_raw_chars"])
        self.step_count = int(state_dict["step_count"])
        self.current_lr = float(state_dict["current_lr"])
        self._apply_lr(self.current_lr)
