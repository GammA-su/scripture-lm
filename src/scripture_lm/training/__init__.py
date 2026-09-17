"""Training package for Scripture-LM: Trainer, optimizers, schedulers, metrics, and checkpoints."""

from scripture_lm.training.checkpoint import load_checkpoint, save_checkpoint
from scripture_lm.training.metrics import (
    EarlyStopping,
    MetricsLogger,
    ValidationMetrics,
    compute_validation_metrics,
)
from scripture_lm.training.optimizer import clip_gradients, configure_optimizer
from scripture_lm.training.scheduler import ExposureCosineScheduler
from scripture_lm.training.trainer import Trainer, serialize_toml_dict, verify_encoding_provenance

__all__ = [
    "EarlyStopping",
    "ExposureCosineScheduler",
    "MetricsLogger",
    "Trainer",
    "ValidationMetrics",
    "clip_gradients",
    "compute_validation_metrics",
    "configure_optimizer",
    "load_checkpoint",
    "save_checkpoint",
    "serialize_toml_dict",
    "verify_encoding_provenance",
]
