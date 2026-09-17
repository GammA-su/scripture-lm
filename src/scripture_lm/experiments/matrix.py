"""Canonical baseline v1 and semantic scientific configuration identity."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from scripture_lm.config import ScriptureLMConfig


def scientific_config(config: ScriptureLMConfig) -> dict[str, Any]:
    """Canonicalize scientific settings, excluding only execution choices."""
    result = config.model_dump(mode="json")
    result["training"].pop("device")
    result["training"].pop("compile")
    if result["data"]["sampling_mode"] == "natural":
        result["data"]["sampling_alpha"] = None
    return result


def config_hash(specification: dict[str, Any]) -> str:
    """Hash canonical JSON, independent of TOML formatting and dictionary order."""
    payload = json.dumps(specification, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def alpha_label(value: str | float) -> str:
    """Format a finite decimal alpha without float rounding or exponent notation."""
    try:
        alpha = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Alpha must be a decimal between 0 and 1") from exc
    if not alpha.is_finite() or not Decimal(0) <= alpha <= Decimal(1):
        raise ValueError("Alpha must be a decimal between 0 and 1")
    label = format(alpha, "f").rstrip("0").rstrip(".") if alpha else "0"
    if alpha == 1:
        label = "1"
    return "a" + label.replace(".", "p")


@dataclass(frozen=True)
class Experiment:
    """Immutable serialized definition; each execution receives a fresh config."""

    name: str
    config_json: str
    baseline: bool = False

    def resolve(
        self, *, device: str | None = None, compile_model: bool | None = None
    ) -> ScriptureLMConfig:
        config = ScriptureLMConfig.model_validate_json(self.config_json)
        if device is not None:
            config.training.device = device
        if compile_model is not None:
            config.training.compile = compile_model
        return config


def baseline_matrix() -> tuple[Experiment, ...]:
    """Load the versioned recipe, independent of editable user configs."""
    base = tomllib.loads(Path(__file__).with_name("baseline_v1.toml").read_text(encoding="utf-8"))
    experiments = []
    for tokenizer in ("bpe", "char"):
        for mode in ("natural", "temperature"):
            data = json.loads(json.dumps(base))
            data["tokenizer"] = (
                {"type": "bpe", "bpe_vocab_size": 4096, "context_length": 512}
                if tokenizer == "bpe"
                else {"type": "character", "context_length": 2048}
            )
            data["data"] = {"sampling_mode": mode, "sampling_alpha": 0.5}
            data["tokenizer"]["special_tokens"] = ["<pad>", "<bos>", "<eos>", "<unk>"]
            config = ScriptureLMConfig.model_validate(data)
            suffix = "natural" if mode == "natural" else "temperature-a05"
            experiments.append(Experiment(f"{tokenizer}-{suffix}", config.model_dump_json(), True))
    return tuple(experiments)


def get_baseline(name: str) -> Experiment:
    for experiment in baseline_matrix():
        if experiment.name == name:
            return experiment
    raise ValueError(f"Unknown baseline experiment: {name}")


def custom_experiment(
    tokenizer: Literal["bpe", "char"],
    sampling_mode: Literal["natural", "temperature"],
    sampling_alpha: str = "0.5",
    effective_epochs: int = 20,
    seed: int = 1337,
    config: ScriptureLMConfig | None = None,
) -> Experiment:
    """Create a distinct custom identity, including all scientific overrides."""
    label = alpha_label(sampling_alpha)
    cfg = config.model_copy(deep=True) if config else get_baseline(f"{tokenizer}-natural").resolve()
    expected_type = "character" if tokenizer == "char" else "bpe"
    if cfg.tokenizer.type != expected_type:
        raise ValueError("Custom config tokenizer disagrees with --tokenizer")
    cfg.data.sampling_mode = sampling_mode
    cfg.data.sampling_alpha = float(sampling_alpha)
    cfg.training.max_effective_epochs = effective_epochs
    cfg.training.seed = seed
    cfg = ScriptureLMConfig.model_validate(cfg.model_dump())
    suffix = f"temperature-{label}" if sampling_mode == "temperature" else "natural"
    digest = config_hash(scientific_config(cfg))
    return Experiment(f"{tokenizer}-{suffix}-custom-{digest[:12]}", cfg.model_dump_json())
