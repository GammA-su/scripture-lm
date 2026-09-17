"""Strongly typed Pydantic configuration models for Scripture-LM."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """Transformer architecture configuration."""

    model_config = ConfigDict(extra="forbid")

    layers: int = 6
    d_model: int = 256
    heads: int = 8
    mlp_hidden: int = 704
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.10
    attention_dropout: float = 0.10
    linear_bias: bool = False
    embedding_tying: bool = True

    @property
    def head_dim(self) -> int:
        """Derived dimension per attention head."""
        return self.d_model // self.heads

    @model_validator(mode="after")
    def validate_architecture(self) -> ModelConfig:
        """Ensure d_model is divisible by heads."""
        if self.d_model % self.heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by heads ({self.heads})")
        return self


class BPETokenizerConfig(BaseModel):
    """Byte-level BPE tokenizer configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["bpe"] = "bpe"
    bpe_vocab_size: int = 4096
    context_length: int = 512
    special_tokens: list[str] = Field(default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"])


class CharacterTokenizerConfig(BaseModel):
    """Unicode codepoint character tokenizer configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["character"] = "character"
    context_length: int = 2048
    special_tokens: list[str] = Field(default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"])


TokenizerConfig = Annotated[
    Union[BPETokenizerConfig, CharacterTokenizerConfig],
    Field(discriminator="type"),
]


class DataConfig(BaseModel):
    """Dataset sampling and loader configuration."""

    model_config = ConfigDict(extra="forbid")

    sampling_mode: Literal["natural", "temperature"] = "natural"
    sampling_alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    """Optimizer, precision, and training schedule configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 1337
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.10
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    gradient_clip: float = 1.0
    precision: Literal["bf16", "fp32", "fp16"] = "bf16"
    compile: bool = True
    device: str = "cuda"
    microbatch_size: int = 8
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.02
    max_effective_epochs: int = Field(default=20, gt=0)
    eval_frequency_per_epoch: int = Field(default=4, gt=0)
    early_stopping_enabled: bool = True
    early_stopping_patience: int = Field(default=8, gt=0)
    monitor_metric: str = "macro_val_bpc"


class BenchmarkConfig(BaseModel):
    """Fixed benchmark generation prompt suite."""

    model_config = ConfigDict(extra="forbid")

    samples_per_prompt: int = 10
    prompts: list[str] = Field(
        default_factory=lambda: [
            "<BOS>",
            "And the Lord said",
            "And the prophet spoke unto the people",
            "Blessed are those who",
            "Then came a man from",
            "And behold, there appeared",
            "For the people had turned away",
            "And in those days",
            "Thus was it written",
        ]
    )
    fixed_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


class GenerationConfig(BaseModel):
    """Inference and text generation parameters."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0
    stop_on_eos: bool = True
    seed: int | None = None
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


class EvaluationConfig(BaseModel):
    """Evaluation metrics configuration."""

    model_config = ConfigDict(extra="forbid")

    calculate_bpc: bool = True
    calculate_perplexity: bool = True
    calculate_memorization: bool = True
    longest_match_thresholds: list[int] = Field(default_factory=lambda: [50, 100, 200])


class ScriptureLMConfig(BaseModel):
    """Complete root configuration for Scripture-LM."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary onto base dictionary."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def unflatten_dict(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert dot-delimited key names (e.g. 'data.sampling_mode') into nested dictionary."""
    result: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def find_default_base_config() -> Path:
    """Locate configs/base.toml relative to cwd or repository root."""
    cwd_path = Path("configs/base.toml")
    if cwd_path.is_file():
        return cwd_path.resolve()

    repo_path = Path(__file__).resolve().parent.parent.parent / "configs" / "base.toml"
    if repo_path.is_file():
        return repo_path

    raise FileNotFoundError("Could not find configs/base.toml")


def load_config(
    config_path: Path | str | None = None,
    base_path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ScriptureLMConfig:
    """Load and validate Scripture-LM configuration with precedence:

    base config (configs/base.toml)
         ↓
    model-specific config (--config)
         ↓
    CLI overrides
         ↓
    Pydantic validation
    """
    resolved_base = Path(base_path) if base_path else find_default_base_config()
    with open(resolved_base, "rb") as f:
        base_dict = tomllib.load(f)

    merged = base_dict

    if config_path:
        specific_path = Path(config_path)
        if not specific_path.is_file():
            raise FileNotFoundError(f"Config file not found: {specific_path}")
        with open(specific_path, "rb") as f:
            specific_dict = tomllib.load(f)
        # Deep merge specific config onto base
        merged = deep_merge(merged, specific_dict)

    if cli_overrides:
        nested_overrides = unflatten_dict(cli_overrides)
        merged = deep_merge(merged, nested_overrides)

    return ScriptureLMConfig.model_validate(merged)
