"""Unit tests for configuration models, inheritance, validation, and overrides."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripture_lm.config import (
    BPETokenizerConfig,
    CharacterTokenizerConfig,
    DataConfig,
    ModelConfig,
    load_config,
)


def test_model_config_defaults_and_derived_head_dim() -> None:
    """Verify ModelConfig defaults and derived head dimension."""
    cfg = ModelConfig()
    assert cfg.layers == 6
    assert cfg.d_model == 256
    assert cfg.heads == 8
    assert cfg.head_dim == 32
    assert cfg.mlp_hidden == 704
    assert cfg.dropout == 0.10
    assert cfg.attention_dropout == 0.10
    assert cfg.linear_bias is False
    assert cfg.embedding_tying is True


def test_architecture_consistency_validation() -> None:
    """Verify that d_model not divisible by heads raises ValidationError."""
    with pytest.raises(ValidationError, match="must be divisible by heads"):
        ModelConfig(d_model=255, heads=8)


def test_config_inheritance_bpe() -> None:
    """Verify loading bpe.toml merges with base.toml properly."""
    config = load_config(config_path=Path("configs/bpe.toml"))
    assert config.model.layers == 6
    assert config.model.d_model == 256
    assert isinstance(config.tokenizer, BPETokenizerConfig)
    assert config.tokenizer.type == "bpe"
    assert config.tokenizer.bpe_vocab_size == 4096
    assert config.tokenizer.context_length == 512
    assert config.data.sampling_mode == "natural"


def test_config_inheritance_char() -> None:
    """Verify loading char.toml merges with base.toml and contains no BPE fields."""
    config = load_config(config_path=Path("configs/char.toml"))
    assert config.model.layers == 6
    assert config.model.d_model == 256
    assert isinstance(config.tokenizer, CharacterTokenizerConfig)
    assert config.tokenizer.type == "character"
    assert config.tokenizer.context_length == 2048
    assert not hasattr(config.tokenizer, "bpe_vocab_size")


def test_cli_override_precedence() -> None:
    """Verify that CLI overrides take precedence over TOML files."""
    overrides = {
        "data.sampling_mode": "temperature",
        "data.sampling_alpha": 0.35,
        "training.seed": 9999,
        "training.compile": False,
        "model.layers": 8,
    }
    config = load_config(config_path=Path("configs/bpe.toml"), cli_overrides=overrides)
    assert config.data.sampling_mode == "temperature"
    assert config.data.sampling_alpha == 0.35
    assert config.training.seed == 9999
    assert config.training.compile is False
    assert config.model.layers == 8
    # Unoverridden values remain intact
    assert config.tokenizer.type == "bpe"
    assert config.model.d_model == 256


def test_invalid_sampling_mode() -> None:
    """Verify that an unsupported sampling mode is rejected."""
    with pytest.raises(ValidationError):
        DataConfig(sampling_mode="invalid_mode")  # type: ignore[arg-type]


def test_sampling_alpha_is_always_range_checked() -> None:
    """Verify that sampling_alpha outside [0.0, 1.0] is always rejected, even in natural mode."""
    # Test above upper bound
    with pytest.raises(ValidationError):
        DataConfig(sampling_mode="natural", sampling_alpha=1.5)
    with pytest.raises(ValidationError):
        DataConfig(sampling_mode="temperature", sampling_alpha=42.0)

    # Test below lower bound
    with pytest.raises(ValidationError):
        DataConfig(sampling_mode="natural", sampling_alpha=-0.1)
    with pytest.raises(ValidationError):
        DataConfig(sampling_mode="temperature", sampling_alpha=-0.5)


def test_natural_mode_ignores_valid_alpha() -> None:
    """Verify that natural mode accepts valid alpha without error (alpha is unused downstream)."""
    cfg = DataConfig(sampling_mode="natural", sampling_alpha=0.7)
    assert cfg.sampling_mode == "natural"
    assert cfg.sampling_alpha == 0.7


def test_unknown_config_keys_fail() -> None:
    """Verify that typo or extra unrecognized keys are rejected by extra='forbid'."""
    # DataConfig typo
    with pytest.raises(ValidationError):
        DataConfig.model_validate({"sampling_mode": "natural", "sampling_alhpa": 0.5})

    # ModelConfig unknown parameter
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"layers": 6, "unknown_field": True})


def test_character_tokenizer_rejects_bpe_vocab_size() -> None:
    """Verify that character tokenizer rejects bpe_vocab_size key."""
    with pytest.raises(ValidationError):
        CharacterTokenizerConfig.model_validate(
            {"type": "character", "bpe_vocab_size": 4096, "context_length": 2048}
        )
