"""Strongly typed configuration for the Scripture-LM Transformer architecture."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripture_lm.config import ScriptureLMConfig


class TransformerConfig(BaseModel):
    """Configuration for decoder-only Transformer language model."""

    model_config = ConfigDict(extra="forbid")

    vocab_size: int = Field(..., gt=4, description="Total vocabulary size including special tokens")
    max_context_length: int = Field(
        ..., gt=0, description="Maximum context length L (strictly enforced)"
    )
    layers: int = Field(default=6, gt=0, description="Number of Transformer blocks")
    d_model: int = Field(default=256, gt=0, description="Hidden dimension")
    heads: int = Field(default=8, gt=0, description="Number of attention heads")
    mlp_hidden: int = Field(default=704, gt=0, description="SwiGLU intermediate dimension")
    rms_eps: float = Field(
        default=1e-5, gt=0.0, description="RMSNorm epsilon for numerical stability"
    )
    rope_theta: float = Field(default=10000.0, gt=0.0, description="RoPE base frequency")
    dropout: float = Field(default=0.10, ge=0.0, lt=1.0, description="Residual dropout probability")
    attention_dropout: float = Field(
        default=0.10, ge=0.0, lt=1.0, description="Attention dropout probability (disabled in eval)"
    )
    linear_bias: bool = Field(default=False, description="Whether to include bias in linear layers")
    embedding_tying: bool = Field(
        default=True, description="Whether lm_head shares weights with input embeddings"
    )
    init_std: float = Field(
        default=0.02, gt=0.0, description="Standard deviation for weight initialization"
    )
    ignore_index: int = Field(default=-100, description="Target index ignored in loss")
    pad_token_id: int = Field(default=0, ge=0, description="Pad token ID")

    @property
    def head_dim(self) -> int:
        """Derived dimension per attention head."""
        return self.d_model // self.heads

    @model_validator(mode="after")
    def validate_architecture(self) -> TransformerConfig:
        """Validate dimension divisibility and RoPE even head dimension requirement."""
        if self.d_model % self.heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by heads ({self.heads})")
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be an even integer for Rotary Position Embeddings"
            )
        return self

    @classmethod
    def from_app_config(
        cls,
        cfg: ScriptureLMConfig,
        vocab_size: int,
        max_context_length: int | None = None,
        **overrides: Any,
    ) -> TransformerConfig:
        """Construct TransformerConfig from top-level ScriptureLMConfig and vocab size."""
        ctx_len = (
            max_context_length if max_context_length is not None else cfg.tokenizer.context_length
        )
        params: dict[str, Any] = {
            "vocab_size": vocab_size,
            "max_context_length": ctx_len,
            "layers": cfg.model.layers,
            "d_model": cfg.model.d_model,
            "heads": cfg.model.heads,
            "mlp_hidden": cfg.model.mlp_hidden,
            "rms_eps": cfg.model.rms_eps,
            "rope_theta": cfg.model.rope_theta,
            "dropout": cfg.model.dropout,
            "attention_dropout": cfg.model.attention_dropout,
            "linear_bias": cfg.model.linear_bias,
            "embedding_tying": cfg.model.embedding_tying,
        }
        params.update(overrides)
        return cls.model_validate(params)
