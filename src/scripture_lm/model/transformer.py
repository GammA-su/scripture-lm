"""Decoder-only Transformer language model implemented from scratch in PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripture_lm.model.block import TransformerBlock
from scripture_lm.model.config import TransformerConfig
from scripture_lm.model.rmsnorm import RMSNorm
from scripture_lm.model.rope import RotaryEmbedding


@dataclass
class ModelOutput:
    """Structured output container for TransformerLM forward pass."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TransformerLM(nn.Module):
    """Decoder-only autoregressive Transformer with RoPE, RMSNorm, SwiGLU, and Flash SDPA."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Shared RoPE instance across all layers
        self.rotary = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_context_length,
            theta=config.rope_theta,
        )

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx=i, rotary_emb=self.rotary)
                for i in range(config.layers)
            ]
        )

        self.final_norm = RMSNorm(config.d_model, eps=config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 1. Initialize ordinary weights
        self.apply(self._init_weights)

        # 2. Apply scaled residual output projection initialization
        resid_std = config.init_std / math.sqrt(2 * config.layers)
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=resid_std)

        # 3. Tie embedding and output projection weights AFTER initialization
        if config.embedding_tying:
            self.lm_head.weight = self.tok_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear layers and embeddings with Normal(0, init_std), RMSNorm to 1."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Forward pass for causal next-token prediction.

        Args:
            input_ids: Tensor of shape (batch, seq_len) with token IDs.
            targets: Optional tensor of shape (batch, seq_len) with shifted next-token targets.

        Returns:
            ModelOutput with logits of shape (batch, seq_len, vocab_size) and optional loss.
        """
        _, t = input_ids.shape

        # Strictly enforce configured context limit
        if t > self.config.max_context_length:
            raise ValueError(
                f"Sequence length {t} exceeds configured "
                f"max_context_length {self.config.max_context_length}."
            )

        x = self.tok_embeddings(input_ids)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            # Compute loss in FP32 for numerical stability
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=self.config.ignore_index,
            )

        return ModelOutput(logits=logits, loss=loss)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return total parameter count, handling tied weights without double-counting."""
        unique_params = {id(p): p for p in self.parameters()}
        if non_embedding:
            emb_id = id(self.tok_embeddings.weight)
            return sum(p.numel() for p_id, p in unique_params.items() if p_id != emb_id)
        return sum(p.numel() for p in unique_params.values())

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total parameter count, handling tied weights without double-counting."""
        if trainable_only:
            unique_params = {id(p): p for p in self.parameters() if p.requires_grad}
            return sum(p.numel() for p in unique_params.values())
        return self.get_num_params(non_embedding=False)

    def parameter_summary(self) -> dict[str, Any]:
        """Return structured summary of model parameters and architecture."""
        total = self.get_num_params(non_embedding=False)
        non_emb = self.get_num_params(non_embedding=True)
        emb_params = total - non_emb

        return {
            "total_parameters": total,
            "non_embedding_parameters": non_emb,
            "embedding_parameters": emb_params,
            "layers": self.config.layers,
            "d_model": self.config.d_model,
            "heads": self.config.heads,
            "head_dim": self.config.head_dim,
            "mlp_hidden": self.config.mlp_hidden,
            "vocab_size": self.config.vocab_size,
            "max_context_length": self.config.max_context_length,
            "embedding_tying": self.config.embedding_tying,
        }
