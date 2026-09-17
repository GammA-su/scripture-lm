"""Autoregressive text generation engine with KV caching and synthetic provenance tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from scripture_lm.evaluation.generation_suite import GenerationResult, GenerationSettings
from scripture_lm.generation.kv_cache import KVCache
from scripture_lm.generation.sampler import sample_next_token
from scripture_lm.model.transformer import TransformerLM
from scripture_lm.tokenization.base import BOS_ID, EOS_ID, BaseTokenizer


class GenerationArtifact(BaseModel):
    """Complete provenance artifact for synthetic model generation."""

    model_config = ConfigDict(extra="forbid")

    is_synthetic: bool = Field(
        default=True,
        description="Explicit label indicating this text is purely synthetic model output",
    )
    disclaimer: str = Field(
        default=(
            "SYNTHETIC MODEL OUTPUT: Generated autoregressively from random initialization "
            "by Scripture-LM. Not authentic historical scripture."
        ),
        description="Synthetic content disclaimer",
    )
    result: GenerationResult
    checkpoint_path: str
    checkpoint_model_sha256: str
    tokenizer_type: str
    training_sampling_mode: str | None = None
    training_temperature_alpha: float | None = None
    generation_temperature: float
    top_p: float
    top_k: int | None = None
    seed: int | None = None
    max_new_tokens: int
    max_new_characters: int | None = None
    timestamp: str


class TextGenerator:
    """Autoregressive language model generator implementing GeneratorProtocol."""

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: BaseTokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        *,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Generate text continuation given a prompt and sampling settings.

        Args:
            prompt: Text prompt string (empty string generates unprompted BOS continuation).
            settings: Immutable GenerationSettings specifying hyperparameters and limits.
            use_cache: If True, uses preallocated KVCache; if False, runs uncached full forward.

        Returns:
            GenerationResult containing decoded text, token IDs, and termination reason.
        """
        self.model.eval()

        with torch.inference_mode():
            # Setup device-matched RNG generator
            generator: torch.Generator | None = None
            if settings.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(settings.seed)

            # 1. Encode prompt
            if not prompt:
                prompt_tokens = [BOS_ID]
            else:
                prompt_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
                if not prompt_tokens:
                    prompt_tokens = [BOS_ID]

            prompt_len = len(prompt_tokens)
            max_ctx = self.model.config.max_context_length
            if prompt_len >= max_ctx:
                raise ValueError(f"Prompt length {prompt_len} exceeds max_context_length {max_ctx}")

            max_chars = settings.max_new_characters
            max_tokens = settings.max_new_tokens
            remaining_ctx = max_ctx - prompt_len
            token_cap = min(max_tokens, remaining_ctx)

            generated_tokens: list[int] = []
            finish_reason = "length"

            if use_cache:
                # 2. Preallocated KV Cache path
                model_dtype = next(self.model.parameters()).dtype
                cache = KVCache(
                    num_layers=self.model.config.layers,
                    batch_size=1,
                    heads=self.model.config.heads,
                    head_dim=self.model.config.head_dim,
                    max_context_length=max_ctx,
                    device=self.device,
                    dtype=model_dtype,
                )

                # Prefill prompt
                prefill_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
                out = self.model(prefill_tensor, kv_cache=cache)
                curr_logits = out.logits[:, -1, :]

                while len(generated_tokens) < token_cap:
                    next_tok_tensor = sample_next_token(
                        curr_logits,
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                        top_k=settings.top_k,
                        generator=generator,
                    )
                    next_tok = int(next_tok_tensor.item())

                    if next_tok == EOS_ID:
                        finish_reason = "eos"
                        break

                    generated_tokens.append(next_tok)

                    # Check character limit if set
                    curr_continuation = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    if max_chars is not None and len(curr_continuation) >= max_chars:
                        finish_reason = "character_limit"
                        break

                    if prompt_len + len(generated_tokens) >= max_ctx:
                        finish_reason = "context_limit"
                        break

                    # Incremental single-token decoding
                    step_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
                    out = self.model(step_tensor, kv_cache=cache)
                    curr_logits = out.logits[:, -1, :]

            else:
                # 3. Uncached path (for equivalence testing)
                all_tokens = list(prompt_tokens)

                while len(generated_tokens) < token_cap:
                    input_tensor = torch.tensor([all_tokens], dtype=torch.long, device=self.device)
                    out = self.model(input_tensor)
                    curr_logits = out.logits[:, -1, :]

                    next_tok_tensor = sample_next_token(
                        curr_logits,
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                        top_k=settings.top_k,
                        generator=generator,
                    )
                    next_tok = int(next_tok_tensor.item())

                    if next_tok == EOS_ID:
                        finish_reason = "eos"
                        break

                    generated_tokens.append(next_tok)
                    all_tokens.append(next_tok)

                    curr_continuation = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    if max_chars is not None and len(curr_continuation) >= max_chars:
                        finish_reason = "character_limit"
                        break

                    if len(all_tokens) >= max_ctx:
                        finish_reason = "context_limit"
                        break

            continuation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if prompt:
                # Append continuation cleanly
                full_text = prompt + continuation
            else:
                full_text = continuation

            return GenerationResult(
                prompt=prompt,
                continuation=continuation,
                full_text=full_text,
                prompt_token_ids=prompt_tokens,
                generated_token_ids=generated_tokens,
                finish_reason=finish_reason,
                characters_generated=len(continuation),
            )


def create_generation_artifact(
    result: GenerationResult,
    checkpoint_path: Path | str,
    checkpoint_model_sha256: str,
    tokenizer_type: str,
    settings: GenerationSettings,
    training_sampling_mode: str | None = None,
    training_temperature_alpha: float | None = None,
) -> GenerationArtifact:
    """Construct a complete provenance-tracked generation artifact."""
    return GenerationArtifact(
        is_synthetic=True,
        disclaimer=(
            "SYNTHETIC MODEL OUTPUT: Generated autoregressively from random initialization "
            "by Scripture-LM. Not authentic historical scripture."
        ),
        result=result,
        checkpoint_path=str(checkpoint_path),
        checkpoint_model_sha256=checkpoint_model_sha256,
        tokenizer_type=tokenizer_type,
        training_sampling_mode=training_sampling_mode,
        training_temperature_alpha=training_temperature_alpha,
        generation_temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        seed=settings.seed,
        max_new_tokens=settings.max_new_tokens,
        max_new_characters=settings.max_new_characters,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


__all__ = [
    "GenerationArtifact",
    "GenerationResult",
    "TextGenerator",
    "create_generation_artifact",
]
