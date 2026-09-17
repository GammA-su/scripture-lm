"""Unit tests for generation engine, KV cache, sampling, and synthetic provenance."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripture_lm.evaluation.generation_suite import (
    GenerationResult,
    GenerationSettings,
)
from scripture_lm.generation.generate import (
    TextGenerator,
    create_generation_artifact,
)
from scripture_lm.generation.kv_cache import KVCache
from scripture_lm.generation.sampler import (
    sample_next_token,
    validate_sampling_parameters,
)
from scripture_lm.model.config import TransformerConfig
from scripture_lm.model.transformer import TransformerLM
from scripture_lm.tokenization.base import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    BaseTokenizer,
)


class DummyTokenizer(BaseTokenizer):
    """Simple character-like mock tokenizer for generation testing."""

    def __init__(self) -> None:
        self.char2id = {"a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9}
        self.id2char = {v: k for k, v in self.char2id.items()}

    @property
    def vocab_size(self) -> int:
        return 32

    @property
    def tokenizer_type(self) -> str:
        return "character"

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for char in text:
            ids.append(self.char2id.get(char, 3))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        chars: list[str] = []
        for i in ids:
            if skip_special_tokens and i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            chars.append(self.id2char.get(i, "?"))
        return "".join(chars)

    def id_to_token(self, token_id: int) -> str | None:
        return self.id2char.get(token_id)

    def token_to_id(self, token: str) -> int | None:
        return self.char2id.get(token)

    def save(self, path: Path | str) -> None:
        pass


def make_tiny_model(vocab_size: int = 32, max_context_length: int = 64) -> TransformerLM:
    """Instantiate a tiny deterministic TransformerLM for generation unit tests."""
    torch.manual_seed(42)
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=64,
        layers=2,
        heads=4,
        mlp_hidden=128,
        max_context_length=max_context_length,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = TransformerLM(cfg)
    model.eval()
    return model


# =========================================================================
# 1. KV Cache Invariants & Masking Rules
# =========================================================================


def test_cache_prefill_uses_causal_attention() -> None:
    """Verify that prefilling an empty cache allows t >= 1 and runs causal attention."""
    model = make_tiny_model()
    cache = KVCache(
        num_layers=model.config.layers,
        batch_size=1,
        heads=model.config.heads,
        head_dim=model.config.head_dim,
        max_context_length=model.config.max_context_length,
        device="cpu",
        dtype=torch.float32,
    )

    tokens = torch.tensor([[BOS_ID, 4, 5, 6]])
    out = model(tokens, kv_cache=cache)
    assert out.logits.shape == (1, 4, model.config.vocab_size)
    assert cache.seq_len == 4


def test_nonempty_cache_rejects_multi_token_input() -> None:
    """Verify that a populated cache strictly rejects t > 1 to eliminate causal masking bugs."""
    model = make_tiny_model()
    cache = KVCache(
        num_layers=model.config.layers,
        batch_size=1,
        heads=model.config.heads,
        head_dim=model.config.head_dim,
        max_context_length=model.config.max_context_length,
        device="cpu",
        dtype=torch.float32,
    )

    # 1. Prefill initial tokens
    model(torch.tensor([[BOS_ID, 4]]), kv_cache=cache)
    assert cache.seq_len == 2

    # 2. Attempt multi-token decode step on populated cache -> must raise ValueError
    multi_step_tokens = torch.tensor([[5, 6]])
    with pytest.raises(
        ValueError,
        match="Non-empty KV cache currently supports single-token decoding only",
    ):
        model(multi_step_tokens, kv_cache=cache)


def test_nonempty_cache_single_token_matches_full_forward() -> None:
    """Verify multi-step cached logits match uncached full-forward logits within tolerance."""
    model = make_tiny_model()
    tokens = [BOS_ID, 4, 5, 6, 7, 8]  # 6 tokens: A B C D E F

    # Uncached full forward pass
    with torch.inference_mode():
        full_tensor = torch.tensor([tokens], dtype=torch.long)
        full_out = model(full_tensor)
        full_logits = full_out.logits[0]  # shape (6, vocab_size)

        # Cached run: prefill first 3 tokens (A, B, C)
        cache = KVCache(
            num_layers=model.config.layers,
            batch_size=1,
            heads=model.config.heads,
            head_dim=model.config.head_dim,
            max_context_length=model.config.max_context_length,
            device="cpu",
            dtype=torch.float32,
        )

        prefill_tensor = torch.tensor([[tokens[0], tokens[1], tokens[2]]], dtype=torch.long)
        prefill_out = model(prefill_tensor, kv_cache=cache)
        assert cache.seq_len == 3

        # Prefill logits for position 2 (after token C) must match full forward position 2
        assert torch.allclose(prefill_out.logits[0, 2], full_logits[2], atol=1e-5)

        # Incremental token D (index 3)
        d_out = model(torch.tensor([[tokens[3]]]), kv_cache=cache)
        assert cache.seq_len == 4
        assert torch.allclose(d_out.logits[0, 0], full_logits[3], atol=1e-5)

        # Incremental token E (index 4)
        e_out = model(torch.tensor([[tokens[4]]]), kv_cache=cache)
        assert cache.seq_len == 5
        assert torch.allclose(e_out.logits[0, 0], full_logits[4], atol=1e-5)

        # Incremental token F (index 5)
        f_out = model(torch.tensor([[tokens[5]]]), kv_cache=cache)
        assert cache.seq_len == 6
        assert torch.allclose(f_out.logits[0, 0], full_logits[5], atol=1e-5)


def test_kv_cache_rejects_dtype_mismatch() -> None:
    """Verify KVCache rejects keys/values with mismatched dtype."""
    cache = KVCache(
        num_layers=2,
        batch_size=1,
        heads=4,
        head_dim=16,
        max_context_length=32,
        dtype=torch.float32,
    )
    k_fp16 = torch.zeros((1, 4, 1, 16), dtype=torch.float16)
    v_fp16 = torch.zeros((1, 4, 1, 16), dtype=torch.float16)

    with pytest.raises(ValueError, match="dtype mismatch"):
        cache.update(0, k_fp16, v_fp16)


def test_kv_cache_rejects_device_mismatch() -> None:
    """Verify KVCache rejects keys/values on a mismatched device."""
    cache = KVCache(
        num_layers=2,
        batch_size=1,
        heads=4,
        head_dim=16,
        max_context_length=32,
        device="cpu",
    )
    # Simulate device mismatch using an explicit target if CUDA available, otherwise mock check
    if torch.cuda.is_available():
        k_cuda = torch.zeros((1, 4, 1, 16), device="cuda")
        v_cuda = torch.zeros((1, 4, 1, 16), device="cuda")
        with pytest.raises(ValueError, match="device mismatch"):
            cache.update(0, k_cuda, v_cuda)
    else:
        # Check that validation logic checks k.device == cache.device
        cache.device = torch.device("meta")
        k_cpu = torch.zeros((1, 4, 1, 16), device="cpu")
        v_cpu = torch.zeros((1, 4, 1, 16), device="cpu")
        with pytest.raises(ValueError, match="device mismatch"):
            cache.update(0, k_cpu, v_cpu)


def test_kv_cache_rejects_context_overflow() -> None:
    """Verify KVCache raises ValueError when context capacity is exceeded."""
    cache = KVCache(
        num_layers=1,
        batch_size=1,
        heads=2,
        head_dim=8,
        max_context_length=4,
    )
    # Fill cache up to max_context_length (4 tokens)
    k = torch.zeros((1, 2, 4, 8))
    v = torch.zeros((1, 2, 4, 8))
    cache.update(0, k, v)
    assert cache.seq_len == 4

    # Adding 1 more token exceeds 4
    k_extra = torch.zeros((1, 2, 1, 8))
    v_extra = torch.zeros((1, 2, 1, 8))
    with pytest.raises(ValueError, match="context overflow"):
        cache.update(0, k_extra, v_extra)


# =========================================================================
# 2. Sampler Invariants
# =========================================================================


def test_generation_never_samples_pad_token() -> None:
    """Verify <pad> (PAD_ID) is masked to -inf and can never be sampled even with high logit."""
    # Huge positive logit for PAD_ID (0)
    logits = torch.zeros(32)
    logits[PAD_ID] = 100.0
    logits[4] = 1.0  # Token 4 has much lower score

    # Greedy sampling
    tok_greedy = sample_next_token(logits, temperature=0.0)
    assert int(tok_greedy.item()) != PAD_ID
    assert int(tok_greedy.item()) == 4

    # Temperature sampling
    gen = torch.Generator().manual_seed(42)
    tok_sample = sample_next_token(logits, temperature=1.0, top_p=1.0, generator=gen)
    assert int(tok_sample.item()) != PAD_ID


def test_sampling_validation_errors() -> None:
    """Verify invalid sampling parameters raise ValueError."""
    with pytest.raises(ValueError, match="temperature must be >= 0.0"):
        validate_sampling_parameters(temperature=-0.1, top_p=0.95, top_k=None)

    with pytest.raises(ValueError, match="top_p must be in"):
        validate_sampling_parameters(temperature=0.8, top_p=0.0, top_k=None)

    with pytest.raises(ValueError, match="top_p must be in"):
        validate_sampling_parameters(temperature=0.8, top_p=1.5, top_k=None)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        validate_sampling_parameters(temperature=0.8, top_p=0.95, top_k=0)


def test_greedy_decoding_temperature_zero() -> None:
    """Verify temperature == 0.0 strictly performs greedy argmax decoding."""
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.5, 10.0, 1.0])
    token = sample_next_token(logits, temperature=0.0)
    assert int(token.item()) == 5  # Index 5 has highest logit 10.0


def test_sampling_rng_with_non_degenerate_logits() -> None:
    """Verify deterministic replay with identical seed and variation across seeds."""
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0])
    gen1 = torch.Generator().manual_seed(100)
    gen2 = torch.Generator().manual_seed(100)

    draws1 = [
        int(sample_next_token(logits, temperature=1.0, top_p=1.0, generator=gen1).item())
        for _ in range(10)
    ]
    draws2 = [
        int(sample_next_token(logits, temperature=1.0, top_p=1.0, generator=gen2).item())
        for _ in range(10)
    ]
    assert draws1 == draws2

    # Different seed yields different random sequence
    gen3 = torch.Generator().manual_seed(999)
    draws3 = [
        int(sample_next_token(logits, temperature=1.0, top_p=1.0, generator=gen3).item())
        for _ in range(10)
    ]
    assert draws1 != draws3


# =========================================================================
# 3. Autoregressive TextGenerator Tests
# =========================================================================


def test_same_checkpoint_settings_seed_gives_same_output() -> None:
    """Verify identical model, settings, and seed produce bitwise identical token sequences."""
    model = make_tiny_model()
    tok = DummyTokenizer()
    generator = TextGenerator(model, tok)

    settings = GenerationSettings(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=15,
        seed=123,
    )

    res1 = generator.generate("ab", settings)
    res2 = generator.generate("ab", settings)

    assert res1.generated_token_ids == res2.generated_token_ids
    assert res1.continuation == res2.continuation
    assert res1.full_text == res2.full_text


def test_eos_stopping_works() -> None:
    """Verify generation halts immediately upon emitting <eos>."""
    model = make_tiny_model()
    tok = DummyTokenizer()

    # Untie lm_head and bias towards EOS_ID so input embeddings remain valid
    model.lm_head.weight = torch.nn.Parameter(torch.zeros_like(model.lm_head.weight))
    model.lm_head.weight.data[EOS_ID] = 10.0

    generator = TextGenerator(model, tok)
    settings = GenerationSettings(temperature=0.0, max_new_tokens=50, seed=42)

    res = generator.generate("a", settings)
    assert res.finish_reason == "eos"
    # Immediately emitted EOS on first generation step -> 0 continued text tokens
    assert len(res.generated_token_ids) == 0


def test_max_new_characters_limit() -> None:
    """Verify generation stops when continuation reaches max_new_characters."""
    model = make_tiny_model()
    tok = DummyTokenizer()
    generator = TextGenerator(model, tok)

    settings = GenerationSettings(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        max_new_characters=5,
        seed=42,
    )

    res = generator.generate("a", settings)
    assert res.characters_generated >= 5
    assert res.finish_reason == "character_limit"


def test_max_new_tokens_limit() -> None:
    """Verify generation halts when hitting max_new_tokens safety bound."""
    model = make_tiny_model()
    tok = DummyTokenizer()
    generator = TextGenerator(model, tok)

    settings = GenerationSettings(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=8,
        max_new_characters=1000,
        seed=42,
    )

    res = generator.generate("a", settings)
    assert len(res.generated_token_ids) == 8
    assert res.finish_reason == "length"


def test_prompt_encoding_both_tokenizers() -> None:
    """Verify prompt encoding: empty prompt uses [BOS], non-empty encodes correctly."""
    tok = DummyTokenizer()
    model = make_tiny_model()
    generator = TextGenerator(model, tok)

    # 1. Empty prompt -> prompt_token_ids == [BOS_ID]
    settings = GenerationSettings(max_new_tokens=3, seed=1)
    res_empty = generator.generate("", settings)
    assert res_empty.prompt == ""
    assert res_empty.prompt_token_ids == [BOS_ID]

    # 2. Non-empty prompt
    res_text = generator.generate("abc", settings)
    assert res_text.prompt == "abc"
    assert res_text.prompt_token_ids[0] == BOS_ID
    assert res_text.prompt_token_ids[1:] == [4, 5, 6]


def test_generation_artifact_provenance() -> None:
    """Verify GenerationArtifact records synthetic labels, provenance, and settings."""
    res = GenerationResult(
        prompt="And the prophet said",
        continuation="unto them",
        full_text="And the prophet said unto them",
        prompt_token_ids=[1, 10, 20],
        generated_token_ids=[30, 40],
        finish_reason="eos",
        characters_generated=9,
    )
    settings = GenerationSettings(
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_new_tokens=64,
        max_new_characters=1024,
        seed=777,
    )

    artifact = create_generation_artifact(
        result=res,
        checkpoint_path="runs/bpe-natural/checkpoints/best/model.safetensors",
        checkpoint_model_sha256="deadbeef" * 8,
        tokenizer_type="bpe",
        settings=settings,
        training_sampling_mode="natural",
    )

    assert artifact.is_synthetic is True
    assert "SYNTHETIC MODEL OUTPUT" in artifact.disclaimer
    assert artifact.checkpoint_model_sha256 == "deadbeef" * 8
    assert artifact.training_sampling_mode == "natural"
    assert artifact.generation_temperature == 0.8
    assert artifact.top_p == 0.95
    assert artifact.top_k == 50
    assert artifact.seed == 777
    assert artifact.max_new_characters == 1024
    assert artifact.result.finish_reason == "eos"
