"""Comprehensive test suite for Scripture-LM Transformer architecture."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripture_lm.config import load_config
from scripture_lm.model import (
    RMSNorm,
    RotaryEmbedding,
    TransformerConfig,
    TransformerLM,
    apply_rotary_emb,
)


@pytest.fixture
def base_config() -> TransformerConfig:
    """Standard 6-layer baseline configuration matching project specification."""
    return TransformerConfig(
        vocab_size=4096,
        max_context_length=512,
        layers=6,
        d_model=256,
        heads=8,
        mlp_hidden=704,
        rms_eps=1e-5,
        rope_theta=10000.0,
        dropout=0.10,
        attention_dropout=0.10,
        linear_bias=False,
        embedding_tying=True,
    )


def test_expected_tensor_shapes(base_config: TransformerConfig) -> None:
    """Input of shape (B, T) produces logits of shape (B, T, vocab_size)."""
    model = TransformerLM(base_config)
    model.eval()

    b, t = 2, 32
    x = torch.randint(0, base_config.vocab_size, (b, t))

    with torch.no_grad():
        out = model(x)

    assert out.logits.shape == (b, t, base_config.vocab_size)
    assert out.loss is None

    # With targets
    targets = torch.randint(0, base_config.vocab_size, (b, t))
    out_with_loss = model(x, targets=targets)
    assert out_with_loss.loss is not None
    assert out_with_loss.loss.ndim == 0  # scalar
    assert not torch.isnan(out_with_loss.loss)


def test_bpe_vocab_size_works() -> None:
    """BPE vocabulary size (4096) forward pass."""
    cfg = TransformerConfig(
        vocab_size=4096,
        max_context_length=512,
        layers=2,
        d_model=64,
        heads=4,
        mlp_hidden=128,
    )
    model = TransformerLM(cfg)
    model.eval()
    x = torch.randint(0, 4096, (1, 16))
    out = model(x)
    assert out.logits.shape == (1, 16, 4096)


def test_char_vocab_size_works() -> None:
    """Character vocabulary size (150) forward pass."""
    cfg = TransformerConfig(
        vocab_size=150,
        max_context_length=2048,
        layers=2,
        d_model=64,
        heads=4,
        mlp_hidden=128,
    )
    model = TransformerLM(cfg)
    model.eval()
    x = torch.randint(0, 150, (1, 32))
    out = model(x)
    assert out.logits.shape == (1, 32, 150)


def test_causality_no_future_token_leakage(base_config: TransformerConfig) -> None:
    """Mutating future tokens must not alter logits at earlier positions."""
    model = TransformerLM(base_config)
    model.eval()

    # Two sequences identical up to position 4, differing at positions 5..7
    seq_a = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    seq_b = torch.tensor([[10, 20, 30, 40, 50, 999, 1000, 1001]])

    with torch.no_grad():
        logits_a = model(seq_a).logits
        logits_b = model(seq_b).logits

    # Positions 0..4 must produce identical logits to numerical precision
    torch.testing.assert_close(logits_a[:, :5], logits_b[:, :5])

    # Position 5 and beyond must differ
    assert not torch.allclose(logits_a[:, 5:], logits_b[:, 5:])


def test_attention_dropout_disabled_in_eval(base_config: TransformerConfig) -> None:
    """In eval mode, SDPA attention dropout must be strictly disabled."""
    model = TransformerLM(base_config)
    x = torch.randint(0, base_config.vocab_size, (2, 16))

    # In eval mode, multiple runs are deterministic
    model.eval()
    with torch.no_grad():
        out1 = model(x).logits
        out2 = model(x).logits
    torch.testing.assert_close(out1, out2)

    # In train mode with dropout > 0, multiple runs differ
    model.train()
    out_train1 = model(x).logits
    out_train2 = model(x).logits
    assert not torch.allclose(out_train1, out_train2)


def test_tied_embedding_output_identity(base_config: TransformerConfig) -> None:
    """When embedding_tying=True, lm_head.weight and tok_embeddings.weight are identical object."""
    model = TransformerLM(base_config)
    assert model.lm_head.weight is model.tok_embeddings.weight

    # Gradients accumulate into the shared parameter
    x = torch.randint(0, base_config.vocab_size, (2, 8))
    targets = torch.randint(0, base_config.vocab_size, (2, 8))
    out = model(x, targets=targets)
    assert out.loss is not None
    out.loss.backward()

    assert model.tok_embeddings.weight.grad is not None
    assert model.lm_head.weight.grad is model.tok_embeddings.weight.grad


def test_tied_weights_counted_once(base_config: TransformerConfig) -> None:
    """get_num_params must not double-count tied embedding weights."""
    model = TransformerLM(base_config)
    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    emb_count = model.tok_embeddings.weight.numel()

    assert total_params == non_emb_params + emb_count

    # Verify manual count of unique parameters matches get_num_params
    manual_unique = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    assert total_params == manual_unique


def test_rmsnorm_matches_reference_formula() -> None:
    """RMSNorm must match exact reference mathematical formula in float32."""
    dim = 256
    eps = 1e-5
    norm = RMSNorm(dim=dim, eps=eps)
    x = torch.randn(4, 16, dim)

    # Reference formula
    x_float = x.float()
    var = x_float.pow(2).mean(-1, keepdim=True)
    expected = (x_float * torch.rsqrt(var + eps)).to(x.dtype) * norm.weight

    actual = norm(x)
    torch.testing.assert_close(actual, expected)


def test_rope_relative_position_property() -> None:
    """Inner product of RoPE-rotated vectors depends strictly on relative position m - n."""
    dim = 32
    rope = RotaryEmbedding(dim=dim, max_seq_len=64, theta=10000.0)

    # Two random vectors q and k
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    # Position pair 1: m=5, n=2 (relative offset = 3)
    cos_5 = rope.cos_cached[5:6]
    sin_5 = rope.sin_cached[5:6]
    q_rot_5 = apply_rotary_emb(q, cos_5, sin_5)

    cos_2 = rope.cos_cached[2:3]
    sin_2 = rope.sin_cached[2:3]
    k_rot_2 = apply_rotary_emb(k, cos_2, sin_2)

    dot_5_2 = (q_rot_5 * k_rot_2).sum().item()

    # Position pair 2: m=15, n=12 (same relative offset = 3)
    cos_15 = rope.cos_cached[15:16]
    sin_15 = rope.sin_cached[15:16]
    q_rot_15 = apply_rotary_emb(q, cos_15, sin_15)

    cos_12 = rope.cos_cached[12:13]
    sin_12 = rope.sin_cached[12:13]
    k_rot_12 = apply_rotary_emb(k, cos_12, sin_12)

    dot_15_12 = (q_rot_15 * k_rot_12).sum().item()

    # Dot products must match
    assert dot_5_2 == pytest.approx(dot_15_12, rel=1e-5)


def test_bpe_context_limit_512() -> None:
    """BPE configuration sets max_context_length to 512."""
    app_cfg = load_config(config_path=Path("configs/bpe.toml"))
    model_cfg = TransformerConfig.from_app_config(app_cfg, vocab_size=4096)
    assert model_cfg.max_context_length == 512


def test_char_context_limit_2048() -> None:
    """Character configuration sets max_context_length to 2048."""
    app_cfg = load_config(config_path=Path("configs/char.toml"))
    model_cfg = TransformerConfig.from_app_config(app_cfg, vocab_size=150)
    assert model_cfg.max_context_length == 2048


def test_sequence_over_context_limit_rejected() -> None:
    """Sequence length T > max_context_length must be rejected with ValueError."""
    cfg = TransformerConfig(
        vocab_size=100,
        max_context_length=16,
        layers=2,
        d_model=32,
        heads=4,
        mlp_hidden=64,
    )
    model = TransformerLM(cfg)

    # 16 is fine
    x_ok = torch.randint(0, 100, (1, 16))
    _ = model(x_ok)

    # 17 exceeds context limit
    x_too_long = torch.randint(0, 100, (1, 17))
    with pytest.raises(ValueError, match="exceeds configured max_context_length"):
        model(x_too_long)


def test_invalid_head_dimension_rejected() -> None:
    """d_model not divisible by heads must be rejected."""
    with pytest.raises(ValueError, match="divisible by heads"):
        TransformerConfig(vocab_size=100, max_context_length=32, d_model=250, heads=8)


def test_rope_requires_even_head_dimension() -> None:
    """head_dim must be an even integer for pair rotation in RoPE."""
    # d_model=21, heads=7 -> head_dim=3 (odd)
    with pytest.raises(ValueError, match="even integer"):
        TransformerConfig(vocab_size=100, max_context_length=32, d_model=24, heads=8)  # 24 // 8 = 3


def test_one_batch_forward_backward(base_config: TransformerConfig) -> None:
    """Forward pass, loss computation, backward pass, and optimizer step succeed cleanly."""
    model = TransformerLM(base_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, base_config.vocab_size, (2, 16))
    y = torch.randint(0, base_config.vocab_size, (2, 16))

    out = model(x, targets=y)
    assert out.loss is not None
    out.loss.backward()

    # Verify every parameter with requires_grad has a valid gradient
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"

    optimizer.step()
    optimizer.zero_grad()


def test_no_nan_or_inf(base_config: TransformerConfig) -> None:
    """Model forward pass does not produce NaNs or Infs."""
    model = TransformerLM(base_config)
    x = torch.randint(0, base_config.vocab_size, (2, 32))
    targets = torch.randint(0, base_config.vocab_size, (2, 32))

    out = model(x, targets=targets)
    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()
    assert out.loss is not None
    assert not torch.isnan(out.loss)
    assert not torch.isinf(out.loss)


def test_parameter_count_report(base_config: TransformerConfig) -> None:
    """Parameter summary returns structured architectural breakdown."""
    model = TransformerLM(base_config)
    summary = model.parameter_summary()

    assert "total_parameters" in summary
    assert "non_embedding_parameters" in summary
    assert "embedding_parameters" in summary
    assert summary["layers"] == 6
    assert summary["d_model"] == 256
    assert summary["heads"] == 8
    assert summary["head_dim"] == 32
    assert summary["mlp_hidden"] == 704
    assert summary["vocab_size"] == 4096
    assert summary["max_context_length"] == 512
    assert summary["embedding_tying"] is True

    # 6 layers, d_model=256, vocab=4096 is approximately 5.8M parameters
    total = summary["total_parameters"]
    assert 5_000_000 < total < 7_000_000


def test_tiny_model_can_overfit_one_batch() -> None:
    """Diagnostic test: tiny model trains on a fixed batch, driving loss down rapidly."""
    cfg = TransformerConfig(
        vocab_size=32,
        max_context_length=16,
        layers=2,
        d_model=64,
        heads=4,
        mlp_hidden=128,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = TransformerLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    torch.manual_seed(42)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))

    # Initial loss
    out = model(x, targets=y)
    assert out.loss is not None
    initial_loss = out.loss.item()

    # Train for 80 steps
    for _ in range(80):
        optimizer.zero_grad()
        out = model(x, targets=y)
        assert out.loss is not None
        out.loss.backward()
        optimizer.step()

    final_out = model(x, targets=y)
    assert final_out.loss is not None
    final_loss = final_out.loss.item()

    # Loss must drop by at least 90% and reach low value (< 0.20)
    assert final_loss < initial_loss * 0.10
    assert final_loss < 0.20
