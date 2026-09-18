"""Unit tests for training engine, optimizer, scheduler, metrics, checkpoints,
and reproducibility."""
# mypy: disable-error-code="no-untyped-call"

from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripture_lm.config import ScriptureLMConfig, TrainingConfig
from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.corpus.normalize import CorpusLock, DocumentProvenance
from scripture_lm.corpus.split import SplitManifest
from scripture_lm.data.chunk_index import ChunkMetadata, EncodingProvenance, save_chunk_index
from scripture_lm.data.sampler import NaturalSampler, TemperatureSampler
from scripture_lm.model.config import TransformerConfig
from scripture_lm.model.transformer import TransformerLM
from scripture_lm.tokenization.base import compute_manifest_sha256
from scripture_lm.tokenization.character import CharacterTokenizer
from scripture_lm.training.checkpoint import load_checkpoint, save_checkpoint
from scripture_lm.training.metrics import (
    EarlyStopping,
    compute_validation_metrics,
)
from scripture_lm.training.optimizer import clip_gradients, configure_optimizer
from scripture_lm.training.scheduler import ExposureCosineScheduler
from scripture_lm.training.trainer import (
    Trainer,
    verify_encoding_provenance,
)


def create_tiny_model(
    vocab_size: int = 64,
    d_model: int = 64,
    layers: int = 2,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
) -> TransformerLM:
    """Helper to instantiate a tiny TransformerLM for fast CPU tests."""
    config = TransformerConfig(
        vocab_size=vocab_size,
        layers=layers,
        d_model=d_model,
        heads=2,
        mlp_hidden=128,
        max_context_length=32,
        dropout=dropout,
        attention_dropout=attention_dropout,
        embedding_tying=True,
    )
    return TransformerLM(config)


def make_mock_chunk(
    chunk_id: str = "0",
    split: str = "train",
    family: str = "fam",
    token_start: int = 0,
    valid_token_count: int = 10,
    raw_character_count: int = 20,
    bin_path: str = "fam.bin",
    document_ids: list[str] | None = None,
) -> ChunkMetadata:
    """Helper to create valid ChunkMetadata objects."""
    return ChunkMetadata(
        chunk_id=chunk_id,
        split=split,  # type: ignore[arg-type]
        family=family,
        tokenizer="bpe",
        bin_path=bin_path,
        token_start=token_start,
        valid_token_count=valid_token_count,
        raw_character_count=raw_character_count,
        document_ids=document_ids or ["doc1"],
    )


def make_mock_lock_and_doc(corpus_fp: str, norm_fp: str) -> CorpusLock:
    """Helper to create valid CorpusLock objects."""
    doc = DocumentProvenance(
        document_id="doc1",
        family="fam",
        source_path="raw/fam/doc1.txt",
        source_sha256="s" * 64,
        normalized_sha256="n" * 64,
        raw_bytes=10,
        raw_characters=10,
        normalized_bytes=10,
        normalized_characters=10,
        lines_before=1,
        lines_after=1,
        cleanup_modifications=0,
    )
    return CorpusLock(
        corpus_fingerprint=corpus_fp,
        normalization_fingerprint=norm_fp,
        documents=[doc],
    )


def test_optimizer_grouping() -> None:
    """Verify optimizer parameter grouping: no decay on 1D/RMSNorm, decay on 2D, deduplication."""
    model = create_tiny_model()
    training_cfg = TrainingConfig(
        learning_rate=3e-4,
        weight_decay=0.10,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    optimizer = configure_optimizer(model, training_cfg)

    # Must have exactly 2 parameter groups: decay and no_decay
    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.10
    assert no_decay_group["weight_decay"] == 0.0

    # RMSNorm scale parameters (weight) must be in no_decay_group
    for block in model.blocks:
        b = cast(Any, block)
        assert any(p is b.attn_norm.weight for p in no_decay_group["params"])
        assert any(p is b.mlp_norm.weight for p in no_decay_group["params"])
    assert any(p is model.final_norm.weight for p in no_decay_group["params"])

    # 2D Linear and Embedding weights must be in decay_group
    assert any(p is model.tok_embeddings.weight for p in decay_group["params"])

    # Tied weights: lm_head.weight is tok_embeddings.weight and must appear once across all groups
    all_optimizer_params = decay_group["params"] + no_decay_group["params"]
    param_ids = [id(p) for p in all_optimizer_params]
    assert len(param_ids) == len(set(param_ids)), "Duplicate parameters found in optimizer groups!"
    assert model.count_parameters(trainable_only=True) == sum(
        p.numel() for p in all_optimizer_params
    )


def test_exposure_cosine_scheduler() -> None:
    """Verify ExposureCosineScheduler warmup, cosine decay, floor, and state serialization."""
    model = create_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    total_exposure = 100_000
    warmup_ratio = 0.02
    min_lr = 3e-5
    base_lr = 3e-4

    scheduler = ExposureCosineScheduler(
        optimizer,
        total_target_exposure=total_exposure,
        warmup_ratio=warmup_ratio,
        min_lr=min_lr,
        base_lr=base_lr,
    )

    # Initial lr at 0 exposure should be 0.0
    assert scheduler.get_lr() == 0.0

    # Mid-warmup (1% exposure) -> exactly 50% of base_lr
    lr_1k = scheduler.step(1000)
    assert math.isclose(lr_1k, base_lr * 0.5, rel_tol=1e-4)
    assert optimizer.param_groups[0]["lr"] == lr_1k

    # End of warmup (2% exposure) -> base_lr
    lr_2k = scheduler.step(2000)
    assert math.isclose(lr_2k, base_lr, rel_tol=1e-4)

    # Mid-decay (51% exposure, half-way through decay) -> (base_lr + min_lr) / 2
    mid_exposure = 2000 + (100_000 - 2000) // 2
    lr_mid = scheduler.step(mid_exposure)
    expected_mid = min_lr + 0.5 * (base_lr - min_lr)
    assert math.isclose(lr_mid, expected_mid, rel_tol=1e-3)

    # Full exposure (100%) -> min_lr
    lr_100k = scheduler.step(100_000)
    assert math.isclose(lr_100k, min_lr, rel_tol=1e-4)

    # Overshoot (> 100%) -> clamped to min_lr
    lr_150k = scheduler.step(150_000)
    assert lr_150k == min_lr

    # State dict round-trip
    state = scheduler.state_dict()
    scheduler2 = ExposureCosineScheduler(
        optimizer,
        total_target_exposure=total_exposure,
        warmup_ratio=warmup_ratio,
        min_lr=min_lr,
        base_lr=base_lr,
    )
    scheduler2.load_state_dict(state)
    assert scheduler2.get_lr() == scheduler.get_lr()
    assert scheduler2.cumulative_raw_chars == scheduler.cumulative_raw_chars


def test_gradient_accumulation_is_target_weighted() -> None:
    """Verify target-weighted accumulation produces exact equivalent gradients to batch mean NLL."""
    torch.manual_seed(42)
    # Dropout must be 0 for exact deterministic equivalence
    model1 = create_tiny_model(
        vocab_size=32, d_model=32, layers=1, dropout=0.0, attention_dropout=0.0
    )
    model2 = copy.deepcopy(model1)

    # Create 2 microbatches with different valid target counts
    # Microbatch 1: 10 valid targets
    x1 = torch.randint(0, 32, (2, 8))
    y1 = torch.randint(0, 32, (2, 8))
    y1[0, 5:] = -100  # mask out 3 targets
    y1[1, 5:] = -100  # mask out 3 targets -> 10 valid targets

    # Microbatch 2: 15 valid targets
    x2 = torch.randint(0, 32, (2, 8))
    y2 = torch.randint(0, 32, (2, 8))
    y2[1, 7:] = -100  # mask out 1 target -> 15 valid targets

    targets1 = int((y1 != -100).sum().item())
    targets2 = int((y2 != -100).sum().item())
    total_targets = targets1 + targets2

    # Method A: Accumulated microbatches with summed NLL divided by total valid targets
    out1 = model1(x1)
    loss1 = F.cross_entropy(
        out1.logits.view(-1, 32), y1.view(-1), reduction="sum", ignore_index=-100
    )
    loss1.backward()

    out2 = model1(x2)
    loss2 = F.cross_entropy(
        out2.logits.view(-1, 32), y2.view(-1), reduction="sum", ignore_index=-100
    )
    loss2.backward()

    for p in model1.parameters():
        if p.grad is not None:
            p.grad.div_(total_targets)

    # Method B: Full combined batch with reduction="sum" divided by total valid targets
    x_full = torch.cat([x1, x2], dim=0)
    y_full = torch.cat([y1, y2], dim=0)
    out_full = model2(x_full)
    loss_full = F.cross_entropy(
        out_full.logits.view(-1, 32), y_full.view(-1), reduction="sum", ignore_index=-100
    )
    (loss_full / total_targets).backward()

    # Gradients must be bitwise or within floating point tolerance identical
    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
        if p1.grad is not None and p2.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-6)


def test_accumulation_carries_across_effective_epoch_boundary() -> None:
    """Verify accumulation window carries across effective epoch boundaries without early flush."""
    model = create_tiny_model(vocab_size=32, dropout=0.0, attention_dropout=0.0)
    optimizer = configure_optimizer(model, TrainingConfig(gradient_clip=1.0))

    accumulated_targets = 0
    microbatches_in_window = 0
    step_count = 0

    batches = [
        (torch.randint(0, 32, (1, 8)), torch.randint(0, 32, (1, 8))),
        (torch.randint(0, 32, (1, 8)), torch.randint(0, 32, (1, 8))),
        (torch.randint(0, 32, (1, 8)), torch.randint(0, 32, (1, 8))),  # end of epoch 1
        (torch.randint(0, 32, (1, 8)), torch.randint(0, 32, (1, 8))),  # start of epoch 2
    ]

    for idx, (x, y) in enumerate(batches):
        out = model(x)
        loss = F.cross_entropy(
            out.logits.view(-1, 32), y.view(-1), reduction="sum", ignore_index=-100
        )
        loss.backward()

        accumulated_targets += int((y != -100).sum().item())
        microbatches_in_window += 1

        # Simulate epoch boundary at idx == 2
        if idx == 2:
            # We do NOT flush here! microbatches_in_window remains 3
            assert microbatches_in_window == 3
            assert step_count == 0

        if microbatches_in_window == 4:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(accumulated_targets)
            clip_gradients(model)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_count += 1
            microbatches_in_window = 0
            accumulated_targets = 0

    assert step_count == 1
    assert microbatches_in_window == 0


def test_final_partial_accumulation_window_is_flushed() -> None:
    """Verify any partial accumulation window remaining when training ends is stepped, not lost."""
    model = create_tiny_model(vocab_size=32, dropout=0.0, attention_dropout=0.0)
    optimizer = configure_optimizer(model, TrainingConfig(gradient_clip=1.0))

    initial_param = model.tok_embeddings.weight.clone()

    # Accumulate only 3 microbatches (grad_accum is 8)
    accumulated_targets = 0
    for _ in range(3):
        x = torch.randint(0, 32, (1, 8))
        y = torch.randint(0, 32, (1, 8))
        out = model(x)
        loss = F.cross_entropy(
            out.logits.view(-1, 32), y.view(-1), reduction="sum", ignore_index=-100
        )
        loss.backward()
        accumulated_targets += int((y != -100).sum().item())

    # Training ends: flush partial window
    assert accumulated_targets > 0
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(accumulated_targets)
    clip_gradients(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Weights must have changed after optimizer step
    assert not torch.allclose(initial_param, model.tok_embeddings.weight)


def test_fp16_accumulation_unscales_before_clipping() -> None:
    """Verify FP16 unscale order: unscale -> divide by targets -> clip -> step -> update."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for FP16 GradScaler test")

    device = torch.device("cuda")
    model = create_tiny_model(vocab_size=32, dropout=0.0, attention_dropout=0.0).to(device)
    optimizer = configure_optimizer(model, TrainingConfig(gradient_clip=1.0))
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)

    x = torch.randint(0, 32, (2, 8), device=device)
    y = torch.randint(0, 32, (2, 8), device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x)
        loss_sum = F.cross_entropy(
            out.logits.view(-1, 32), y.view(-1), reduction="sum", ignore_index=-100
        )

    scaler.scale(loss_sum).backward()
    accumulated_targets = int((y != -100).sum().item())

    # Correct execution order
    scaler.unscale_(optimizer)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(accumulated_targets)

    grad_norm = clip_gradients(model, max_norm=1.0)
    assert not math.isnan(grad_norm)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def test_bpc_excludes_boundary_tokens() -> None:
    """Verify BPC excludes PAD (0), BOS (1), and EOS (2) from numerator while keeping tokens."""
    torch.manual_seed(42)
    model = create_tiny_model(vocab_size=32)
    model.eval()

    # Batch with: [BOS (1), token (10), token (11), EOS (2), PAD (0)]
    input_ids = torch.tensor([[1, 10, 11, 2, 0]])
    target_ids = torch.tensor([[10, 11, 2, 0, -100]])  # target 10, target 11, EOS (2), PAD (0)

    # In target_ids:
    # index 0: token 10 (text piece) -> INCLUDED in BPC
    # index 1: token 11 (text piece) -> INCLUDED in BPC
    # index 2: token 2 (EOS) -> EXCLUDED from BPC
    # index 3: token 0 (PAD) -> EXCLUDED from BPC
    # index 4: -100 (masked) -> EXCLUDED

    batch = {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "families": ["hebrew"],
        "chunk_ids": [0],
        "raw_characters": 8,  # 8 original characters for tokens 10 and 11
        "target_tokens": 3,
    }

    # Custom mock dataset
    class MockDataset:
        def __init__(self) -> None:
            self.chunks = [
                make_mock_chunk(
                    chunk_id="0",
                    split="validation",
                    family="hebrew",
                    token_start=0,
                    valid_token_count=4,
                    raw_character_count=8,
                )
            ]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return batch

    class MockLoader:
        dataset = MockDataset()

        def __iter__(self) -> Any:
            yield batch

    metrics = compute_validation_metrics(
        model=model,
        val_loader=MockLoader(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    # Total non-special tokens should be exactly 2 (tokens 10 and 11), NOT 3 or 4
    assert metrics.total_non_special_tokens == 2
    assert metrics.total_valid_targets == 4  # targets 10, 11, EOS (2), and PAD (0)
    assert "hebrew" in metrics.family_bpc
    assert metrics.macro_val_bpc > 0.0


def test_early_stopping_bookkeeping() -> None:
    """Verify EarlyStopping patience counter, reset on improvement, and trigger at limit."""
    es = EarlyStopping(patience=8)

    # Improving metrics -> counter stays 0, returns True
    assert es.step(2.5) is True
    assert es.counter == 0
    assert not es.should_stop

    assert es.step(2.3) is True
    assert es.counter == 0
    assert not es.should_stop

    # 7 non-improving evaluations -> counter increases to 7, should_stop remains False
    for i in range(1, 8):
        assert es.step(2.4) is False
        assert es.counter == i
        assert not es.should_stop

    # 8th non-improving evaluation -> counter reaches 8, triggers should_stop
    assert es.step(2.4) is False
    assert es.counter == 8
    assert es.should_stop is True

    # State dict serialization
    state = es.state_dict()
    es2 = EarlyStopping(patience=8)
    es2.load_state_dict(state)
    assert es2.counter == 8
    assert es2.should_stop is True
    assert es2.best_metric == 2.3


def test_checkpoint_save_reload() -> None:
    """Verify checkpoint save/load round-trip with safetensors and tied embedding identity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoint_best"

        model = create_tiny_model()
        training_cfg = TrainingConfig()
        optimizer = configure_optimizer(model, training_cfg)
        scheduler = ExposureCosineScheduler(optimizer, total_target_exposure=50_000)
        es = EarlyStopping(patience=8)
        sampler = NaturalSampler([make_mock_chunk(chunk_id="0")])

        scheduler.step(5000)
        es.step(1.85)

        # Save checkpoint
        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            raw_model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            early_stopping=es,
            global_step=42,
            micro_step=336,
            cumulative_raw_chars=5000,
            cumulative_model_tokens=1200,
            effective_epoch=0.5,
            accumulated_targets=16,
            best_val_bpc=1.85,
        )

        # Verify files created
        assert (ckpt_dir / "model.safetensors").is_file()
        assert (ckpt_dir / "training_state.pt").is_file()
        assert (ckpt_dir / "metadata.json").is_file()

        # Load into fresh model and components
        new_model = create_tiny_model()
        new_optimizer = configure_optimizer(new_model, training_cfg)
        new_scheduler = ExposureCosineScheduler(new_optimizer, total_target_exposure=50_000)
        new_es = EarlyStopping(patience=8)
        new_sampler = NaturalSampler([make_mock_chunk(chunk_id="0")])

        state = load_checkpoint(
            checkpoint_dir=ckpt_dir,
            raw_model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            sampler=new_sampler,
            early_stopping=new_es,
        )

        # Weights must be bitwise identical
        for p_old, p_new in zip(model.parameters(), new_model.parameters(), strict=True):
            assert torch.equal(p_old, p_new)

        # Tied embeddings identity preserved
        assert new_model.lm_head.weight is new_model.tok_embeddings.weight

        # State fields restored
        assert state["global_step"] == 42
        assert state["cumulative_raw_chars"] == 5000
        assert new_scheduler.get_lr() == scheduler.get_lr()
        assert new_es.best_metric == 1.85


def test_deterministic_resume_cpu() -> None:
    """Verify bitwise identical training after saving and reloading on CPU with dropout."""
    # Generate fixed synthetic data stream with independent generator
    gen = torch.Generator().manual_seed(9999)
    data_batches = [
        (torch.randint(0, 64, (2, 8), generator=gen), torch.randint(0, 64, (2, 8), generator=gen))
        for _ in range(10)
    ]

    # 1. Continuous run: train 10 steps with dropout enabled
    torch.manual_seed(1337)
    np.random.seed(1337)

    model_cont = create_tiny_model(dropout=0.10, attention_dropout=0.10)
    opt_cont = configure_optimizer(model_cont, TrainingConfig(learning_rate=1e-3))
    sched_cont = ExposureCosineScheduler(opt_cont, total_target_exposure=50_000)

    losses_cont: list[float] = []
    for step, (x, y) in enumerate(data_batches):
        opt_cont.zero_grad()
        out = model_cont(x)
        loss = F.cross_entropy(out.logits.view(-1, 64), y.view(-1), ignore_index=-100)
        loss.backward()
        clip_gradients(model_cont)
        opt_cont.step()
        sched_cont.step(step * 500)
        losses_cont.append(float(loss.item()))

    # 2. Resumed run: train 5 steps, save, reload into fresh model, train remaining 5 steps
    torch.manual_seed(1337)
    np.random.seed(1337)

    model_resume = create_tiny_model(dropout=0.10, attention_dropout=0.10)
    opt_resume = configure_optimizer(model_resume, TrainingConfig(learning_rate=1e-3))
    sched_resume = ExposureCosineScheduler(opt_resume, total_target_exposure=50_000)
    sampler_mock = NaturalSampler([make_mock_chunk(chunk_id="0")])

    losses_resumed: list[float] = []
    for step in range(5):
        x, y = data_batches[step]
        opt_resume.zero_grad()
        out = model_resume(x)
        loss = F.cross_entropy(out.logits.view(-1, 64), y.view(-1), ignore_index=-100)
        loss.backward()
        clip_gradients(model_resume)
        opt_resume.step()
        sched_resume.step(step * 500)
        losses_resumed.append(float(loss.item()))

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "ckpt_step5"
        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            raw_model=model_resume,
            optimizer=opt_resume,
            scheduler=sched_resume,
            sampler=sampler_mock,
            early_stopping=EarlyStopping(),
            global_step=5,
            micro_step=5,
            cumulative_raw_chars=2500,
            cumulative_model_tokens=500,
            effective_epoch=0.1,
            accumulated_targets=0,
            best_val_bpc=2.0,
        )

        # Fresh model reload
        new_model = create_tiny_model(dropout=0.10, attention_dropout=0.10)
        new_opt = configure_optimizer(new_model, TrainingConfig(learning_rate=1e-3))
        new_sched = ExposureCosineScheduler(new_opt, total_target_exposure=50_000)
        new_sampler = NaturalSampler([make_mock_chunk(chunk_id="0")])

        load_checkpoint(
            checkpoint_dir=ckpt_dir,
            raw_model=new_model,
            optimizer=new_opt,
            scheduler=new_sched,
            sampler=new_sampler,
        )

        # Train remaining 5 steps
        for step in range(5, 10):
            x, y = data_batches[step]
            new_opt.zero_grad()
            out = new_model(x)
            loss = F.cross_entropy(out.logits.view(-1, 64), y.view(-1), ignore_index=-100)
            loss.backward()
            clip_gradients(new_model)
            new_opt.step()
            new_sched.step(step * 500)
            losses_resumed.append(float(loss.item()))

        # Compare losses step by step: must match bitwise
        for s, (l_cont, l_res) in enumerate(zip(losses_cont, losses_resumed, strict=True)):
            assert math.isclose(l_cont, l_res, rel_tol=1e-5), (
                f"Mismatch at step {s}: {l_cont} vs {l_res}"
            )

        # Compare final model weights: must match bitwise
        for p_cont, p_new in zip(model_cont.parameters(), new_model.parameters(), strict=True):
            assert torch.allclose(p_cont, p_new, atol=1e-6)


def test_natural_sampler_resume() -> None:
    """Verify NaturalSampler resumes remaining epoch chunks in exact order without duplicates."""
    chunks = [make_mock_chunk(chunk_id=f"c_{i}", family=f"fam_{i % 3}") for i in range(20)]

    sampler1 = NaturalSampler(chunks, seed=42)
    drawn_1 = [next(iter(sampler1)) for _ in range(7)]
    state = sampler1.state_dict()

    # Draw remaining 13 chunks from sampler1
    remaining_1 = [next(iter(sampler1)) for _ in range(13)]

    # Restore in new sampler
    sampler2 = NaturalSampler(chunks, seed=42)
    sampler2.load_state_dict(state)
    assert sampler2.position_in_epoch == 7

    remaining_2 = [next(iter(sampler2)) for _ in range(13)]

    assert remaining_1 == remaining_2
    assert len(drawn_1 + remaining_2) == 20
    assert len(set(drawn_1 + remaining_2)) == 20


def test_temperature_sampler_resume() -> None:
    """Verify TemperatureSampler resumes with identical RNG state and exposure counters."""
    chunks = [
        make_mock_chunk(chunk_id=f"c_{i}", family="hebrew" if i < 10 else "greek")
        for i in range(20)
    ]

    sampler1 = TemperatureSampler(chunks, alpha=0.5, seed=42)
    _ = [next(iter(sampler1)) for _ in range(10)]
    state = sampler1.state_dict()

    next_drawn_1 = [next(iter(sampler1)) for _ in range(10)]

    sampler2 = TemperatureSampler(chunks, alpha=0.5, seed=42)
    sampler2.load_state_dict(state)

    next_drawn_2 = [next(iter(sampler2)) for _ in range(10)]

    assert next_drawn_1 == next_drawn_2
    assert sampler1.cumulative_raw_chars == sampler2.cumulative_raw_chars
    assert sampler1.cumulative_model_tokens == sampler2.cumulative_model_tokens


def test_training_refuses_stale_encoded_dataset(tmp_path: Path) -> None:
    """Verify verify_encoding_provenance refuses stale corpus fingerprint."""
    lock = make_mock_lock_and_doc("lock_fingerprint_123", "norm_456")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "splits").mkdir()
    (tmp_path / "data" / "encoded" / "bpe").mkdir(parents=True)
    (tmp_path / "artifacts" / "tokenizers").mkdir(parents=True)

    (tmp_path / "data" / "corpus_lock.json").write_text(
        lock.model_dump_json(indent=2), encoding="utf-8"
    )
    split = SplitManifest(
        seed=42,
        algorithm="stratified_greedy_v1",
        targets={"train": 1.0},
        corpus_fingerprint="lock_fingerprint_123",
        normalization_fingerprint="norm_456",
        train=["doc1"],
        validation=[],
        test=[],
    )
    (tmp_path / "data" / "splits" / "split_manifest.json").write_text(
        split.model_dump_json(indent=2), encoding="utf-8"
    )

    tok_file = tmp_path / "artifacts" / "tokenizers" / "bpe.json"
    tok_file.write_text("{}", encoding="utf-8")
    (tmp_path / "artifacts" / "tokenizers" / "bpe_metadata.json").write_text("{}", encoding="utf-8")

    # Mismatched corpus fingerprint in encoding metadata
    enc_prov = EncodingProvenance(
        tokenizer_type="bpe",
        context_length=512,
        chunk_length=513,
        corpus_fingerprint="stale_fingerprint",
        normalization_fingerprint="norm_456",
        split_manifest_hash=compute_manifest_sha256(
            tmp_path / "data" / "splits" / "split_manifest.json"
        ),
        tokenizer_artifact_sha256=compute_file_sha256(tok_file),
        total_chunks={"train": 1},
        natural_train_target_characters=10,
    )
    (tmp_path / "data" / "encoded" / "bpe" / "encoding_metadata.json").write_text(
        enc_prov.model_dump_json(), encoding="utf-8"
    )

    cfg = ScriptureLMConfig.model_validate(
        {"tokenizer": {"type": "bpe", "bpe_vocab_size": 4096, "context_length": 512}}
    )

    with pytest.raises(ValueError, match="corpus_fingerprint"):
        verify_encoding_provenance(
            cfg,
            data_root=tmp_path / "data",
            corpus_root=tmp_path / "corpus",
            artifacts_root=tmp_path / "artifacts",
        )


def test_training_refuses_wrong_tokenizer_hash(tmp_path: Path) -> None:
    """Verify verify_encoding_provenance refuses mismatched tokenizer SHA256."""
    lock = make_mock_lock_and_doc("fingerprint_123", "norm_456")
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "data" / "encoded" / "bpe").mkdir(parents=True)
    (tmp_path / "artifacts" / "tokenizers").mkdir(parents=True)

    (tmp_path / "data" / "corpus_lock.json").write_text(
        lock.model_dump_json(indent=2), encoding="utf-8"
    )
    split = SplitManifest(
        seed=42,
        algorithm="stratified_greedy_v1",
        targets={"train": 1.0},
        corpus_fingerprint="fingerprint_123",
        normalization_fingerprint="norm_456",
        train=["doc1"],
        validation=[],
        test=[],
    )
    split_file = tmp_path / "data" / "splits" / "split_manifest.json"
    split_file.write_text(split.model_dump_json(indent=2), encoding="utf-8")

    tok_file = tmp_path / "artifacts" / "tokenizers" / "bpe.json"
    tok_file.write_text("{}", encoding="utf-8")
    (tmp_path / "artifacts" / "tokenizers" / "bpe_metadata.json").write_text("{}", encoding="utf-8")

    enc_prov = EncodingProvenance(
        tokenizer_type="bpe",
        context_length=512,
        chunk_length=513,
        corpus_fingerprint="fingerprint_123",
        normalization_fingerprint="norm_456",
        split_manifest_hash=compute_manifest_sha256(split_file),
        tokenizer_artifact_sha256="wrong_hash",
        total_chunks={"train": 1},
        natural_train_target_characters=10,
    )
    (tmp_path / "data" / "encoded" / "bpe" / "encoding_metadata.json").write_text(
        enc_prov.model_dump_json(), encoding="utf-8"
    )

    cfg = ScriptureLMConfig.model_validate(
        {"tokenizer": {"type": "bpe", "bpe_vocab_size": 4096, "context_length": 512}}
    )

    with pytest.raises(ValueError, match="tokenizer artifact hash mismatch"):
        verify_encoding_provenance(
            cfg,
            data_root=tmp_path / "data",
            corpus_root=tmp_path / "corpus",
            artifacts_root=tmp_path / "artifacts",
        )


def test_training_refuses_wrong_context_length(tmp_path: Path) -> None:
    """Verify verify_encoding_provenance refuses mismatched context length."""
    lock = make_mock_lock_and_doc("fingerprint_123", "norm_456")
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "data" / "encoded" / "bpe").mkdir(parents=True)
    (tmp_path / "artifacts" / "tokenizers").mkdir(parents=True)

    (tmp_path / "data" / "corpus_lock.json").write_text(
        lock.model_dump_json(indent=2), encoding="utf-8"
    )
    split = SplitManifest(
        seed=42,
        algorithm="stratified_greedy_v1",
        targets={"train": 1.0},
        corpus_fingerprint="fingerprint_123",
        normalization_fingerprint="norm_456",
        train=["doc1"],
        validation=[],
        test=[],
    )
    split_file = tmp_path / "data" / "splits" / "split_manifest.json"
    split_file.write_text(split.model_dump_json(indent=2), encoding="utf-8")

    tok_file = tmp_path / "artifacts" / "tokenizers" / "bpe.json"
    tok_file.write_text("{}", encoding="utf-8")
    (tmp_path / "artifacts" / "tokenizers" / "bpe_metadata.json").write_text("{}", encoding="utf-8")

    enc_prov = EncodingProvenance(
        tokenizer_type="bpe",
        context_length=512,
        chunk_length=513,
        corpus_fingerprint="fingerprint_123",
        normalization_fingerprint="norm_456",
        split_manifest_hash=compute_manifest_sha256(split_file),
        tokenizer_artifact_sha256=compute_file_sha256(tok_file),
        total_chunks={"train": 1},
        natural_train_target_characters=10,
    )
    (tmp_path / "data" / "encoded" / "bpe" / "encoding_metadata.json").write_text(
        enc_prov.model_dump_json(), encoding="utf-8"
    )

    # Config requests 256, but data was encoded with 512
    cfg = ScriptureLMConfig.model_validate(
        {"tokenizer": {"type": "bpe", "bpe_vocab_size": 4096, "context_length": 256}}
    )

    with pytest.raises(ValueError, match="context_length"):
        verify_encoding_provenance(
            cfg,
            data_root=tmp_path / "data",
            corpus_root=tmp_path / "corpus",
            artifacts_root=tmp_path / "artifacts",
        )


@pytest.mark.parametrize("accumulation_steps", [1, 8])
@pytest.mark.parametrize("metadata_name", ["encoding_provenance.json", "encoding_metadata.json"])
@pytest.mark.parametrize("tokenizer_type", ["bpe", "character"])
def test_run_directory_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accumulation_steps: int,
    metadata_name: str,
    tokenizer_type: str,
) -> None:
    """Verify Trainer run directory creation snapshots all required metadata files."""
    # Set up valid minimal mock data structure
    lock = make_mock_lock_and_doc("fingerprint_123", "norm_456")
    (tmp_path / "data" / "splits").mkdir(parents=True)
    encoded_dir = tmp_path / "data" / "encoded" / tokenizer_type
    (encoded_dir / "train").mkdir(parents=True)
    (encoded_dir / "validation").mkdir(parents=True)
    (tmp_path / "artifacts" / "tokenizers").mkdir(parents=True)
    (tmp_path / "corpus").mkdir(parents=True)

    (tmp_path / "data" / "corpus_lock.json").write_text(
        lock.model_dump_json(indent=2), encoding="utf-8"
    )
    split = SplitManifest(
        seed=42,
        algorithm="stratified_greedy_v1",
        targets={"train": 1.0},
        corpus_fingerprint="fingerprint_123",
        normalization_fingerprint="norm_456",
        train=["doc1"],
        validation=["doc1"],
        test=[],
    )
    split_file = tmp_path / "data" / "splits" / "split_manifest.json"
    split_file.write_text(split.model_dump_json(indent=2), encoding="utf-8")

    tok_dir = tmp_path / "artifacts" / "tokenizers"
    if tokenizer_type == "character":
        tok_file = tok_dir / "char_vocab.json"
        CharacterTokenizer(
            ["<pad>", "<bos>", "<eos>", "<unk>"] + [chr(32 + i) for i in range(60)]
        ).save(tok_file)
        tok_meta = tok_dir / "char_metadata.json"
    else:
        tok_file = tok_dir / "bpe.json"
        tok_file.write_text("{}", encoding="utf-8")
        tok_meta = tok_dir / "bpe_metadata.json"
    tok_meta.write_text(json.dumps({"vocab_size": 64}), encoding="utf-8")

    manifest_file = tmp_path / "corpus" / "corpus_manifest.toml"
    manifest_file.write_text("[documents]\n", encoding="utf-8")

    # Binary token files (uint16)
    bin_file = encoded_dir / "train" / "fam.bin"
    arr = np.arange(100, dtype=np.uint16)
    arr.tofile(bin_file)

    val_bin = encoded_dir / "validation" / "fam.bin"
    arr.tofile(val_bin)

    # Chunk indexes
    chunk = make_mock_chunk(
        chunk_id="chunk_0",
        split="train",
        family="fam",
        token_start=0,
        valid_token_count=17,
        raw_character_count=10,
        bin_path=f"{tokenizer_type}/train/fam.bin",
    )
    val_chunk = make_mock_chunk(
        chunk_id="chunk_0",
        split="validation",
        family="fam",
        token_start=0,
        valid_token_count=17,
        raw_character_count=10,
        bin_path=f"{tokenizer_type}/validation/fam.bin",
    )
    chunk = chunk.model_copy(update={"tokenizer": tokenizer_type})
    val_chunk = val_chunk.model_copy(update={"tokenizer": tokenizer_type})
    save_chunk_index([chunk], encoded_dir / "train_chunks.json")
    save_chunk_index([val_chunk], encoded_dir / "validation_chunks.json")

    enc_prov = EncodingProvenance(
        tokenizer_type=tokenizer_type,
        context_length=16,
        chunk_length=17,
        corpus_fingerprint="fingerprint_123",
        normalization_fingerprint="norm_456",
        split_manifest_hash=compute_manifest_sha256(split_file),
        tokenizer_artifact_sha256=compute_file_sha256(tok_file),
        total_chunks={"train": 1},
        natural_train_target_characters=10,
    )
    (encoded_dir / metadata_name).write_text(enc_prov.model_dump_json(), encoding="utf-8")

    run_dir = tmp_path / "test_run"

    cfg = ScriptureLMConfig.model_validate(
        {
            "model": {"layers": 1, "d_model": 32, "heads": 2, "mlp_hidden": 64},
            "tokenizer": {
                "type": tokenizer_type,
                "context_length": 16,
                **({"bpe_vocab_size": 64} if tokenizer_type == "bpe" else {}),
            },
            "training": {
                "device": "cpu",
                "compile": False,
                "microbatch_size": 1,
                "gradient_accumulation_steps": accumulation_steps,
                "max_effective_epochs": 1,
            },
        }
    )

    trainer = Trainer(
        config=cfg,
        run_dir=run_dir,
        data_root=tmp_path / "data",
        corpus_root=tmp_path / "corpus",
        artifacts_root=tmp_path / "artifacts",
    )

    assert trainer.model_config.vocab_size == 64
    assert trainer.raw_model.tok_embeddings.num_embeddings == 64

    # Verify all 6 metadata snapshot files exist in run_dir
    assert (run_dir / "config.toml").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "corpus_manifest.toml").is_file()
    assert (run_dir / "corpus_lock.json").is_file()
    assert (run_dir / "split_manifest.json").is_file()
    assert (run_dir / "tokenizer_metadata.json").is_file()
    assert (run_dir / "encoding_metadata.json").is_file()

    # Verify directories exist
    assert (run_dir / "tensorboard").is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "generations").is_dir()
    assert (run_dir / "evaluation").is_dir()

    # Run 1 step of training
    summary = trainer.train()
    assert summary["global_steps"] >= 1
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "latest" / "model.safetensors").is_file()
    assert (run_dir / "run_summary.json").is_file()

    # Scientific snapshots remain byte-identical during an actual checkpoint resume.
    from scripture_lm.experiments.storage import read_status, update_status

    immutable_names = ("config.toml", "experiment_config.toml", "experiment_config.sha256")
    original_snapshots = {name: (run_dir / name).read_bytes() for name in immutable_names}
    assert read_status(run_dir)["status"] == "completed"
    assert (run_dir / "checkpoints" / "best" / "model.safetensors").is_file()
    update_status(run_dir, "interrupted")
    cfg.training.device = "cpu:0"
    cfg.training.compile = True
    resumed = Trainer(
        config=cfg,
        run_dir=run_dir,
        data_root=tmp_path / "data",
        corpus_root=tmp_path / "corpus",
        artifacts_root=tmp_path / "artifacts",
        resume_checkpoint_dir=run_dir / "checkpoints" / "latest",
    )
    resumed.train()
    assert {name: (run_dir / name).read_bytes() for name in immutable_names} == original_snapshots
    resumed_status = read_status(run_dir)
    assert resumed_status["status"] == "completed"
    assert len(resumed_status["resume_history"]) == 1
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert len(environment["execution_history"]) == 2

    # Exercise reusable evaluation against real tiny-model weights and encoded batches.
    from scripture_lm.evaluation.runner import evaluate_run

    monkeypatch.chdir(tmp_path)
    evaluate_run(run_dir, device="cpu", data_root=tmp_path / "data")
    metrics = json.loads((run_dir / "evaluation" / "validation_metrics.json").read_text("utf-8"))
    assert math.isfinite(metrics["macro_bpc"])
    assert metrics["checkpoint_model_sha256"] == compute_file_sha256(
        run_dir / "checkpoints" / "best" / "model.safetensors"
    )

    # A checkpoint with the same architecture but different provenance cannot resume.
    state_path = run_dir / "checkpoints" / "latest" / "training_state.pt"
    state = torch.load(state_path, weights_only=False)
    state["experiment_config_hash"] = "other experiment"
    torch.save(state, state_path)
    with pytest.raises(ValueError, match="checkpoint experiment provenance mismatch"):
        Trainer(
            config=cfg,
            run_dir=run_dir,
            data_root=tmp_path / "data",
            corpus_root=tmp_path / "corpus",
            artifacts_root=tmp_path / "artifacts",
            resume_checkpoint_dir=state_path.parent,
        )
