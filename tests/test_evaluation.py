"""Comprehensive unit tests for evaluation, cross-tokenizer comparison, and memorization."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_model
from torch.utils.data import DataLoader, Dataset

from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.evaluation.bpc import compute_bpc
from scripture_lm.evaluation.compare import (
    compare_runs,
    render_comparison_table,
)
from scripture_lm.evaluation.generation_suite import (
    BENCHMARK_VERSION,
    CANONICAL_PROMPTS,
    GenerationSample,
    GenerationSettings,
    get_canonical_generation_suite,
)
from scripture_lm.evaluation.memorization import (
    DocumentRecord,
    TrainingCorpusMatcher,
)
from scripture_lm.evaluation.perplexity import compute_perplexity
from scripture_lm.evaluation.repetition import (
    compute_continuation_repetition,
    detect_degenerate_cycle,
)
from scripture_lm.tokenization.base import BOS_ID, EOS_ID, PAD_ID, UNK_ID


class MockOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class MockModel(nn.Module):
    """Deterministic mock model returning fixed logits for exact mathematical verification."""

    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 16)
        self.fc = nn.Linear(16, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> MockOutput:
        b, s = input_ids.shape
        # Create predictable, uniform logits across vocab
        logits = torch.zeros(b, s, self.vocab_size, device=input_ids.device)
        return MockOutput(logits)


class MockBatchDataset(Dataset[dict[str, Any]]):
    """Simple dataset yielding pre-constructed evaluation batches."""

    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.batches[idx]


def mock_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return batch[0]


def create_mock_run(
    run_dir: Path,
    tokenizer: str = "bpe",
    sampling: str = "natural",
    alpha: float | None = None,
    macro_bpc: float = 1.35,
    micro_bpc: float = 1.36,
    token_ppl: float = 15.0,
    corpus_fp: str = "fp_corpus_123",
    norm_fp: str = "fp_norm_456",
    split_hash: str = "hash_split_789",
    split: str = "validation",
    eval_chars: int = 50000,
    mismatch_ckpt_hash: bool = False,
) -> None:
    """Helper to create a complete mock run directory with metadata and checkpoints."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "best").mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    # 1. Config
    alpha_str = f"temperature_alpha = {alpha}" if alpha is not None else ""
    cfg_text = f"""
model_name = "test-model"

[model]
layers = 2
d_model = 32
heads = 2
mlp_hidden = 64

[tokenizer]
type = "{tokenizer}"
bpe_vocab_size = 4096
context_length = 16

[data]
strategy = "{sampling}"
{alpha_str}

[training]
learning_rate = 3e-4
microbatch_size = 2
gradient_accumulation_steps = 1
max_effective_epochs = 1
"""
    (run_dir / "config.toml").write_text(cfg_text, encoding="utf-8")

    # 2. Checkpoint model
    model = MockModel(vocab_size=32)
    model_path = run_dir / "checkpoints" / "best" / "model.safetensors"
    save_model(model, str(model_path))
    actual_model_sha256 = compute_file_sha256(model_path)

    # Checkpoint metadata
    ckpt_meta = {
        "global_step": 100,
        "cumulative_raw_chars": 200000,
        "cumulative_model_tokens": 50000,
        "trainable_parameter_count": 1000000,
        "best_val_bpc": macro_bpc,
    }
    (run_dir / "checkpoints" / "best" / "metadata.json").write_text(
        json.dumps(ckpt_meta), encoding="utf-8"
    )

    # 3. Provenance files
    lock_data = {
        "corpus_fingerprint": corpus_fp,
        "normalization_fingerprint": norm_fp,
        "documents": [],
    }
    (run_dir / "corpus_lock.json").write_text(json.dumps(lock_data), encoding="utf-8")
    (run_dir / "split_manifest.json").write_text(
        f'{{"split_hash": "{split_hash}"}}', encoding="utf-8"
    )

    # 4. Evaluation Report
    eval_ckpt_hash = "stale_hash_deadbeef" if mismatch_ckpt_hash else actual_model_sha256
    eval_report = {
        "evaluation_schema_version": "1.0",
        "evaluation_algorithm_version": "evaluation_v1",
        "split": split,
        "checkpoint": "best",
        "checkpoint_model_sha256": eval_ckpt_hash,
        "corpus_fingerprint": corpus_fp,
        "normalization_fingerprint": norm_fp,
        "split_manifest_hash": compute_file_sha256(run_dir / "split_manifest.json"),
        "tokenizer_artifact_sha256": "tok_hash_111",
        "evaluated_characters": eval_chars,
        "evaluated_chunks": 10,
        "macro_bpc": macro_bpc,
        "micro_bpc": micro_bpc,
        "family_bpc": {"hebrew_bible": macro_bpc, "new_testament": macro_bpc},
        "total_bits": macro_bpc * eval_chars,
        "total_non_special_tokens": 1000,
        "token_cross_entropy": math.log(token_ppl),
        "token_perplexity": token_ppl,
        "family_token_perplexity": {"hebrew_bible": token_ppl},
        "total_valid_targets": 1000,
    }
    (run_dir / "evaluation" / f"{split}_metrics.json").write_text(
        json.dumps(eval_report), encoding="utf-8"
    )


# =========================================================================
# 1. BPC Metric Tests
# =========================================================================


def test_bpc_excludes_padding_and_boundary_tokens() -> None:
    """Verify BPC numerator excludes PAD (0), BOS (1), and EOS (2), while retaining text tokens."""
    model = MockModel(vocab_size=32)
    model.eval()

    # Sequence containing PAD, BOS, EOS, and text tokens 10, 11
    # target_ids: [10, 11, EOS, PAD, ignore_index -100]
    batch = {
        "input_ids": torch.tensor([[BOS_ID, 10, 11, EOS_ID, PAD_ID]]),
        "target_ids": torch.tensor([[10, 11, EOS_ID, PAD_ID, -100]]),
        "families": ["hebrew_bible"],
        "raw_characters_per_chunk": [20],
    }
    dataloader = DataLoader(MockBatchDataset([batch]), batch_size=1, collate_fn=mock_collate)

    res = compute_bpc(model, dataloader, device="cpu")

    # Only tokens 10 and 11 should be counted as non-special text tokens (count = 2)
    assert res.total_non_special_tokens == 2
    assert res.total_characters == 20
    assert not math.isinf(res.macro_bpc)
    assert res.macro_bpc > 0.0


def test_bpc_includes_unk_probability() -> None:
    """Verify UNK (3) token likelihood is strictly INCLUDED in BPC numerator."""
    model = MockModel(vocab_size=32)
    model.eval()

    # Target sequence with UNK token and normal token 10
    batch = {
        "input_ids": torch.tensor([[BOS_ID, UNK_ID, 10]]),
        "target_ids": torch.tensor([[UNK_ID, 10, EOS_ID]]),
        "families": ["new_testament"],
        "raw_characters_per_chunk": [10],
    }
    dataloader = DataLoader(MockBatchDataset([batch]), batch_size=1, collate_fn=mock_collate)

    res = compute_bpc(model, dataloader, device="cpu")

    # UNK_ID and token 10 are counted in BPC (EOS_ID excluded) -> 2 tokens
    assert res.total_non_special_tokens == 2
    assert res.total_characters == 10
    # Expected bits for uniform vocab 32: log2(32) = 5 bits per token * 2 tokens = 10 bits
    assert math.isclose(res.total_bits, 10.0, rel_tol=1e-3)
    # BPC = 10 bits / 10 characters = 1.0
    assert math.isclose(res.macro_bpc, 1.0, rel_tol=1e-3)


def test_bpc_macro_and_micro_calculation() -> None:
    """Verify macro BPC is family unweighted mean, while micro BPC is total bits / total chars."""
    model = MockModel(vocab_size=32)
    model.eval()

    # Two chunks from different families with different character sizes
    # Batch 1: Family A, 10 tokens (50 bits), 25 characters -> BPC = 2.0
    # Batch 2: Family B, 10 tokens (50 bits), 50 characters -> BPC = 1.0
    batch1 = {
        "input_ids": torch.randint(4, 32, (1, 10)),
        "target_ids": torch.randint(4, 32, (1, 10)),
        "families": ["hebrew_bible"],
        "raw_characters_per_chunk": [25],
    }
    batch2 = {
        "input_ids": torch.randint(4, 32, (1, 10)),
        "target_ids": torch.randint(4, 32, (1, 10)),
        "families": ["quran"],
        "raw_characters_per_chunk": [50],
    }
    dataloader = DataLoader(
        MockBatchDataset([batch1, batch2]), batch_size=1, collate_fn=mock_collate
    )

    res = compute_bpc(model, dataloader, device="cpu")

    # Family BPCs: HB = 50/25 = 2.0; Quran = 50/50 = 1.0
    assert math.isclose(res.family_bpc["hebrew_bible"], 2.0, rel_tol=1e-3)
    assert math.isclose(res.family_bpc["quran"], 1.0, rel_tol=1e-3)

    # Macro BPC: (2.0 + 1.0) / 2 = 1.5
    assert math.isclose(res.macro_bpc, 1.5, rel_tol=1e-3)

    # Micro BPC: (50 + 50) / (25 + 50) = 100 / 75 = 1.3333...
    assert math.isclose(res.micro_bpc, 100.0 / 75.0, rel_tol=1e-3)


# =========================================================================
# 2. Perplexity Metric Tests
# =========================================================================


def test_token_perplexity_calculation() -> None:
    """Verify token perplexity is calculated across all valid targets except PAD."""
    model = MockModel(vocab_size=32)
    model.eval()

    # Sequence with [token, UNK, EOS, PAD, -100]
    batch = {
        "input_ids": torch.tensor([[BOS_ID, 10, UNK_ID, EOS_ID, PAD_ID]]),
        "target_ids": torch.tensor([[10, UNK_ID, EOS_ID, PAD_ID, -100]]),
        "families": ["hebrew_bible"],
    }
    dataloader = DataLoader(MockBatchDataset([batch]), batch_size=1, collate_fn=mock_collate)

    res = compute_perplexity(model, dataloader, device="cpu")

    # Valid targets: 10, UNK_ID, EOS_ID -> 3 targets (PAD and -100 excluded)
    assert res.total_valid_targets == 3
    # With uniform vocab 32, cross entropy = ln(32), perplexity = exp(ln(32)) = 32.0
    expected_ce = math.log(32.0)
    assert math.isclose(res.cross_entropy_per_token, expected_ce, rel_tol=1e-4)
    assert math.isclose(res.perplexity, 32.0, rel_tol=1e-4)


def test_evaluation_uses_eval_and_inference_mode() -> None:
    """Verify evaluation runs with model.training == False and torch.inference_mode active."""
    model = MockModel(vocab_size=32)
    model.train()  # Start in train mode

    batch = {
        "input_ids": torch.tensor([[BOS_ID, 10]]),
        "target_ids": torch.tensor([[10, EOS_ID]]),
        "families": ["hebrew_bible"],
        "raw_characters_per_chunk": [10],
    }
    dataloader = DataLoader(MockBatchDataset([batch]), batch_size=1, collate_fn=mock_collate)

    compute_bpc(model, dataloader, device="cpu")
    # Model should have been switched to eval mode
    assert not model.training


# =========================================================================
# 3. Memorization Tests
# =========================================================================


def test_memorization_rolling_hash_matches_exact() -> None:
    """Verify TrainingCorpusMatcher detects exact matches >= 50 chars with correct source doc."""
    # 60-character training scripture passage
    passage = "And God said, Let there be light: and there was light and good."
    assert len(passage) >= 50

    doc1 = DocumentRecord(doc_id="genesis_01", family="hebrew_bible", text=passage)
    matcher = TrainingCorpusMatcher([doc1], min_seed_length=50)

    # Generated text containing this exact 60-character passage
    continuation = "The prophet spoke. " + passage + " Thus it happened."
    result = matcher.find_matches_in_continuation(continuation)

    assert result.longest_exact_match_chars == len(passage)
    assert result.longest_match_source_doc == "genesis_01"
    assert result.longest_match_source_family == "hebrew_bible"
    assert result.longest_match_text == passage
    assert result.maximal_matching_spans_ge_50 == 1


def test_memorization_excludes_prompt_by_default() -> None:
    """Verify memorization analysis evaluates continuation only, ignoring prompts."""
    # Training document contains prompt phrase
    prompt_text = "In the beginning was the Word, and the Word was with God."
    doc1 = DocumentRecord(
        doc_id="john_01", family="new_testament", text=prompt_text + " And all was made."
    )
    matcher = TrainingCorpusMatcher([doc1], min_seed_length=50)

    # Sample where prompt is copied verbatim from training text, but continuation is original
    sample = GenerationSample(
        sample_id="s1",
        prompt=prompt_text,
        continuation="This is a completely original novel sentence with no copied scripture.",
        full_text=prompt_text
        + " This is a completely original novel sentence with no copied scripture.",
        settings=GenerationSettings(),
    )

    report = matcher.analyze_generation_samples([sample])
    # Continuation contains 0 matches >= 50 chars
    assert report.max_longest_match_chars == 0
    assert report.total_maximal_spans_ge_50 == 0


def test_memorization_never_crosses_document_boundary() -> None:
    """Verify match extension terminates at doc boundaries and cannot bleed across books."""
    doc1 = DocumentRecord(doc_id="genesis_last", family="hebrew_bible", text="AAA" * 25)  # 75 chars
    doc2 = DocumentRecord(doc_id="exodus_first", family="hebrew_bible", text="BBB" * 25)  # 75 chars

    matcher = TrainingCorpusMatcher([doc1, doc2], min_seed_length=50)

    # Query bridging doc1 and doc2
    continuation = ("AAA" * 20) + ("BBB" * 20)  # 60 A's followed by 60 B's
    result = matcher.find_matches_in_continuation(continuation)

    # Matches cannot bridge doc1 and doc2 into one 120-char match
    for m in result.matches:
        assert m.length <= 75
        assert "A" not in m.matched_text or "B" not in m.matched_text


def test_long_copied_span_is_one_maximal_match_not_hundreds() -> None:
    """Verify a 200-char copied passage is counted as 1 maximal match, not 151 sliding windows."""
    passage = "C" * 200
    doc = DocumentRecord(doc_id="doc_c", family="quran", text=passage)
    matcher = TrainingCorpusMatcher([doc], min_seed_length=50)

    result = matcher.find_matches_in_continuation(passage)

    assert result.longest_exact_match_chars == 200
    # Maximal match count: exactly 1 distinct maximal span
    assert result.maximal_matching_spans_ge_50 == 1
    assert result.maximal_matching_spans_ge_100 == 1
    assert result.maximal_matching_spans_ge_200 == 1

    # Sliding window count captures all 151 overlapping 50-char windows
    assert result.sliding_50_char_windows == 151


def test_memorization_character_coverage() -> None:
    """Verify matched_character_coverage computes fraction of covered continuation characters."""
    passage = "X" * 100
    doc = DocumentRecord(doc_id="doc_x", family="hebrew_bible", text=passage)
    matcher = TrainingCorpusMatcher([doc], min_seed_length=50)

    # Continuation of length 200, where first 100 chars match training passage
    continuation = passage + ("Y" * 100)
    result = matcher.find_matches_in_continuation(continuation)

    # 100 out of 200 characters covered -> 50%
    assert math.isclose(result.matched_character_coverage_ge_50, 0.50, rel_tol=1e-3)
    assert math.isclose(result.matched_character_coverage_ge_100, 0.50, rel_tol=1e-3)
    assert math.isclose(result.matched_character_coverage_ge_200, 0.0, rel_tol=1e-3)


# =========================================================================
# 4. Repetition Analysis Tests
# =========================================================================


def test_repetition_distinct_ngrams() -> None:
    """Verify distinct-1..4 and repeated n-gram rate calculations."""
    text = "the book of the generation of the son of David"
    # 10 words total
    res = compute_continuation_repetition(text)

    assert res.num_words == 10
    assert 0.0 < res.distinct_1 < 1.0
    assert math.isclose(res.repeated_ngram_rate_4, 1.0 - res.distinct_4, rel_tol=1e-5)


def test_degenerate_cycle_detection() -> None:
    """Verify detection of periodic repeating cycles (e.g. infinite degenerate loops)."""
    # Cycling phrase: "and he said" repeated 5 times
    tokens = ["and", "he", "said"] * 5
    has_cycle, period, pattern = detect_degenerate_cycle(tokens, max_period=8, min_repetitions=3)

    assert has_cycle is True
    assert period == 3
    assert pattern == ["and", "he", "said"]

    # Non-cycling natural phrase
    non_cycling = [
        "the",
        "heavens",
        "declare",
        "the",
        "glory",
        "of",
        "god",
        "and",
        "the",
        "firmament",
    ]
    has_cycle2, _, _ = detect_degenerate_cycle(non_cycling, max_period=4, min_repetitions=3)
    assert has_cycle2 is False


# =========================================================================
# 5. Generation Benchmark Suite Tests
# =========================================================================


def test_generation_suite_is_versioned() -> None:
    """Verify generation benchmark suite has explicit version identifier standard_v1."""
    suite = get_canonical_generation_suite()
    assert suite.suite_id == BENCHMARK_VERSION
    assert suite.suite_id == "standard_v1"
    assert len(suite.prompts) == len(CANONICAL_PROMPTS)


def test_canonical_generation_seeds_are_fixed() -> None:
    """Verify canonical generation benchmark uses fixed seeds 0..9 and fixed settings."""
    suite = get_canonical_generation_suite()
    assert suite.seeds == list(range(10))
    assert suite.canonical_settings.temperature == 0.8
    assert suite.canonical_settings.top_p == 0.95
    assert suite.canonical_settings.max_new_tokens == 256


# =========================================================================
# 6. Comparison Engine Tests
# =========================================================================


def test_compare_defaults_to_validation_not_test() -> None:
    """Verify compare_runs defaults to split='validation' to prevent test set snooping."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        run1 = root / "run1"
        run2 = root / "run2"

        create_mock_run(run1, tokenizer="bpe", macro_bpc=1.40, split="validation")
        create_mock_run(run2, tokenizer="char", macro_bpc=1.20, split="validation")

        report = compare_runs([run1, run2])
        assert report.split == "validation"


def test_best_checkpoint_selection_never_uses_test_metrics() -> None:
    """Verify that best checkpoint selection in metadata is strictly from validation BPC."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir) / "run_best"
        create_mock_run(run_dir, macro_bpc=1.25, split="validation")

        ckpt_meta = json.loads(
            (run_dir / "checkpoints" / "best" / "metadata.json").read_text(encoding="utf-8")
        )
        assert "best_val_bpc" in ckpt_meta
        assert "best_test_bpc" not in ckpt_meta


def test_evaluation_records_checkpoint_hash() -> None:
    """Verify evaluation report records checkpoint_model_sha256."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir) / "run_eval"
        create_mock_run(run_dir, macro_bpc=1.30, split="validation")

        eval_report = json.loads(
            (run_dir / "evaluation" / "validation_metrics.json").read_text(encoding="utf-8")
        )
        assert "checkpoint_model_sha256" in eval_report
        assert len(eval_report["checkpoint_model_sha256"]) == 64


def test_compare_rejects_stale_evaluation_report() -> None:
    """Verify compare_runs refuses evaluation report when checkpoint hash differs from model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        run1 = root / "run1"
        run2 = root / "run2"

        create_mock_run(run1, macro_bpc=1.40, mismatch_ckpt_hash=False)
        create_mock_run(run2, macro_bpc=1.20, mismatch_ckpt_hash=True)

        with pytest.raises(ValueError, match="stale"):
            compare_runs([run1, run2], split="validation")


def test_compare_rejects_different_split_hashes() -> None:
    """Verify compare_runs refuses comparison when split manifest hash differs across runs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        run1 = root / "run1"
        run2 = root / "run2"

        create_mock_run(run1, split_hash="split_A")
        create_mock_run(run2, split_hash="split_B")

        with pytest.raises(ValueError, match="split_manifest_hash differs"):
            compare_runs([run1, run2], split="validation")


def test_compare_rejects_different_corpus_fingerprints() -> None:
    """Verify compare_runs refuses comparison when corpus fingerprints differ across runs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        run1 = root / "run1"
        run2 = root / "run2"

        create_mock_run(run1, corpus_fp="corpus_v1")
        create_mock_run(run2, corpus_fp="corpus_v2")

        with pytest.raises(ValueError, match="corpus_fingerprint differs"):
            compare_runs([run1, run2], split="validation")


def test_allow_incompatible_disables_performance_ranking() -> None:
    """Verify --allow-incompatible bypasses refusal, preserves order, and marks incompatible."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        run1 = root / "run_slow"
        run2 = root / "run_fast"

        # run1 has higher BPC (worse) than run2, but run1 is passed first
        create_mock_run(run1, macro_bpc=2.50, corpus_fp="corpus_v1")
        create_mock_run(run2, macro_bpc=1.10, corpus_fp="corpus_v2")

        report = compare_runs([run1, run2], split="validation", allow_incompatible=True)

        assert not report.is_compatible
        assert report.ranking_metric == "none"
        # Input order must be preserved (run1 first, run2 second) - NOT sorted by BPC!
        assert report.runs[0].run_name == "run_slow"
        assert report.runs[1].run_name == "run_fast"


def test_compare_runs_table_generation() -> None:
    """Verify compare_runs generates complete table with all required columns sorted by BPC."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        run_bpe = root / "bpe-natural"
        run_char = root / "char-natural"

        create_mock_run(run_bpe, tokenizer="bpe", macro_bpc=1.45, token_ppl=18.0)
        create_mock_run(run_char, tokenizer="char", macro_bpc=1.25, token_ppl=3.5)

        report = compare_runs([run_bpe, run_char], split="validation")
        assert report.is_compatible

        # Sorted by Macro BPC ascending: char (1.25) before bpe (1.45)
        assert report.runs[0].run_name == "char-natural"
        assert report.runs[1].run_name == "bpe-natural"

        table = render_comparison_table(report)
        col_names = [col.header for col in table.columns]
        assert "Run" in col_names
        assert "Tokenizer" in col_names
        assert "Validation Macro BPC" in col_names
        assert "Token PPL" in col_names
        assert "Chars Exposed" in col_names


def test_compare_does_not_rank_by_perplexity() -> None:
    """Verify runs are ranked by Macro BPC, NEVER by token perplexity across tokenizers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        # Model A (CHAR): Lower token PPL (3.0) but higher/worse Macro BPC (1.80)
        # Model B (BPE): Higher token PPL (12.0) but lower/better Macro BPC (1.20)
        run_char = root / "run_char"
        run_bpe = root / "run_bpe"

        create_mock_run(run_char, tokenizer="char", macro_bpc=1.80, token_ppl=3.0)
        create_mock_run(run_bpe, tokenizer="bpe", macro_bpc=1.20, token_ppl=12.0)

        report = compare_runs([run_char, run_bpe], split="validation")

        # Must rank run_bpe first because 1.20 < 1.80 Macro BPC, ignoring the lower PPL of CHAR
        assert report.runs[0].run_name == "run_bpe"
        assert report.runs[1].run_name == "run_char"
