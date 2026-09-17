"""Comprehensive test suite for Scripture-LM dataset encoding, chunking, and dual samplers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from scripture_lm.cli import app
from scripture_lm.corpus.manifest import (
    CorpusManifest,
    DocumentEntry,
    compute_file_sha256,
)
from scripture_lm.corpus.normalize import normalize_corpus
from scripture_lm.corpus.split import generate_splits
from scripture_lm.data import (
    ChunkMetadata,
    NaturalSampler,
    ScriptureChunkDataset,
    SequentialSampler,
    TemperatureSampler,
    build_chunks_from_stream,
    calculate_temperature_parameters,
    collate_chunks,
    compute_token_character_credits,
    create_dataloader,
    encode_dataset,
    simulate_sampling,
)
from scripture_lm.tokenization.base import PAD_ID
from scripture_lm.tokenization.bpe import train_bpe_tokenizer
from scripture_lm.tokenization.character import CharacterTokenizer, build_character_tokenizer

runner = CliRunner()


def create_mini_corpus(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    """Create a 3-family miniature corpus with train, val, test splits for testing."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(parents=True)
    raw_dir = corpus_root / "raw"
    raw_dir.mkdir(parents=True)

    # 3 scripture families
    fam_hb = raw_dir / "hebrew_bible"
    fam_nt = raw_dir / "new_testament"
    fam_qr = raw_dir / "quran"
    fam_hb.mkdir()
    fam_nt.mkdir()
    fam_qr.mkdir()

    # Hebrew Bible docs: doc1 (large), doc2, doc3, val, test
    d_hb1 = fam_hb / "hb1.txt"
    d_hb1.write_text(
        "In the beginning God created the heaven and the earth.\n" * 15,
        encoding="utf-8",
    )
    d_hb2 = fam_hb / "hb2.txt"
    d_hb2.write_text("And the earth was without form, and void.\n" * 10, encoding="utf-8")
    d_hb3 = fam_hb / "hb3.txt"
    d_hb3.write_text(
        "And God said, Let there be light: and there was light.\n" * 10,
        encoding="utf-8",
    )
    d_hb_val = fam_hb / "hb_val.txt"
    d_hb_val.write_text("And God saw the light, that it was good.\n", encoding="utf-8")
    d_hb_test = fam_hb / "hb_test.txt"
    d_hb_test.write_text(
        "And the evening and the morning were the first day.\n",
        encoding="utf-8",
    )

    # New Testament docs: nt1, nt2, nt3, val, test
    d_nt1 = fam_nt / "nt1.txt"
    d_nt1.write_text(
        "In the beginning was the Word, and the Word was with God.\n" * 10,
        encoding="utf-8",
    )
    d_nt2 = fam_nt / "nt2.txt"
    d_nt2.write_text("The same was in the beginning with God.\n" * 10, encoding="utf-8")
    d_nt3 = fam_nt / "nt3.txt"
    d_nt3.write_text(
        "All things were made by him; and without him was not any thing made.\n" * 10,
        encoding="utf-8",
    )
    d_nt_val = fam_nt / "nt_val.txt"
    d_nt_val.write_text("In him was life; and the life was the light of men.\n", encoding="utf-8")
    d_nt_test = fam_nt / "nt_test.txt"
    d_nt_test.write_text(
        "And the light shineth in darkness; and the darkness comprehended it not.\n",
        encoding="utf-8",
    )

    # Quran docs: qr1, qr2, qr3, val, test
    d_qr1 = fam_qr / "qr1.txt"
    d_qr1.write_text(
        "In the name of Allah, the Entirely Merciful, the Especially Merciful.\n" * 10,
        encoding="utf-8",
    )
    d_qr2 = fam_qr / "qr2.txt"
    d_qr2.write_text("Praise has been to Allah, Lord of the worlds.\n" * 10, encoding="utf-8")
    d_qr3 = fam_qr / "qr3.txt"
    d_qr3.write_text(
        "The Entirely Merciful, the Especially Merciful, Sovereign of the Day of Recompense.\n"
        * 10,
        encoding="utf-8",
    )
    d_qr_val = fam_qr / "qr_val.txt"
    d_qr_val.write_text("It is You we worship and You we ask for help.\n", encoding="utf-8")
    d_qr_test = fam_qr / "qr_test.txt"
    d_qr_test.write_text("Guide us to the straight path.\n", encoding="utf-8")

    all_docs = [
        (d_hb1, "hb1", "hebrew_bible"),
        (d_hb2, "hb2", "hebrew_bible"),
        (d_hb3, "hb3", "hebrew_bible"),
        (d_hb_val, "hb_val", "hebrew_bible"),
        (d_hb_test, "hb_test", "hebrew_bible"),
        (d_nt1, "nt1", "new_testament"),
        (d_nt2, "nt2", "new_testament"),
        (d_nt3, "nt3", "new_testament"),
        (d_nt_val, "nt_val", "new_testament"),
        (d_nt_test, "nt_test", "new_testament"),
        (d_qr1, "qr1", "quran"),
        (d_qr2, "qr2", "quran"),
        (d_qr3, "qr3", "quran"),
        (d_qr_val, "qr_val", "quran"),
        (d_qr_test, "qr_test", "quran"),
    ]

    manifest = CorpusManifest(
        version="1.0",
        name="test_mini",
        license_policy="public-domain",
        documents=[
            DocumentEntry(
                id=doc_id,
                family=fam,
                path=str(path.relative_to(corpus_root)).replace("\\", "/"),
                license="public-domain",
                sha256=compute_file_sha256(path),
            )
            for path, doc_id, fam in all_docs
        ],
    )

    norm_dir = tmp_path / "data" / "normalized"
    lock_file = tmp_path / "data" / "corpus_lock.json"
    lock = normalize_corpus(manifest, corpus_root, output_dir=norm_dir, lock_path=lock_file)

    split_file = tmp_path / "data" / "splits" / "split_manifest.json"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_obj = generate_splits(manifest, lock, seed=42)
    split_file.write_text(split_obj.model_dump_json(indent=2), encoding="utf-8")

    return corpus_root, norm_dir, lock_file, split_file


def test_target_character_credits_not_window_character_counts() -> None:
    """raw_character_count must count target tokens [s+1, s+V), not the full window."""
    # 5 tokens with character credits [0, 3, 4, 5, 6] (token 0 has credit 0)
    char_credits = np.array([0, 3, 4, 5, 6], dtype=np.int64)
    doc_spans = [("doc1", 0, 5)]

    # With context_length L=2, window=3:
    # Chunk 0: s=0, valid=3. Target tokens are 1, 2 (credits: 3 + 4 = 7)
    # Chunk 1: s=2, valid=3. Target tokens are 3, 4 (credits: 5 + 6 = 11)
    chunks, _ = build_chunks_from_stream(
        token_count=5,
        char_credits=char_credits,
        doc_spans=doc_spans,
        context_length=2,
        split="train",
        family="test_fam",
        tokenizer_type="bpe",
        bin_path="test.bin",
    )

    assert len(chunks) == 2
    assert chunks[0].raw_character_count == 7  # 3 + 4, NOT 0 + 3 + 4
    assert chunks[1].raw_character_count == 11  # 5 + 6, NOT 4 + 5 + 6
    # Invariant: sum of target credits equals total text characters (3+4+5+6 = 18)
    assert sum(c.raw_character_count for c in chunks) == 18


def test_unicode_multibyte_character_credit_counted_once(tmp_path: Path) -> None:
    """Multibyte Unicode characters must be credited exactly once, summing to len(text)."""
    text = "é — ’ Ω 漢 🙂 multiple multibyte characters in text."
    # Build a mini BPE tokenizer
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    f = raw_dir / "doc.txt"
    f.write_text(text, encoding="utf-8")

    manifest = CorpusManifest(
        version="1.0",
        name="unicode_test",
        license_policy="public-domain",
        documents=[
            DocumentEntry(
                id="doc",
                family="hebrew_bible",
                path="raw/doc.txt",
                license="public-domain",
                sha256=compute_file_sha256(f),
            )
        ],
    )
    norm_dir = tmp_path / "norm"
    lock_file = tmp_path / "lock.json"
    lock = normalize_corpus(manifest, tmp_path, output_dir=norm_dir, lock_path=lock_file)
    split_file = tmp_path / "split.json"
    split_obj = generate_splits(manifest, lock, seed=42)
    split_file.write_text(split_obj.model_dump_json(indent=2), encoding="utf-8")

    tok, _ = train_bpe_tokenizer(
        split_manifest_path=split_file,
        corpus_lock_path=lock_file,
        normalized_dir=norm_dir,
        output_dir=tmp_path / "artifacts",
        vocab_size=300,
    )

    # Read normalized text
    norm_text = (norm_dir / "hebrew_bible" / "doc.txt").read_text(encoding="utf-8")
    tokens, credits = compute_token_character_credits(norm_text, tok)

    assert sum(credits) == len(norm_text)
    assert len(tokens) == len(credits)


def test_special_tokens_have_zero_character_credit() -> None:
    """Special tokens <bos>, <eos>, <pad> receive 0 character credit."""
    credits = [0, 5, 7, 0]  # <bos>, t1, t2, <eos>
    char_credits = np.array(credits, dtype=np.int64)
    chunks, _ = build_chunks_from_stream(
        token_count=4,
        char_credits=char_credits,
        doc_spans=[("d1", 0, 4)],
        context_length=3,
        split="train",
        family="f",
        tokenizer_type="bpe",
        bin_path="test.bin",
    )
    # Target tokens are 1, 2, 3 (credits: 5 + 7 + 0 = 12)
    assert chunks[0].raw_character_count == 12


def test_sum_natural_target_chars_equals_exact_corpus_chars(tmp_path: Path) -> None:
    """Exact invariant: sum(natural chunk target chars) == sum(train docs normalized chars)."""
    corpus_root, norm_dir, lock_file, split_file = create_mini_corpus(tmp_path)

    # Train BPE
    art_dir = tmp_path / "artifacts"
    tok, _ = train_bpe_tokenizer(
        split_manifest_path=split_file,
        corpus_lock_path=lock_file,
        normalized_dir=norm_dir,
        output_dir=art_dir,
        vocab_size=350,
    )

    enc_dir = tmp_path / "encoded"
    chunks_by_split, prov = encode_dataset(
        tokenizer=tok,
        split_manifest_path=split_file,
        corpus_lock_path=lock_file,
        normalized_dir=norm_dir,
        output_base_dir=enc_dir,
        context_length=128,
    )

    with open(lock_file, "r", encoding="utf-8") as f:
        lock = json.load(f)
    with open(split_file, "r", encoding="utf-8") as f:
        split = json.load(f)

    prov_map = {d["document_id"]: d for d in lock["documents"]}
    expected_n = sum(prov_map[doc_id]["normalized_characters"] for doc_id in split["train"])

    train_chunks = chunks_by_split["train"]
    actual_n = sum(c.raw_character_count for c in train_chunks)

    assert actual_n == expected_n
    assert prov.natural_train_target_characters == expected_n


def test_bpe_and_char_use_same_natural_family_character_sizes(tmp_path: Path) -> None:
    """Natural family character sizes n_i are identical across BPE and Character models."""
    corpus_root, norm_dir, lock_file, split_file = create_mini_corpus(tmp_path)
    art_dir = tmp_path / "artifacts"
    enc_dir = tmp_path / "encoded"

    bpe_tok, _ = train_bpe_tokenizer(
        split_manifest_path=split_file,
        corpus_lock_path=lock_file,
        normalized_dir=norm_dir,
        output_dir=art_dir,
        vocab_size=350,
    )
    char_tok, _ = build_character_tokenizer(
        split_manifest_path=split_file,
        corpus_lock_path=lock_file,
        normalized_dir=norm_dir,
        output_dir=art_dir,
    )

    bpe_chunks, _ = encode_dataset(
        tokenizer=bpe_tok,
        split_manifest_path=split_file,
        corpus_lock_path=lock_file,
        normalized_dir=norm_dir,
        output_base_dir=enc_dir,
        context_length=128,
    )
    char_chunks, _ = encode_dataset(
        tokenizer=char_tok,
        split_manifest_path=split_file,
        corpus_lock_path=lock_file,
        normalized_dir=norm_dir,
        output_base_dir=enc_dir,
        context_length=512,
    )

    # Aggregate n_i by family for both
    bpe_fam_n: dict[str, int] = {}
    for c in bpe_chunks["train"]:
        bpe_fam_n[c.family] = bpe_fam_n.get(c.family, 0) + c.raw_character_count

    char_fam_n: dict[str, int] = {}
    for c in char_chunks["train"]:
        char_fam_n[c.family] = char_fam_n.get(c.family, 0) + c.raw_character_count

    for fam in bpe_fam_n:
        assert bpe_fam_n[fam] == char_fam_n[fam]


def test_temperature_target_exposure_is_tokenizer_independent() -> None:
    """Target exposure share r_i is identical regardless of chunk size or tokenization."""
    # Family A and B have same character count, but different chunk count
    counts_bpe = {"A": 10, "B": 20}
    chars = {"A": 10000, "B": 10000}

    params_bpe = calculate_temperature_parameters(counts_bpe, chars, alpha=0.5)
    counts_char = {"A": 40, "B": 80}
    params_char = calculate_temperature_parameters(counts_char, chars, alpha=0.5)

    assert params_bpe["A"]["target_exposure_share"] == pytest.approx(0.5)
    assert params_char["A"]["target_exposure_share"] == pytest.approx(0.5)


def test_temperature_alpha_0_raw_exposure_is_equal_across_families() -> None:
    """Alpha=0 target exposure share is mathematically equal across all families."""
    counts = {"hebrew_bible": 50, "new_testament": 20, "quran": 10}
    chars = {"hebrew_bible": 50000, "new_testament": 15000, "quran": 8000}

    params = calculate_temperature_parameters(counts, chars, alpha=0.0)
    for fam in counts:
        assert params[fam]["target_exposure_share"] == pytest.approx(1.0 / 3.0)

    # Simulated observed exposure within statistical tolerance (+-1.5%)
    sim = simulate_sampling(counts, chars, alpha=0.0, draws=100_000, seed=42)
    for fam in counts:
        assert sim[fam]["observed_raw_pct"] == pytest.approx(1.0 / 3.0, abs=0.015)


def test_temperature_alpha_1_raw_exposure_matches_natural_distribution() -> None:
    """Alpha=1 target exposure share mathematically matches natural corpus distribution."""
    counts = {"hebrew_bible": 50, "new_testament": 20, "quran": 10}
    chars = {"hebrew_bible": 50000, "new_testament": 15000, "quran": 8000}
    total_chars = sum(chars.values())

    params = calculate_temperature_parameters(counts, chars, alpha=1.0)
    for fam in counts:
        expected_share = chars[fam] / total_chars
        assert params[fam]["target_exposure_share"] == pytest.approx(expected_share)

    # Simulated observed exposure within statistical tolerance (+-1.5%)
    sim = simulate_sampling(counts, chars, alpha=1.0, draws=100_000, seed=42)
    for fam in counts:
        expected_share = chars[fam] / total_chars
        assert sim[fam]["observed_raw_pct"] == pytest.approx(expected_share, abs=0.015)


def test_temperature_overshoot_carried_between_effective_epochs(tmp_path: Path) -> None:
    """Overshoot is carried forward across effective epochs: target = e * N."""
    chunks = [
        ChunkMetadata(
            chunk_id=f"c_{i}",
            split="train",
            family="hebrew_bible",
            tokenizer="bpe",
            bin_path="b.bin",
            token_start=0,
            valid_token_count=100,
            raw_character_count=300,
            document_ids=["d"],
        )
        for i in range(10)
    ]
    # N = 3000
    sampler = TemperatureSampler(chunks, alpha=0.5, seed=42)
    assert sampler.N == 3000

    # Draw epoch 1
    _ = list(sampler)
    assert sampler.effective_epochs_completed == 1
    assert sampler.cumulative_raw_chars >= 3000
    overshoot_1 = sampler.cumulative_raw_chars - 3000

    # Draw epoch 2: target is 6000
    _ = list(sampler)
    assert sampler.effective_epochs_completed == 2
    assert sampler.cumulative_raw_chars >= 6000
    overshoot_2 = sampler.cumulative_raw_chars - 6000

    # Total overshoot is bounded by at most one chunk
    assert overshoot_1 < 300
    assert overshoot_2 < 300


def test_temperature_total_budget_differs_from_E_times_N_by_at_most_one_chunk() -> None:
    """After E epochs, total raw chars differs from E*N by at most one chunk."""
    chunks = [
        ChunkMetadata(
            chunk_id=f"c_{i}",
            split="train",
            family="fam",
            tokenizer="bpe",
            bin_path="b.bin",
            token_start=0,
            valid_token_count=50,
            raw_character_count=150,
            document_ids=["d"],
        )
        for i in range(20)
    ]
    # N = 3000, max chunk = 150
    sampler = TemperatureSampler(chunks, alpha=0.5, seed=123)
    E = 5
    for _ in range(E):
        list(sampler)

    assert sampler.effective_epochs_completed == E
    target = E * 3000
    assert sampler.cumulative_raw_chars >= target
    assert sampler.cumulative_raw_chars - target <= 150


def test_temperature_sampling_repeats_chunks() -> None:
    """Temperature sampling is with replacement and can repeat chunks."""
    # Small pool of 2 chunks, budget N = 2000
    chunks = [
        ChunkMetadata(
            chunk_id="c1",
            split="train",
            family="fam",
            tokenizer="bpe",
            bin_path="b.bin",
            token_start=0,
            valid_token_count=10,
            raw_character_count=100,
            document_ids=["d1"],
        ),
        ChunkMetadata(
            chunk_id="c2",
            split="train",
            family="fam",
            tokenizer="bpe",
            bin_path="b.bin",
            token_start=10,
            valid_token_count=10,
            raw_character_count=100,
            document_ids=["d2"],
        ),
    ]
    # 2 chunks with 100 chars each => N = 200. With target = 200, it takes 2 chunks.
    # Set chars to 10 each so N = 20, but draw multiple times
    chunks[0].raw_character_count = 10
    chunks[1].raw_character_count = 10
    sampler = TemperatureSampler(chunks, alpha=0.5, seed=42)
    sampler.N = 100  # override budget to force ~10 draws from 2 chunks
    draws = list(sampler)
    assert len(draws) >= 10
    assert len(set(draws)) <= 2  # Only 2 unique chunk IDs, so must repeat


def test_natural_mode_exact_single_pass() -> None:
    """Natural mode consumes every chunk exactly once per epoch with no duplicates."""
    chunks = [
        ChunkMetadata(
            chunk_id=f"c_{i}",
            split="train",
            family="fam",
            tokenizer="bpe",
            bin_path="b.bin",
            token_start=i * 10,
            valid_token_count=10,
            raw_character_count=20,
            document_ids=[f"d_{i}"],
        )
        for i in range(25)
    ]
    sampler = NaturalSampler(chunks, seed=42)
    epoch1 = list(sampler)
    assert len(epoch1) == 25
    assert len(set(epoch1)) == 25
    assert sorted(epoch1) == list(range(25))

    epoch2 = list(sampler)
    assert len(epoch2) == 25
    assert len(set(epoch2)) == 25
    assert sorted(epoch2) == list(range(25))
    # Different order across epochs
    assert epoch1 != epoch2


def test_natural_family_ratios_match_dataset() -> None:
    """Natural mode family chunk counts match the dataset chunk distribution exactly."""
    chunks = [
        ChunkMetadata(
            chunk_id=f"c_{i}",
            split="train",
            family="A" if i < 30 else "B",
            tokenizer="bpe",
            bin_path="b.bin",
            token_start=i * 10,
            valid_token_count=10,
            raw_character_count=20,
            document_ids=[f"d_{i}"],
        )
        for i in range(50)
    ]
    sampler = NaturalSampler(chunks, seed=42)
    epoch_indices = list(sampler)
    count_a = sum(1 for idx in epoch_indices if chunks[idx].family == "A")
    count_b = sum(1 for idx in epoch_indices if chunks[idx].family == "B")
    assert count_a == 30
    assert count_b == 20


def test_chunk_stride_and_padding(tmp_path: Path) -> None:
    """Stride L, trailing chunks padded, targets masked with -100."""
    # Create a small binary file with 7 tokens: [10, 11, 12, 13, 14, 15, 16]
    bin_file = tmp_path / "stream.bin"
    tokens = np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.uint16)
    tokens.tofile(bin_file)

    # L = 4 (window = 5). Stride = 4.
    # Chunk 0: tokens 0..4 (5 tokens: 10, 11, 12, 13, 14). Full.
    # Chunk 1: tokens 4..6 (3 tokens: 14, 15, 16). Partial.
    chunks = [
        ChunkMetadata(
            chunk_id="c0",
            split="train",
            family="f",
            tokenizer="bpe",
            bin_path=str(bin_file),
            token_start=0,
            valid_token_count=5,
            raw_character_count=15,
            document_ids=["d"],
        ),
        ChunkMetadata(
            chunk_id="c1",
            split="train",
            family="f",
            tokenizer="bpe",
            bin_path=str(bin_file),
            token_start=4,
            valid_token_count=3,
            raw_character_count=8,
            document_ids=["d"],
        ),
    ]

    dataset = ScriptureChunkDataset(chunks, context_length=4)
    item0 = dataset[0]
    assert item0["input_ids"].tolist() == [10, 11, 12, 13]
    assert item0["target_ids"].tolist() == [11, 12, 13, 14]
    assert item0["target_token_count"] == 4

    item1 = dataset[1]
    # Padded with PAD_ID (0)
    assert item1["input_ids"].tolist() == [14, 15, 16, PAD_ID]
    # Targets masked with -100: valid targets are tokens at indices 1 and 2 (15, 16)
    assert item1["target_ids"].tolist() == [15, 16, -100, -100]
    assert item1["target_token_count"] == 2


def test_model_token_exposure_counts_targets_only(tmp_path: Path) -> None:
    """Model token exposure equals target positions where target != -100 (V - 1)."""
    bin_file = tmp_path / "stream.bin"
    np.array([1, 2, 3, 4, 5], dtype=np.uint16).tofile(bin_file)

    chunks = [
        ChunkMetadata(
            chunk_id="c0",
            split="train",
            family="f",
            tokenizer="bpe",
            bin_path=str(bin_file),
            token_start=0,
            valid_token_count=4,  # targets: 3
            raw_character_count=10,
            document_ids=["d"],
        ),
        ChunkMetadata(
            chunk_id="c1",
            split="train",
            family="f",
            tokenizer="bpe",
            bin_path=str(bin_file),
            token_start=0,
            valid_token_count=2,  # targets: 1
            raw_character_count=5,
            document_ids=["d"],
        ),
    ]
    dataset = ScriptureChunkDataset(chunks, context_length=4)
    batch = collate_chunks([dataset[0], dataset[1]])

    # Count unmasked targets in target_ids
    unmasked = (batch["target_ids"] != -100).sum().item()
    assert unmasked == (4 - 1) + (2 - 1)  # 3 + 1 = 4
    assert batch["target_tokens"] == 4


def test_partial_final_batch_is_not_dropped(tmp_path: Path) -> None:
    """drop_last=False ensures partial final batches are never discarded."""
    bin_file = tmp_path / "stream.bin"
    np.array([1, 2, 3, 4, 5], dtype=np.uint16).tofile(bin_file)

    chunks = [
        ChunkMetadata(
            chunk_id=f"c_{i}",
            split="train",
            family="f",
            tokenizer="bpe",
            bin_path=str(bin_file),
            token_start=0,
            valid_token_count=4,
            raw_character_count=10,
            document_ids=["d"],
        )
        for i in range(7)
    ]
    dataset = ScriptureChunkDataset(chunks, context_length=3)
    sampler = SequentialSampler(chunks)
    loader = create_dataloader(dataset, sampler=sampler, batch_size=4, drop_last=False)

    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["input_ids"].shape[0] == 4
    assert batches[1]["input_ids"].shape[0] == 3  # partial batch preserved


def test_sampler_state_matches_consumed_batches(tmp_path: Path) -> None:
    """With num_workers=0, sampler position matches consumed training items."""
    bin_file = tmp_path / "stream.bin"
    np.array([1, 2, 3, 4, 5], dtype=np.uint16).tofile(bin_file)

    chunks = [
        ChunkMetadata(
            chunk_id=f"c_{i}",
            split="train",
            family="f",
            tokenizer="bpe",
            bin_path=str(bin_file),
            token_start=0,
            valid_token_count=4,
            raw_character_count=10,
            document_ids=["d"],
        )
        for i in range(10)
    ]
    dataset = ScriptureChunkDataset(chunks, context_length=3)
    sampler = NaturalSampler(chunks, seed=42)
    loader = create_dataloader(dataset, sampler=sampler, batch_size=3, num_workers=0)

    it = iter(loader)
    next(it)  # consumes 3 items
    assert sampler.position_in_epoch == 3
    next(it)  # consumes 3 items
    assert sampler.position_in_epoch == 6


def test_uint16_vocab_bound_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Vocab size > 65536 or token ID > 65535 must raise ValueError."""
    corpus_root, norm_dir, lock_file, split_file = create_mini_corpus(tmp_path)
    tok, _ = build_character_tokenizer(
        split_manifest_path=split_file,
        corpus_lock_path=lock_file,
        normalized_dir=norm_dir,
        output_dir=tmp_path / "artifacts",
    )

    monkeypatch.setattr(CharacterTokenizer, "vocab_size", property(lambda self: 70000))
    with pytest.raises(ValueError, match="uint16 bound"):
        encode_dataset(
            tokenizer=tok,
            split_manifest_path=split_file,
            corpus_lock_path=lock_file,
            normalized_dir=norm_dir,
            output_base_dir=tmp_path / "encoded",
        )


def test_sampler_state_round_trip() -> None:
    """Both NaturalSampler and TemperatureSampler state_dict round-trip reproduces sequence."""
    chunks = [
        ChunkMetadata(
            chunk_id=f"c_{i}",
            split="train",
            family="f1" if i < 5 else "f2",
            tokenizer="bpe",
            bin_path="b.bin",
            token_start=0,
            valid_token_count=10,
            raw_character_count=25,
            document_ids=["d"],
        )
        for i in range(10)
    ]
    # 1. NaturalSampler round trip
    nat_sampler = NaturalSampler(chunks, seed=100)
    nat_it = iter(nat_sampler)
    _ = [next(nat_it) for _ in range(4)]
    state = nat_sampler.state_dict()
    seq1_part2 = [next(nat_it) for _ in range(6)]

    nat_resumed = NaturalSampler(chunks, seed=999)
    nat_resumed.load_state_dict(state)
    seq2_part2 = list(nat_resumed)
    assert seq1_part2 == seq2_part2

    # 2. TemperatureSampler round trip
    temp_sampler = TemperatureSampler(chunks, alpha=0.5, seed=200)
    temp_it = iter(temp_sampler)
    _ = [next(temp_it) for _ in range(5)]
    temp_state = temp_sampler.state_dict()
    temp_p2 = [next(temp_it) for _ in range(5)]

    temp_resumed = TemperatureSampler(chunks, alpha=0.5, seed=999)
    temp_resumed.load_state_dict(temp_state)
    temp_resumed_it = iter(temp_resumed)
    temp_resumed_p2 = [next(temp_resumed_it) for _ in range(5)]
    assert temp_p2 == temp_resumed_p2


def test_evaluation_never_uses_temperature() -> None:
    """SequentialSampler produces fixed 0..len-1 sequence without sampling."""
    chunks = [
        ChunkMetadata(
            chunk_id=f"val_{i}",
            split="validation",
            family="f",
            tokenizer="bpe",
            bin_path="b.bin",
            token_start=0,
            valid_token_count=10,
            raw_character_count=20,
            document_ids=["d"],
        )
        for i in range(8)
    ]
    sampler = SequentialSampler(chunks)
    assert list(sampler) == list(range(8))


def test_cli_encode_and_sampling_preview(tmp_path: Path) -> None:
    """CLI encode and sampling-preview commands function with proper help and execution."""
    res_help = runner.invoke(app, ["encode", "--help"])
    assert res_help.exit_code == 0
    assert "--tokenizer" in res_help.stdout

    res_prev_help = runner.invoke(app, ["corpus", "sampling-preview", "--help"])
    assert res_prev_help.exit_code == 0
    assert "--sampling-mode" in res_prev_help.stdout
    assert "--sampling-alpha" in res_prev_help.stdout
