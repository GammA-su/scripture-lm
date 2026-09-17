"""Comprehensive unit test suite for Scripture-LM tokenizers and provenance verification."""

import json
import unicodedata
from pathlib import Path

from typer.testing import CliRunner

from scripture_lm.cli import app
from scripture_lm.corpus.manifest import (
    CorpusManifest,
    DocumentEntry,
    compute_file_sha256,
)
from scripture_lm.corpus.normalize import normalize_corpus
from scripture_lm.corpus.split import generate_splits
from scripture_lm.tokenization.base import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
)
from scripture_lm.tokenization.bpe import train_bpe_tokenizer
from scripture_lm.tokenization.character import build_character_tokenizer
from scripture_lm.tokenization.encode import (
    compute_and_update_tokenizer_stats,
)

runner = CliRunner()


def create_mini_split_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Helper to create a normalized scripture corpus with a valid split manifest and lock."""
    corpus_root = tmp_path / "corpus"
    raw_dir = corpus_root / "raw/hebrew_bible"
    raw_dir.mkdir(parents=True)

    # Train docs contain standard Latin text
    d1 = raw_dir / "doc1.txt"
    d1.write_text("In the beginning God created the heaven and the earth.\n", encoding="utf-8")
    d2 = raw_dir / "doc2.txt"
    d2.write_text(
        "And the earth was without form, and void; and darkness was upon the deep.\n",
        encoding="utf-8",
    )
    d3 = raw_dir / "doc3.txt"
    d3.write_text("And the Spirit of God moved upon the face of the waters.\n", encoding="utf-8")
    # Validation doc contains a unique Greek letter Omega (Ω) not in train
    d4 = raw_dir / "val_doc.txt"
    d4.write_text("And God said, Let there be light: Ω and there was light.\n", encoding="utf-8")
    # Test doc contains a unique symbol dagger (†) not in train
    d5 = raw_dir / "test_doc.txt"
    d5.write_text(
        "And God saw the light, that it was good: † and God divided the light.\n",
        encoding="utf-8",
    )

    manifest = CorpusManifest(
        version="1.0",
        name="mini",
        license_policy="public-domain",
        documents=[
            DocumentEntry(
                id="doc1",
                family="hebrew_bible",
                path="raw/hebrew_bible/doc1.txt",
                license="public-domain",
                sha256=compute_file_sha256(d1),
            ),
            DocumentEntry(
                id="doc2",
                family="hebrew_bible",
                path="raw/hebrew_bible/doc2.txt",
                license="public-domain",
                sha256=compute_file_sha256(d2),
            ),
            DocumentEntry(
                id="doc3",
                family="hebrew_bible",
                path="raw/hebrew_bible/doc3.txt",
                license="public-domain",
                sha256=compute_file_sha256(d3),
            ),
            DocumentEntry(
                id="val_doc",
                family="hebrew_bible",
                path="raw/hebrew_bible/val_doc.txt",
                license="public-domain",
                sha256=compute_file_sha256(d4),
            ),
            DocumentEntry(
                id="test_doc",
                family="hebrew_bible",
                path="raw/hebrew_bible/test_doc.txt",
                license="public-domain",
                sha256=compute_file_sha256(d5),
            ),
        ],
    )

    norm_dir = tmp_path / "normalized"
    lock_file = tmp_path / "corpus_lock.json"
    lock = normalize_corpus(manifest, corpus_root, output_dir=norm_dir, lock_path=lock_file)

    split_file = tmp_path / "split_manifest.json"
    split = generate_splits(manifest, lock, seed=1337)
    # Explicitly enforce our train vs val vs test allocation for the test
    split.train = ["doc1", "doc2", "doc3"]
    split.validation = ["val_doc"]
    split.test = ["test_doc"]
    split_file.write_text(json.dumps(split.model_dump(), indent=2), encoding="utf-8")

    return split_file, lock_file, norm_dir


def test_bpe_special_token_ids_exact(tmp_path: Path) -> None:
    """Byte-level BPE must assign <pad>=0, <bos>=1, <eos>=2, <unk>=3."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok, _ = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts",
        vocab_size=300,
    )
    assert tok.pad_id == PAD_ID == 0
    assert tok.bos_id == BOS_ID == 1
    assert tok.eos_id == EOS_ID == 2
    assert tok.unk_id == UNK_ID == 3
    assert tok.token_to_id(SPECIAL_TOKENS[0]) == 0
    assert tok.token_to_id(SPECIAL_TOKENS[1]) == 1
    assert tok.token_to_id(SPECIAL_TOKENS[2]) == 2
    assert tok.token_to_id(SPECIAL_TOKENS[3]) == 3


def test_char_special_token_ids_exact(tmp_path: Path) -> None:
    """Character tokenizer must assign <pad>=0, <bos>=1, <eos>=2, <unk>=3."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok, _ = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts",
    )
    assert tok.pad_id == PAD_ID == 0
    assert tok.bos_id == BOS_ID == 1
    assert tok.eos_id == EOS_ID == 2
    assert tok.unk_id == UNK_ID == 3
    assert tok.token_to_id(SPECIAL_TOKENS[0]) == 0
    assert tok.token_to_id(SPECIAL_TOKENS[1]) == 1
    assert tok.token_to_id(SPECIAL_TOKENS[2]) == 2
    assert tok.token_to_id(SPECIAL_TOKENS[3]) == 3


def test_char_vocabulary_only_sees_training_text(tmp_path: Path) -> None:
    """Character vocabulary must never contain characters appearing only in val or test."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok, _ = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts",
    )
    # Omega ('Ω') is in val_doc only (not in train vocabulary)
    assert tok.token_to_id("Ω") is None
    assert tok.encode("Ω") == [UNK_ID]
    # Dagger ('†') is in test_doc only (not in train vocabulary)
    assert tok.token_to_id("†") is None
    assert tok.encode("†") == [UNK_ID]
    # Standard character in train is present
    g_id = tok.token_to_id("G")
    assert g_id is not None
    assert g_id > 3


def test_bpe_round_trip_normalized_text(tmp_path: Path) -> None:
    """BPE round trip on normalized scripture text is lossless: decode(encode(text)) == text."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok, _ = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts",
        vocab_size=300,
    )
    sample = "In the beginning God created the heaven and the earth.\n"
    enc = tok.encode(sample)
    dec = tok.decode(enc)
    assert dec == sample

    # With BOS/EOS
    enc_bounded = tok.encode(sample, add_bos=True, add_eos=True)
    assert enc_bounded[0] == BOS_ID
    assert enc_bounded[-1] == EOS_ID
    assert tok.decode(enc_bounded, skip_special_tokens=True) == sample


def test_char_round_trip_normalized_text(tmp_path: Path) -> None:
    """Character round trip on known training text is lossless."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok, _ = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts",
    )
    sample = "In the beginning God created.\n"
    enc = tok.encode(sample)
    dec = tok.decode(enc)
    assert dec == sample

    # With BOS/EOS
    enc_bounded = tok.encode(sample, add_bos=True, add_eos=True)
    assert enc_bounded[0] == BOS_ID
    assert enc_bounded[-1] == EOS_ID
    assert tok.decode(enc_bounded, skip_special_tokens=True) == sample


def test_both_tokenizers_canonicalize_nfc(tmp_path: Path) -> None:
    """Both tokenizers canonicalize input to NFC: decode(encode(x)) == NFC(x)."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    bpe_tok, _ = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts_bpe",
        vocab_size=300,
    )
    char_tok, _ = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts_char",
    )

    # Decomposed: 'e' followed by combining acute accent (\u0301)
    decomposed = "e\u0301"
    expected_nfc = unicodedata.normalize("NFC", decomposed)  # "é"

    # BPE decode(encode(x)) == NFC(x)
    assert bpe_tok.decode(bpe_tok.encode(decomposed)) == expected_nfc

    # Char tokenizer maps to NFC before checking vocab; if é was in train it maps directly
    enc_char = char_tok.encode(decomposed)
    # Character tokenizer applies NFC internally
    assert len(enc_char) == 1


def test_unknown_character_handling_char_tokenizer(tmp_path: Path) -> None:
    """Unseen character encodes to UNK_ID (3) and decodes to <unk>."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok, _ = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts",
    )
    # 'Ω' is not in training vocabulary
    ids = tok.encode("Alpha Ω Omega")
    omega_idx = "Alpha Ω Omega".index("Ω")
    assert ids[omega_idx] == UNK_ID
    decoded = tok.decode(ids)
    assert "<unk>" in decoded


def test_bpe_has_zero_unknown_tokens(tmp_path: Path) -> None:
    """Byte-level BPE represents any valid UTF-8 string without <unk> tokens."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok, _ = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "artifacts",
        vocab_size=300,
    )
    # Arbitrary Unicode symbols not in train
    arbitrary = "Ω † § 宗教 🌟"
    ids = tok.encode(arbitrary)
    assert UNK_ID not in ids
    assert tok.decode(ids) == arbitrary


def test_tokenizer_refuses_modified_normalized_file(tmp_path: Path) -> None:
    """Tampering with normalized file triggers SHA-256 mismatch refusal."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    # Tamper with doc1.txt
    tampered = norm_d / "hebrew_bible/doc1.txt"
    tampered.write_text("TAMPERED DATA.\n", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="does not match corpus_lock.json"):
        train_bpe_tokenizer(
            split_manifest_path=split_f,
            corpus_lock_path=lock_f,
            normalized_dir=norm_d,
            output_dir=tmp_path / "artifacts",
            vocab_size=300,
        )

    with pytest.raises(ValueError, match="does not match corpus_lock.json"):
        build_character_tokenizer(
            split_manifest_path=split_f,
            corpus_lock_path=lock_f,
            normalized_dir=norm_d,
            output_dir=tmp_path / "artifacts",
        )


def test_tokenizer_refuses_mismatched_corpus_lock(tmp_path: Path) -> None:
    """Mismatched corpus fingerprint raises refusal error."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    # Mutate lock corpus fingerprint
    lock_data = json.loads(lock_f.read_text(encoding="utf-8"))
    lock_data["corpus_fingerprint"] = "0" * 64
    lock_f.write_text(json.dumps(lock_data), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="corpus fingerprint mismatch"):
        train_bpe_tokenizer(
            split_manifest_path=split_f,
            corpus_lock_path=lock_f,
            normalized_dir=norm_d,
            output_dir=tmp_path / "artifacts",
            vocab_size=300,
        )


def test_bpe_training_is_deterministic(tmp_path: Path) -> None:
    """Training BPE twice produces identical vocabulary and merges."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok1, meta1 = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "art1",
        vocab_size=300,
    )
    tok2, meta2 = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "art2",
        vocab_size=300,
    )
    assert meta1.tokenizer_artifact_sha256 == meta2.tokenizer_artifact_sha256
    assert tok1.vocab_size == tok2.vocab_size


def test_character_training_is_deterministic(tmp_path: Path) -> None:
    """Building character vocabulary twice produces identical vocabulary."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    tok1, meta1 = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "art1",
    )
    tok2, meta2 = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=tmp_path / "art2",
    )
    assert meta1.tokenizer_artifact_sha256 == meta2.tokenizer_artifact_sha256
    assert tok1.vocab_size == tok2.vocab_size


def test_bpe_metadata_records_training_provenance(tmp_path: Path) -> None:
    """bpe_metadata.json records training provenance correctly."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    art_dir = tmp_path / "artifacts"
    _, meta = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=art_dir,
        vocab_size=300,
    )
    meta_path = art_dir / "bpe_metadata.json"
    assert meta_path.is_file()
    loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded_meta["tokenizer_type"] == "bpe"
    assert loaded_meta["training_document_ids"] == ["doc1", "doc2", "doc3"]
    assert loaded_meta["special_token_ids"]["<pad>"] == 0
    assert loaded_meta["special_token_ids"]["<bos>"] == 1


def test_char_metadata_records_training_provenance(tmp_path: Path) -> None:
    """char_metadata.json records training provenance correctly."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    art_dir = tmp_path / "artifacts"
    _, meta = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=art_dir,
    )
    meta_path = art_dir / "char_metadata.json"
    assert meta_path.is_file()
    loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded_meta["tokenizer_type"] == "character"
    assert loaded_meta["training_document_ids"] == ["doc1", "doc2", "doc3"]


def test_stats_partial_and_comparison(tmp_path: Path) -> None:
    """Stats file handles BPE-only, Char-only, and comparison states gracefully."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    art_dir = tmp_path / "artifacts"
    stats_file = art_dir / "tokenizer_stats.json"

    # 1. Train BPE only
    bpe_tok, bpe_meta = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=art_dir,
        vocab_size=300,
    )
    stats1 = compute_and_update_tokenizer_stats(
        tokenizer=bpe_tok,
        metadata=bpe_meta,
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        stats_path=stats_file,
    )
    assert stats1["bpe"] is not None
    assert stats1["character"] is None
    assert stats1["comparison"] is None

    # 2. Build Character tokenizer
    char_tok, char_meta = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=art_dir,
    )
    stats2 = compute_and_update_tokenizer_stats(
        tokenizer=char_tok,
        metadata=char_meta,
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        stats_path=stats_file,
    )
    assert stats2["bpe"] is not None
    assert stats2["character"] is not None
    # Provenance matches, so comparison is populated
    assert stats2["comparison"] is not None
    assert stats2["comparison"]["compression_ratio_char_over_bpe"] > 1.0


def test_stats_refuse_cross_corpus_comparison(tmp_path: Path) -> None:
    """Stats sets comparison to None if BPE and Character were built on different corpus hashes."""
    split_f, lock_f, norm_d = create_mini_split_corpus(tmp_path)
    art_dir = tmp_path / "artifacts"
    stats_file = art_dir / "tokenizer_stats.json"

    bpe_tok, bpe_meta = train_bpe_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=art_dir,
        vocab_size=300,
    )
    compute_and_update_tokenizer_stats(bpe_tok, bpe_meta, split_f, lock_f, norm_d, stats_file)

    char_tok, char_meta = build_character_tokenizer(
        split_manifest_path=split_f,
        corpus_lock_path=lock_f,
        normalized_dir=norm_d,
        output_dir=art_dir,
    )
    # Spoof different corpus fingerprint in char metadata
    char_meta.corpus_fingerprint = "different_corpus_hash"
    stats = compute_and_update_tokenizer_stats(
        char_tok, char_meta, split_f, lock_f, norm_d, stats_file
    )

    assert stats["comparison"] is None
    assert "Comparison unavailable" in stats["comparison_status"]


def test_cli_tokenizer_help_and_execution() -> None:
    """Verify tokenizer CLI subcommands provide clean help."""
    res_bpe = runner.invoke(app, ["tokenizer", "train-bpe", "--help"])
    assert res_bpe.exit_code == 0
    assert "--vocab-size" in res_bpe.stdout

    res_char = runner.invoke(app, ["tokenizer", "build-char", "--help"])
    assert res_char.exit_code == 0

    res_stats = runner.invoke(app, ["tokenizer", "stats", "--help"])
    assert res_stats.exit_code == 0
