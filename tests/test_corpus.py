"""Comprehensive test suite for Scripture-LM corpus preparation subsystem."""

import json
from pathlib import Path

import pytest

from scripture_lm.corpus.audit import audit_corpus
from scripture_lm.corpus.manifest import (
    CorpusManifest,
    DocumentEntry,
    EditorialCleanupConfig,
    RegexReplacement,
    compute_file_sha256,
    validate_cleanup_regexes,
    validate_manifest,
)
from scripture_lm.corpus.normalize import (
    DocumentProvenance,
    compute_corpus_fingerprint,
    compute_normalization_fingerprint,
    conservative_normalize,
    normalize_corpus,
)
from scripture_lm.corpus.split import (
    generate_splits,
    load_or_generate_splits,
)
from scripture_lm.corpus.statistics import compute_corpus_stats


@pytest.fixture
def sample_corpus(tmp_path: Path) -> tuple[Path, CorpusManifest]:
    """Create a temporary multi-family corpus directory with valid text files and manifest."""
    corpus_root = tmp_path / "corpus"
    raw_dir = corpus_root / "raw"
    (raw_dir / "hebrew_bible").mkdir(parents=True)
    (raw_dir / "new_testament").mkdir(parents=True)
    (raw_dir / "quran").mkdir(parents=True)

    docs: list[DocumentEntry] = []

    # Hebrew Bible books (sizes varying)
    hb_data = {
        "genesis": (
            "In the beginning God created the heaven and the earth.\n\n"
            "And the earth was without form.\n"
        ),
        "exodus": (
            "Now these are the names of the children of Israel, which came into Egypt;\n\n"
            "every man and his household came with Jacob.\n"
        ),
        "leviticus": (
            "And the LORD called unto Moses, and spake unto him out of the "
            "tabernacle of the congregation.\n"
        ),
        "psalms": (
            "Blessed is the man that walketh not in the counsel of the ungodly.\n\n"
            "Nor standeth in the way of sinners.\n"
        )
        * 10,
    }
    for doc_id, text in hb_data.items():
        file_rel = f"raw/hebrew_bible/{doc_id}.txt"
        file_path = corpus_root / file_rel
        file_path.write_text(text, encoding="utf-8")
        sha = compute_file_sha256(file_path)
        docs.append(
            DocumentEntry(
                id=doc_id,
                family="hebrew_bible",
                path=file_rel,
                license="public-domain",
                sha256=sha,
            )
        )

    # New Testament books
    nt_data = {
        "matthew": "The book of the generation of Jesus Christ, the son of David.\n",
        "mark": "The beginning of the gospel of Jesus Christ, the Son of God;\n",
        "luke": (
            "Forasmuch as many have taken in hand to set forth in order a "
            "declaration of those things.\n"
        ),
        "john": "In the beginning was the Word, and the Word was with God, and the Word was God.\n",
    }
    for doc_id, text in nt_data.items():
        file_rel = f"raw/new_testament/{doc_id}.txt"
        file_path = corpus_root / file_rel
        file_path.write_text(text, encoding="utf-8")
        sha = compute_file_sha256(file_path)
        docs.append(
            DocumentEntry(
                id=doc_id,
                family="new_testament",
                path=file_rel,
                license="public-domain",
                sha256=sha,
            )
        )

    # Quran surahs
    q_data = {
        "001": (
            "In the name of Allah, the Beneficent, the Merciful.\n"
            "Praise be to Allah, Lord of the Worlds.\n"
        ),
        "002": (
            "Alif. Lam. Mim.\n"
            "This is the Scripture whereof there is no doubt, "
            "a guidance unto those who ward off evil;\n"
        )
        * 5,
        "112": "Say: He is Allah, the One!\nAllah, the eternally Besought of all!\n",
    }
    for doc_id, text in q_data.items():
        file_rel = f"raw/quran/{doc_id}.txt"
        file_path = corpus_root / file_rel
        file_path.write_text(text, encoding="utf-8")
        sha = compute_file_sha256(file_path)
        docs.append(
            DocumentEntry(
                id=f"quran_{doc_id}",
                family="quran",
                path=file_rel,
                license="public-domain",
                sha256=sha,
            )
        )

    manifest = CorpusManifest(
        version="1.0",
        name="test-corpus",
        license_policy="public-domain",
        documents=docs,
    )
    return corpus_root, manifest


def test_normalized_round_trip() -> None:
    """Normalizing normalized text is idempotent: norm(x) == norm(norm(x))."""
    raw_text = "In the   beginning  God created.\n\nAnd   the earth was   void.\n"
    norm1, _ = conservative_normalize(raw_text)
    norm2, _ = conservative_normalize(norm1)
    assert norm1 == norm2
    # Verify multiple spaces collapsed and Unix newlines
    assert "   " not in norm1
    assert "\r" not in norm1
    assert norm1.endswith("\n")


def test_conservative_normalization_invariants() -> None:
    """Conservative normalization preserves case, archaic language, accents, and punctuation."""
    text = (
        "And the LORD said unto Moses: 'Speak thou!' \r\n\r\n  Hallowed be Thy Name;  résumé.   \n"
    )
    norm, _ = conservative_normalize(text)
    # Preserves casing
    assert "LORD" in norm
    assert "Moses" in norm
    assert "Thy Name" in norm
    # Preserves accents
    assert "résumé" in norm
    # Preserves punctuation
    assert "'" in norm and "!" in norm and ";" in norm
    # Trims trailing spaces per line
    for line in norm.splitlines():
        assert line == line.rstrip()


def test_utf8_bom_removed() -> None:
    """Leading UTF-8 BOM must be removed."""
    bom_text = "\ufeffIn the beginning God created.\n"
    norm, _ = conservative_normalize(bom_text)
    assert not norm.startswith("\ufeff")
    assert norm.startswith("In the beginning")


def test_cleanup_runs_after_unicode_and_newline_normalization() -> None:
    """Regex cleanup operates on canonical NFC and Unix newlines."""
    # Text with CRLF and decomposed e + acute accent (e\u0301)
    decomposed = "CHAPTER 1\r\nRe\u0301sume\u0301 of holy text.\r\n"
    # Pattern matching "CHAPTER 1\n" with Unix newline
    cleanup = EditorialCleanupConfig(strip_header_patterns=[r"^CHAPTER 1\n"])
    norm, count = conservative_normalize(decomposed, cleanup)
    assert count == 1
    assert "CHAPTER 1" not in norm
    assert "Résumé" in norm or "R\u00e9sum\u00e9" in norm


def test_cleanup_reports_number_of_modifications() -> None:
    """Cleanup modifications count records all substitutions accurately."""
    raw = "EDITORIAL HEADER\n1:1 In the beginning.\n1:2 And darkness was there.\nEDITORIAL FOOTER"
    cleanup = EditorialCleanupConfig(
        strip_header_patterns=[r"EDITORIAL HEADER\n+"],
        strip_footer_patterns=[r"\n+EDITORIAL FOOTER"],
        line_prefix_patterns=[r"^\d+:\d+\s+"],
        replace_patterns=[RegexReplacement(pattern=r"darkness", replacement="obscurity")],
    )
    norm, mods = conservative_normalize(raw, cleanup)
    assert "EDITORIAL HEADER" not in norm
    assert "EDITORIAL FOOTER" not in norm
    assert "1:1" not in norm
    assert "1:2" not in norm
    assert "obscurity" in norm
    # 1 header + 1 footer + 2 line prefixes + 1 replacement = 5 mods
    assert mods == 5


def test_whitespace_only_document_rejected() -> None:
    """Documents containing only whitespace are rejected during normalization and audit."""
    with pytest.raises(ValueError, match="empty or contains only whitespace"):
        conservative_normalize("   \n\t  \r\n   \n")


def test_invalid_cleanup_regex_rejected_at_audit() -> None:
    """Malformed cleanup regex patterns are caught during manifest validation."""
    cleanup = EditorialCleanupConfig(line_prefix_patterns=["^[0-9+("])
    errors = validate_cleanup_regexes(cleanup, "genesis")
    assert len(errors) == 1
    assert "Invalid regex in document 'genesis'" in errors[0]


def test_path_traversal_rejection(sample_corpus: tuple[Path, CorpusManifest]) -> None:
    """Document paths escaping corpus/raw are rejected."""
    corpus_root, manifest = sample_corpus
    manifest.documents.append(
        DocumentEntry(
            id="escape_test",
            family="hebrew_bible",
            path="../../secret.txt",
            license="public-domain",
        )
    )
    errors = validate_manifest(manifest, corpus_root)
    assert any("Path traversal detected" in e for e in errors)


def test_empty_or_missing_files_rejected(sample_corpus: tuple[Path, CorpusManifest]) -> None:
    """Missing files and 0-byte files are rejected."""
    corpus_root, manifest = sample_corpus
    # Missing file
    manifest.documents.append(
        DocumentEntry(
            id="missing_doc",
            family="hebrew_bible",
            path="raw/hebrew_bible/non_existent.txt",
            license="public-domain",
        )
    )
    # Empty file
    empty_path = corpus_root / "raw/hebrew_bible/empty.txt"
    empty_path.write_text("", encoding="utf-8")
    manifest.documents.append(
        DocumentEntry(
            id="empty_doc",
            family="hebrew_bible",
            path="raw/hebrew_bible/empty.txt",
            license="public-domain",
        )
    )
    errors = validate_manifest(manifest, corpus_root)
    assert any("references non-existent file" in e for e in errors)
    assert any("empty or contains only whitespace" in e for e in errors)


def test_duplicate_ids_or_paths_rejected(sample_corpus: tuple[Path, CorpusManifest]) -> None:
    """Duplicate document IDs or duplicate paths are rejected."""
    corpus_root, manifest = sample_corpus
    manifest.documents.append(
        DocumentEntry(
            id="genesis",  # duplicate ID
            family="hebrew_bible",
            path="raw/hebrew_bible/exodus.txt",
            license="public-domain",
        )
    )
    manifest.documents.append(
        DocumentEntry(
            id="genesis_2",
            family="hebrew_bible",
            path="raw/hebrew_bible/genesis.txt",  # duplicate path
            license="public-domain",
        )
    )
    errors = validate_manifest(manifest, corpus_root)
    assert any("Duplicate document ID" in e for e in errors)
    assert any("Duplicate document path" in e for e in errors)


def test_manifest_sha256_mismatch_fails(sample_corpus: tuple[Path, CorpusManifest]) -> None:
    """Manifest specifying incorrect sha256 fails validation."""
    corpus_root, manifest = sample_corpus
    manifest.documents[0].sha256 = "0" * 64
    errors = validate_manifest(manifest, corpus_root)
    assert any("SHA-256 mismatch" in e for e in errors)


def test_corpus_purity(sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path) -> None:
    """Audit verifies clean corpus and passes when all files are tracked."""
    corpus_root, manifest = sample_corpus
    manifest_path = tmp_path / "corpus_manifest.toml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write('version = "1.0"\nname = "test"\nlicense_policy = "public-domain"\n')
        for doc in manifest.documents:
            f.write(
                f'[[documents]]\nid = "{doc.id}"\nfamily = "{doc.family}"\n'
                f'path = "{doc.path}"\nlicense = "{doc.license}"\nsha256 = "{doc.sha256}"\n'
            )

    result = audit_corpus(manifest_path=manifest_path, corpus_root=corpus_root)
    assert result.is_clean is True
    assert result.num_documents == len(manifest.documents)
    assert len(result.untracked_files) == 0


def test_unregistered_non_txt_file_is_flagged(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """Unregistered non-txt files in corpus/raw (e.g. .md, .text) trigger purity error."""
    corpus_root, manifest = sample_corpus
    rogue_file = corpus_root / "raw/quran/untracked.md"
    rogue_file.write_text("Untracked text", encoding="utf-8")

    manifest_path = tmp_path / "corpus_manifest.toml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write('version = "1.0"\nname = "test"\nlicense_policy = "public-domain"\n')
        for doc in manifest.documents:
            f.write(
                f'[[documents]]\nid = "{doc.id}"\nfamily = "{doc.family}"\n'
                f'path = "{doc.path}"\nlicense = "{doc.license}"\nsha256 = "{doc.sha256}"\n'
            )

    result = audit_corpus(manifest_path=manifest_path, corpus_root=corpus_root)
    assert result.is_clean is False
    assert any("untracked.md" in u for u in result.untracked_files)


def test_fingerprints_are_independent_of_absolute_project_path() -> None:
    """Fingerprint calculation uses relative paths/IDs and is deterministic across machines."""
    p1 = DocumentProvenance(
        document_id="genesis",
        family="hebrew_bible",
        source_path="raw/hebrew_bible/genesis.txt",
        source_sha256="abc123",
        normalized_sha256="def456",
        raw_bytes=100,
        raw_characters=100,
        normalized_bytes=95,
        normalized_characters=95,
        lines_before=10,
        lines_after=10,
        cleanup_modifications=0,
    )
    p2 = DocumentProvenance(
        document_id="matthew",
        family="new_testament",
        source_path="raw/new_testament/matthew.txt",
        source_sha256="111222",
        normalized_sha256="333444",
        raw_bytes=200,
        raw_characters=200,
        normalized_bytes=190,
        normalized_characters=190,
        lines_before=20,
        lines_after=20,
        cleanup_modifications=0,
    )

    fp_corpus1 = compute_corpus_fingerprint([p1, p2])
    fp_corpus2 = compute_corpus_fingerprint([p2, p1])  # Order-independent input
    assert fp_corpus1 == fp_corpus2

    fp_norm1 = compute_normalization_fingerprint([p1, p2])
    fp_norm2 = compute_normalization_fingerprint([p2, p1])
    assert fp_norm1 == fp_norm2


def test_normalization_fingerprint_changes_when_cleanup_config_changes() -> None:
    """Changing cleanup configuration changes the normalization fingerprint."""
    p1 = DocumentProvenance(
        document_id="genesis",
        family="hebrew_bible",
        source_path="raw/hebrew_bible/genesis.txt",
        source_sha256="abc123",
        normalized_sha256="def456",
        raw_bytes=100,
        raw_characters=100,
        normalized_bytes=95,
        normalized_characters=95,
        lines_before=10,
        lines_after=10,
        cleanup_modifications=0,
        cleanup_config=None,
    )
    p2 = p1.model_copy(update={"cleanup_config": {"strip_header_patterns": ["^HEADER"]}})
    fp1 = compute_normalization_fingerprint([p1])
    fp2 = compute_normalization_fingerprint([p2])
    assert fp1 != fp2


def test_deterministic_splitting(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """Splitting with identical seed produces identical document assignments every time."""
    corpus_root, manifest = sample_corpus
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=tmp_path / "lock.json"
    )

    split1 = generate_splits(manifest, lock, seed=1337)
    split2 = generate_splits(manifest, lock, seed=1337)

    assert split1.train == split2.train
    assert split1.validation == split2.validation
    assert split1.test == split2.test
    assert split1.actual_proportions == split2.actual_proportions


def test_no_document_in_multiple_splits(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """Splits are strictly disjoint: train, val, and test share no documents."""
    corpus_root, manifest = sample_corpus
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=tmp_path / "lock.json"
    )
    split = generate_splits(manifest, lock, seed=1337)

    s_train = set(split.train)
    s_val = set(split.validation)
    s_test = set(split.test)

    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)


def test_every_manifest_document_represented(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """All manifest documents are allocated to exactly one split."""
    corpus_root, manifest = sample_corpus
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=tmp_path / "lock.json"
    )
    split = generate_splits(manifest, lock, seed=1337)

    all_split_docs = set(split.train) | set(split.validation) | set(split.test)
    all_manifest_docs = {d.id for d in manifest.documents}
    assert all_split_docs == all_manifest_docs


def test_split_ratio_is_character_based(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """Split proportions reflect character counts rather than document counts."""
    corpus_root, manifest = sample_corpus
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=tmp_path / "lock.json"
    )
    split = generate_splits(manifest, lock, seed=1337)

    # Train should have the vast majority of characters (~85-95%)
    assert 0.70 <= split.actual_proportions["train"] <= 0.98
    assert split.actual_characters["train"] > split.actual_characters["validation"]
    assert split.actual_characters["train"] > split.actual_characters["test"]


def test_small_family_split_behavior(tmp_path: Path) -> None:
    """Families with fewer than 3 documents do not crash and report warnings."""
    corpus_root = tmp_path / "small_corpus"
    raw_dir = corpus_root / "raw/small_fam"
    raw_dir.mkdir(parents=True)

    d1 = raw_dir / "doc1.txt"
    d1.write_text("First document.\n", encoding="utf-8")
    d2 = raw_dir / "doc2.txt"
    d2.write_text("Second document.\n", encoding="utf-8")

    manifest = CorpusManifest(
        version="1.0",
        name="small",
        license_policy="public-domain",
        documents=[
            DocumentEntry(
                id="doc1",
                family="small_fam",
                path="raw/small_fam/doc1.txt",
                license="public-domain",
            ),
            DocumentEntry(
                id="doc2",
                family="small_fam",
                path="raw/small_fam/doc2.txt",
                license="public-domain",
            ),
        ],
    )
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=tmp_path / "lock.json"
    )
    split = generate_splits(manifest, lock, seed=1337)

    assert len(split.warnings) > 0
    assert any("contains only 2 documents" in w for w in split.warnings)
    # Ensure neither document is lost
    assert set(split.train) | set(split.validation) | set(split.test) == {"doc1", "doc2"}


def test_changed_corpus_invalidates_persisted_split(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """Changing corpus content modifies corpus_fingerprint, refusing split reuse without force."""
    corpus_root, manifest = sample_corpus
    split_file = tmp_path / "split_manifest.json"
    lock_file = tmp_path / "lock.json"

    # Initial generation
    lock1 = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=lock_file
    )
    split1, reused1 = load_or_generate_splits(manifest, lock1, split_path=split_file, seed=1337)
    assert reused1 is False

    # Modify a file
    (corpus_root / "raw/hebrew_bible/genesis.txt").write_text(
        "MODIFIED CONTENT.\n", encoding="utf-8"
    )
    lock2 = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=lock_file
    )
    assert lock1.corpus_fingerprint != lock2.corpus_fingerprint

    # Without force_split, reuse must raise ValueError
    with pytest.raises(ValueError, match="Corpus fingerprint mismatch"):
        load_or_generate_splits(
            manifest, lock2, split_path=split_file, seed=1337, force_split=False
        )

    # With force_split, succeeds and regenerates
    split2, reused2 = load_or_generate_splits(
        manifest, lock2, split_path=split_file, seed=1337, force_split=True
    )
    assert reused2 is False
    assert split2.corpus_fingerprint == lock2.corpus_fingerprint


def test_changed_seed_invalidates_persisted_split(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """Requesting a different seed refuses reuse of existing split without --force-split."""
    corpus_root, manifest = sample_corpus
    split_file = tmp_path / "split_manifest.json"
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=tmp_path / "lock.json"
    )

    load_or_generate_splits(manifest, lock, split_path=split_file, seed=1337)

    with pytest.raises(ValueError, match="Seed mismatch"):
        load_or_generate_splits(manifest, lock, split_path=split_file, seed=42, force_split=False)


def test_changed_split_algorithm_invalidates_persisted_split(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """Altered algorithm in existing split file refuses reuse without --force-split."""
    corpus_root, manifest = sample_corpus
    split_file = tmp_path / "split_manifest.json"
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=tmp_path / "lock.json"
    )

    split, _ = load_or_generate_splits(manifest, lock, split_path=split_file, seed=1337)

    # Mutate algorithm field in saved file
    data = json.loads(split_file.read_text(encoding="utf-8"))
    data["algorithm"] = "different_algorithm_v2"
    split_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Algorithm mismatch"):
        load_or_generate_splits(manifest, lock, split_path=split_file, seed=1337, force_split=False)


def test_changed_split_targets_invalidate_persisted_split(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """Requesting different target ratios refuses reuse without --force-split."""
    corpus_root, manifest = sample_corpus
    split_file = tmp_path / "split_manifest.json"
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=tmp_path / "norm", lock_path=tmp_path / "lock.json"
    )

    load_or_generate_splits(
        manifest,
        lock,
        split_path=split_file,
        targets={"train": 0.90, "validation": 0.05, "test": 0.05},
    )

    with pytest.raises(ValueError, match="Target ratios mismatch"):
        load_or_generate_splits(
            manifest,
            lock,
            split_path=split_file,
            targets={"train": 0.80, "validation": 0.10, "test": 0.10},
            force_split=False,
        )


def test_corpus_stats_computation(
    sample_corpus: tuple[Path, CorpusManifest], tmp_path: Path
) -> None:
    """compute_corpus_stats aggregates document, character, byte, and word counts correctly."""
    corpus_root, manifest = sample_corpus
    norm_dir = tmp_path / "norm"
    lock = normalize_corpus(
        manifest, corpus_root, output_dir=norm_dir, lock_path=tmp_path / "lock.json"
    )
    split = generate_splits(manifest, lock, seed=1337)

    stats = compute_corpus_stats(split, normalized_dir=norm_dir)
    assert stats["total"]["documents"] == len(manifest.documents)
    assert stats["total"]["characters"] > 0
    assert stats["total"]["words"] > 0
    assert "train" in stats["splits"]
    assert "hebrew_bible" in stats["families"]
