"""Offline corpus construction uses temporary sources and never the real corpus."""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

import pytest

from scripture_lm.corpus.build_core_canon import (
    BUILDER_VERSION,
    HEBREW_BOOKS,
    NT_BOOKS,
    PICKTHALL_AYAH_COUNTS,
    Book,
    BuildError,
    BuildPlan,
    Document,
    build,
    clean_chapter,
    cleaning_report,
    inspect_sources,
    main,
    manifest_text,
    parse_pickthall,
    provenance_report,
    scan_ebible,
    validate_destinations,
)


def chapter_path(root: Path, source: str, book: Book, chapter: int) -> Path:
    prefix = "engjps" if source == "jps" else "eng-kjv2006"
    # Observed padding: PSA uses 001..150, while GEN uses 01..50; MAT starts at 070.
    number = f"{chapter:03d}" if book.code == "PSA" else f"{chapter:02d}"
    return (
        root
        / "sources"
        / "ebible"
        / source
        / (f"{prefix}_{book.index:03d}_{book.code}_{number}_read.txt")
    )


def quran_fixture() -> str:
    return (
        "\n".join(
            f"{surah}|{ayah}|Text {surah}-{ayah}; 'punctuation.'"
            for surah, count in enumerate(PICKTHALL_AYAH_COUNTS, 1)
            for ayah in range(1, count + 1)
        )
        + "\n\n# Source: fixture\n"
    )


@pytest.fixture(scope="module")
def source_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("builder_sources")
    for source, books in (("jps", HEBREW_BOOKS), ("kjv", (*NT_BOOKS, HEBREW_BOOKS[0]))):
        directory = root / "sources" / "ebible" / source
        directory.mkdir(parents=True)
        prefix = "engjps" if source == "jps" else "eng-kjv2006"
        (directory / f"{prefix}_000_000_000_read.txt").write_text(
            "This set of files contains a script\n"
            "for the purpose of reading to make an audio recording\n",
            encoding="utf-8",
        )
        (directory / "copr.htm").write_text("<html>metadata</html>", encoding="utf-8")
        (directory / "keys.asc").write_text("fixture signing key", encoding="utf-8")
        for book in books:
            for chapter in range(1, book.chapters + 1):
                text = (
                    f"Editorial title: {book.slug}.\nChapter {chapter}.\n"
                    f"{source} {book.code} {chapter}; 'Keep THIS!'\n¶ Next line.\n"
                )
                chapter_path(root, source, book, chapter).write_text(text, encoding="utf-8")
    for name in ("engjps_readaloud.zip", "eng-kjv2006_readaloud.zip"):
        (root / "sources" / "ebible" / name).write_bytes(b"opaque archive fixture")
    quran = root / "sources" / "quran"
    quran.mkdir()
    (quran / "pickthall.txt").write_text(quran_fixture(), encoding="utf-8")
    return root


@pytest.fixture
def writable_sources(source_fixture: Path, tmp_path: Path) -> Path:
    shutil.copytree(source_fixture / "sources", tmp_path / "sources")
    return tmp_path


def test_chapter_headings_pilcrows_and_punctuation() -> None:
    text = "Book title.\r\nChapter 1.\r\n  ¶ And He said: 'Keep THIS!';  \r\n\r\nNext.\r\n"
    prose, audit = clean_chapter(text, 1, "fixture")
    assert prose == "And He said: 'Keep THIS!'; Next."
    assert audit["lines_discarded_before_heading"] == 1
    assert audit["chapter_headings_removed"] == 1
    assert audit["pilcrow_markers_removed"] == 1
    assert audit["scripture_characters_retained"] == len(prose)


def test_later_chapter_without_title_and_equivalent_pilcrow_whitespace() -> None:
    prose, audit = clean_chapter("Chapter 10.\nOne. ¶\tTwo.\n", 10, "fixture")
    assert prose == "One. Two."
    assert audit["nonempty_header_lines_removed"] == 0


@pytest.mark.parametrize(
    "text",
    [
        "Title\nScripture",
        "Title\nChapter 2.\nScripture",
        "Chapter 1.\n   \n",
        "Chapter 1.\nChapter 1.\nScripture",
        "Chapter 1.\nThis set of files contains a script",
    ],
)
def test_invalid_or_empty_chapter_rejected(text: str) -> None:
    with pytest.raises(BuildError):
        clean_chapter(text, 1, "fixture")


def test_ordinary_chapter_word_and_other_parentheses_preserved() -> None:
    text, _ = clean_chapter("Chapter 1.\nAn ordinary chapter; (remember this).", 1, "fixture")
    assert text == "An ordinary chapter; (remember this)."


def test_approved_jps_markers_are_removed_conservatively() -> None:
    prose, audit = clean_chapter(
        "Psalms.\nChapter 42.\nBOOK II (42-1) For the Leader; (42-2) As the hart.\n",
        42,
        "fixture",
        jps=True,
        book_code="PSA",
    )
    assert prose == "For the Leader; As the hart."
    assert audit["jps_inline_references_removed"] == 2
    assert audit["jps_book_labels_removed"] == 1
    text, _ = clean_chapter(
        "Malachi.\nChapter 4.\n(3-22) Remember ye (the law).\n",
        4,
        "fixture",
        jps=True,
        book_code="MAL",
    )
    assert text == "Remember ye (the law)."
    with pytest.raises(BuildError, match="Unexpected JPS BOOK"):
        clean_chapter("Chapter 2.\nBOOK II Text.", 2, "fixture", jps=True, book_code="PSA")


def test_jps_cleanup_never_rewrites_kjv_markers() -> None:
    text, audit = clean_chapter("Chapter 1.\n(1-1) Keep unchanged.", 1, "fixture")
    assert text == "(1-1) Keep unchanged."
    assert audit["jps_inline_references_removed"] == 0


def test_real_layout_mapping_and_complete_document_invariant(source_fixture: Path) -> None:
    plan = inspect_sources(source_fixture)
    assert not plan.errors, plan.errors
    plan.validate()
    assert len(plan.documents) == 180
    assert plan.counts["jps_books_discovered"] == 39
    assert plan.counts["jps_chapters_discovered"] == 929
    assert plan.counts["kjv_nt_books_discovered"] == 27
    assert plan.counts["kjv_nt_chapters_discovered"] == 260
    assert plan.counts["kjv_ot_chapters_excluded"] == 50
    assert plan.counts["metadata_files_excluded"] == 6
    docs = {doc.id: doc for doc in plan.documents}
    genesis = docs["hb_genesis"].text
    assert "jps GEN" in genesis and "kjv GEN" not in genesis
    assert "kjv MAT" in docs["nt_matthew"].text
    assert "nt_genesis" not in docs
    assert genesis.index("GEN 2;") < genesis.index("GEN 10;")
    assert genesis.count("\n\n") == 49
    assert all(doc.text.endswith("\n") and not doc.text.endswith("\n\n") for doc in plan.documents)
    assert all(
        "Chapter " not in doc.text
        and "Editorial title" not in doc.text
        and "This set of files" not in doc.text
        for doc in plan.documents
    )
    assert len([unit for unit in plan.units if unit["role"] == "excluded_kjv_old_testament"]) == 50
    plan.documents.pop()
    with pytest.raises(BuildError, match="180"):
        plan.validate()


def test_metadata_not_scripture_and_unknown_filename_fails(tmp_path: Path) -> None:
    directory = tmp_path / "sources/ebible/jps"
    directory.mkdir(parents=True)
    (directory / "engjps_000_000_000_read.txt").write_text("Metadata only", encoding="utf-8")
    (directory / "surprise.txt").write_text("Unknown", encoding="utf-8")
    plan = BuildPlan()
    scan_ebible(tmp_path, "jps", plan)
    assert plan.counts["metadata_files_excluded"] == 1
    assert not plan.documents
    assert any("Unexpected eBible filename" in error for error in plan.errors)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "wrong_index", "decoding"])
def test_bad_chapter_inventory_rejected(writable_sources: Path, mutation: str) -> None:
    path = chapter_path(writable_sources, "jps", HEBREW_BOOKS[0], 1)
    if mutation == "duplicate":
        path.with_name(path.name.replace("_01_", "_001_")).write_bytes(path.read_bytes())
    elif mutation == "missing":
        path.unlink()
    elif mutation == "wrong_index":
        path.rename(path.with_name(path.name.replace("_002_", "_001_")))
    else:
        path.write_bytes(b"\xff\xfe")
    plan = inspect_sources(writable_sources)
    assert plan.errors
    with pytest.raises(BuildError, match="inspection failed"):
        build(writable_sources, plan)
    assert not (writable_sources / "corpus").exists()


def test_pickthall_reference_removal_and_numeric_ayah_order() -> None:
    lines = quran_fixture().splitlines()
    lines[:7] = reversed(lines[:7])
    parsed = parse_pickthall("\n".join(lines))
    assert len(parsed) == 114
    assert sum(map(len, parsed.values())) == 6236
    assert parsed[1][1] == "Text 1-1; 'punctuation.'"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_surah",
        "duplicate_surah",
        "duplicate_ayah",
        "missing_ayah",
        "missing_last_ayah",
        "malformed",
        "noninteger",
        "empty",
        "zero_ayah",
        "unknown_comment",
    ],
)
def test_pickthall_strict_validation(mutation: str) -> None:
    lines = quran_fixture().splitlines()
    if mutation == "missing_surah":
        lines = [line for line in lines if not line.startswith("114|")]
    elif mutation == "duplicate_surah":
        lines.append("1|1|Repeated block")
    elif mutation == "duplicate_ayah":
        lines.insert(1, lines[0])
    elif mutation == "missing_ayah":
        lines.pop(1)
    elif mutation == "missing_last_ayah":
        lines = [line for line in lines if not line.startswith("114|6|")]
    else:
        lines[0] = {
            "malformed": "1|1",
            "noninteger": "one|1|text",
            "empty": "1|1|  ",
            "zero_ayah": "1|0|text",
            "unknown_comment": "// unrecognized comment",
        }[mutation]
    with pytest.raises(BuildError):
        parse_pickthall("\n".join(lines))


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_dry_run_writes_nothing(source_fixture: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = snapshot(source_fixture)
    assert main(["--root", str(source_fixture), "--dry-run"]) == 0
    assert snapshot(source_fixture) == before
    assert "180" in capsys.readouterr().out


def test_build_manifest_hashes_reports_and_safe_force(writable_sources: Path) -> None:
    plan = inspect_sources(writable_sources)
    build(writable_sources, plan)
    manifest_path = writable_sources / "corpus/corpus_manifest.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["license_policy"] == "public-domain-only"
    assert len(manifest["documents"]) == 180
    for entry in manifest["documents"]:
        raw = (writable_sources / "corpus" / entry["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert b"\r" not in raw
    quran = (writable_sources / "corpus/raw/quran/001.txt").read_text(encoding="utf-8")
    assert "|" not in quran and "#" not in quran
    assert quran.startswith("Text 1-1;")
    assert quran.index("Text 1-2;") < quran.index("Text 1-7;")
    provenance = json.loads(
        (writable_sources / "research/source_provenance.json").read_text("utf-8")
    )
    assert provenance["builder_version"] == BUILDER_VERSION
    assert provenance["kjv_old_testament_excluded"] is True
    assert len([source for source in provenance["sources"] if source["path"].endswith(".zip")]) == 2
    for source in provenance["sources"]:
        assert not Path(source["path"]).is_absolute()
        assert "\\" not in source["path"]
        payload = (writable_sources / source["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["sha256"]
        assert len(payload) == source["byte_size"]
    audit = cleaning_report(plan)
    assert audit["aggregate"]["chapter_headings_removed"] == 1189
    assert audit["aggregate"]["pilcrow_markers_removed"] == 1189
    assert audit["aggregate"]["total_characters"] == sum(len(doc.text) for doc in plan.documents)
    initial = snapshot(writable_sources)
    with pytest.raises(BuildError, match="--force"):
        build(writable_sources, plan)
    sentinel = writable_sources / "corpus/raw/hebrew_bible/notes.md"
    sentinel.write_text("Keep me", encoding="utf-8")
    build(writable_sources, inspect_sources(writable_sources), force=True)
    after = snapshot(writable_sources)
    assert all(after[name] == content for name, content in initial.items())
    assert sentinel.read_text("utf-8") == "Keep me"
    stray = writable_sources / "corpus/raw/new_testament/genesis.txt"
    stray.write_text("Unowned OT text", encoding="utf-8")
    with pytest.raises(BuildError, match="Unknown corpus"):
        build(writable_sources, plan, force=True)
    assert stray.read_text("utf-8") == "Unowned OT text"


def test_existing_unowned_corpus_refuses_even_force(writable_sources: Path) -> None:
    plan = inspect_sources(writable_sources)
    target = writable_sources / "corpus/raw/hebrew_bible/genesis.txt"
    target.parent.mkdir(parents=True)
    target.write_text("Existing user text", encoding="utf-8")
    with pytest.raises(BuildError, match="--force"):
        build(writable_sources, plan)
    with pytest.raises(BuildError, match="ownership"):
        build(writable_sources, plan, force=True)
    assert target.read_text("utf-8") == "Existing user text"


def test_source_change_after_inspection_aborts(writable_sources: Path) -> None:
    plan = inspect_sources(writable_sources)
    chapter_path(writable_sources, "jps", HEBREW_BOOKS[0], 1).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(BuildError, match="Source changed"):
        build(writable_sources, plan)
    assert not (writable_sources / "corpus").exists()


def test_manifest_and_provenance_are_deterministic_and_portable(
    source_fixture: Path, writable_sources: Path
) -> None:
    first = inspect_sources(source_fixture)
    second = inspect_sources(writable_sources)
    assert not first.errors
    assert manifest_text(first) == manifest_text(second)
    assert provenance_report(first) == provenance_report(second)
    assert cleaning_report(first) == cleaning_report(second)


def test_malicious_output_mapping_cannot_escape_root(source_fixture: Path, tmp_path: Path) -> None:
    plan = inspect_sources(source_fixture)
    doc = plan.documents[0]
    plan.documents[0] = Document(doc.id, doc.family, "../elsewhere.txt", doc.text)
    with pytest.raises(BuildError, match="mapping"):
        build(tmp_path, plan)
    assert list(tmp_path.iterdir()) == []


def test_destination_symlink_is_rejected(source_fixture: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "project"
    (root / "corpus/raw").mkdir(parents=True)
    try:
        (root / "corpus/raw/hebrew_bible").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    with pytest.raises(BuildError, match="escapes|Linked"):
        validate_destinations(root, inspect_sources(source_fixture), force=True)
