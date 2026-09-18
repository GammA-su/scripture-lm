"""Deterministic, offline construction of core-canon-v1 from inspected sources.

Observed eBible layout: <translation>_<archive index>_<USFM code>_<chapter>_read.txt.
Indices are 002..040 for GEN..MAL and 070..096 for MAT..REV (not 040..066).
Psalms uses three chapter digits; other observed books use two. Identification
uses both the explicit archive index and USFM code, never alphabetical order.
The chapter counts below were checked against the supplied extracted archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUILDER_VERSION = "core-canon-builder-v1"


@dataclass(frozen=True)
class Book:
    index: int
    code: str
    slug: str
    chapters: int


# Explicit mappings observed in both archives, including JPS's 39-book decomposition.
HEBREW_BOOKS = (
    Book(2, "GEN", "genesis", 50),
    Book(3, "EXO", "exodus", 40),
    Book(4, "LEV", "leviticus", 27),
    Book(5, "NUM", "numbers", 36),
    Book(6, "DEU", "deuteronomy", 34),
    Book(7, "JOS", "joshua", 24),
    Book(8, "JDG", "judges", 21),
    Book(9, "RUT", "ruth", 4),
    Book(10, "1SA", "1_samuel", 31),
    Book(11, "2SA", "2_samuel", 24),
    Book(12, "1KI", "1_kings", 22),
    Book(13, "2KI", "2_kings", 25),
    Book(14, "1CH", "1_chronicles", 29),
    Book(15, "2CH", "2_chronicles", 36),
    Book(16, "EZR", "ezra", 10),
    Book(17, "NEH", "nehemiah", 13),
    Book(18, "EST", "esther", 10),
    Book(19, "JOB", "job", 42),
    Book(20, "PSA", "psalms", 150),
    Book(21, "PRO", "proverbs", 31),
    Book(22, "ECC", "ecclesiastes", 12),
    Book(23, "SNG", "song_of_songs", 8),
    Book(24, "ISA", "isaiah", 66),
    Book(25, "JER", "jeremiah", 52),
    Book(26, "LAM", "lamentations", 5),
    Book(27, "EZK", "ezekiel", 48),
    Book(28, "DAN", "daniel", 12),
    Book(29, "HOS", "hosea", 14),
    Book(30, "JOL", "joel", 3),
    Book(31, "AMO", "amos", 9),
    Book(32, "OBA", "obadiah", 1),
    Book(33, "JON", "jonah", 4),
    Book(34, "MIC", "micah", 7),
    Book(35, "NAM", "nahum", 3),
    Book(36, "HAB", "habakkuk", 3),
    Book(37, "ZEP", "zephaniah", 3),
    Book(38, "HAG", "haggai", 2),
    Book(39, "ZEC", "zechariah", 14),
    Book(40, "MAL", "malachi", 4),
)
NT_BOOKS = (
    Book(70, "MAT", "matthew", 28),
    Book(71, "MRK", "mark", 16),
    Book(72, "LUK", "luke", 24),
    Book(73, "JHN", "john", 21),
    Book(74, "ACT", "acts", 28),
    Book(75, "ROM", "romans", 16),
    Book(76, "1CO", "1_corinthians", 16),
    Book(77, "2CO", "2_corinthians", 13),
    Book(78, "GAL", "galatians", 6),
    Book(79, "EPH", "ephesians", 6),
    Book(80, "PHP", "philippians", 4),
    Book(81, "COL", "colossians", 4),
    Book(82, "1TH", "1_thessalonians", 5),
    Book(83, "2TH", "2_thessalonians", 3),
    Book(84, "1TI", "1_timothy", 6),
    Book(85, "2TI", "2_timothy", 4),
    Book(86, "TIT", "titus", 3),
    Book(87, "PHM", "philemon", 1),
    Book(88, "HEB", "hebrews", 13),
    Book(89, "JAS", "james", 5),
    Book(90, "1PE", "1_peter", 5),
    Book(91, "2PE", "2_peter", 3),
    Book(92, "1JN", "1_john", 5),
    Book(93, "2JN", "2_john", 1),
    Book(94, "3JN", "3_john", 1),
    Book(95, "JUD", "jude", 1),
    Book(96, "REV", "revelation", 22),
)
BOOK_BY_CODE = {book.code: book for book in (*HEBREW_BOOKS, *NT_BOOKS)}
# Counts observed in the supplied Tanzil en.pickthall file (6,236 ayahs).
# Pinning them catches a missing last ayah as well as internal reference gaps.
PICKTHALL_AYAH_COUNTS = (
    7,
    286,
    200,
    176,
    120,
    165,
    206,
    75,
    129,
    109,
    123,
    111,
    43,
    52,
    99,
    128,
    111,
    110,
    98,
    135,
    112,
    78,
    118,
    64,
    77,
    227,
    93,
    88,
    69,
    60,
    34,
    30,
    73,
    54,
    45,
    83,
    182,
    88,
    75,
    85,
    54,
    53,
    89,
    59,
    37,
    35,
    38,
    29,
    18,
    45,
    60,
    49,
    62,
    55,
    78,
    96,
    29,
    22,
    24,
    13,
    14,
    11,
    11,
    18,
    12,
    12,
    30,
    52,
    52,
    44,
    28,
    28,
    20,
    56,
    40,
    31,
    50,
    40,
    46,
    42,
    29,
    19,
    36,
    25,
    22,
    17,
    19,
    26,
    30,
    20,
    15,
    21,
    11,
    8,
    8,
    19,
    5,
    8,
    8,
    11,
    11,
    8,
    3,
    9,
    5,
    4,
    7,
    3,
    6,
    3,
    5,
    4,
    5,
    6,
)
CHAPTER_HEADING = re.compile(r"^Chapter\s+([0-9]+)\.\s*$")
PILCROW = re.compile(r"(?<!\S)¶[ \t]*")
INLINE_JPS_REFERENCE = re.compile(r"(?<!\S)\([0-9]+-[0-9]+\)(?=\s|$)[ \t]*")
PSALM_BOOK = re.compile(r"^BOOK (I|II|III|IV|V)(?=\s)[ \t]*")
PSALM_BOOK_LABELS = {1: "I", 42: "II", 73: "III", 90: "IV", 107: "V"}
PARATEXT = re.compile(
    r"This set of files contains a script|for the purpose of reading to make an audio recording"
    r"|(?<!\w)Chapter[ \t]+[0-9]+\.(?=\s|$)",
    re.IGNORECASE,
)


class BuildError(ValueError):
    """An input or destination would violate the corpus contract."""


@dataclass(frozen=True)
class Document:
    id: str
    family: str
    path: str
    text: str

    @property
    def payload(self) -> bytes:
        return self.text.encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass
class BuildPlan:
    documents: list[Document] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    units: list[dict[str, Any]] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    def validate(self) -> None:
        expected = {"hebrew_bible": 39, "new_testament": 27, "quran": 114}
        actual = Counter(document.family for document in self.documents)
        if actual != expected or len(self.documents) != 180:
            raise BuildError(f"Expected 39 + 27 + 114 = 180 documents; found {dict(actual)}")
        if len({doc.id for doc in self.documents}) != 180:
            raise BuildError("Duplicate final document IDs")
        if len({doc.path for doc in self.documents}) != 180:
            raise BuildError("Duplicate final document paths")
        expected_documents = (
            {
                (f"hb_{book.slug}", "hebrew_bible", f"raw/hebrew_bible/{book.slug}.txt")
                for book in HEBREW_BOOKS
            }
            | {
                (f"nt_{book.slug}", "new_testament", f"raw/new_testament/{book.slug}.txt")
                for book in NT_BOOKS
            }
            | {(f"quran_{n:03d}", "quran", f"raw/quran/{n:03d}.txt") for n in range(1, 115)}
        )
        if {(doc.id, doc.family, doc.path) for doc in self.documents} != expected_documents:
            raise BuildError("Unexpected final book mapping, including possible KJV OT output")
        for document in self.documents:
            validate_prose(document.text, document.path)
            if not document.text.endswith("\n") or document.text.endswith("\n\n"):
                raise BuildError(f"Document must end with exactly one newline: {document.path}")


def safe_path(root: Path, relative: str) -> Path:
    """Reject traversal, symlinks, and junctions before any source read or output write."""
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise BuildError(f"Path escapes project root: {relative}")
    for part in (candidate, *candidate.parents):
        if part == root:
            break
        if part.is_symlink() or part.is_junction():
            raise BuildError(f"Linked paths are not allowed: {relative}")
    return candidate


def read_source(root: Path, relative: str, translation: str, role: str, plan: BuildPlan) -> str:
    path = safe_path(root, relative)
    raw = path.read_bytes()
    # utf-8-sig tolerates only a leading UTF-8 BOM; decoding errors are never replaced.
    text = raw.decode("utf-8-sig", errors="strict")
    if "\x00" in text:
        raise BuildError(f"NUL byte in text source: {relative}")
    plan.sources.append(
        {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_size": len(raw),
            "translation": translation,
            "role": role,
            "builder_version": BUILDER_VERSION,
        }
    )
    return text


def validate_prose(text: str, source: str) -> None:
    if not text.strip():
        raise BuildError(f"Empty scripture text: {source}")
    if PARATEXT.search(text):
        raise BuildError(f"Known eBible paratext remains in scripture: {source}")


def clean_chapter(
    text: str, chapter: int, source: str, *, jps: bool = False, book_code: str = ""
) -> tuple[str, dict[str, int]]:
    """Remove explicit headings and whitespace-delimited pilcrows; preserve other prose."""
    lines = text.splitlines()
    headings = [
        (i, match)
        for i, line in enumerate(lines)
        if (match := CHAPTER_HEADING.fullmatch(line)) is not None
    ]
    if len(headings) != 1:
        raise BuildError(
            f"Expected exactly one Chapter N. heading: {source}; found {len(headings)}"
        )
    position, heading = headings[0]
    if int(heading.group(1)) != chapter:
        raise BuildError(f"Chapter heading disagrees with filename: {source}")
    prose = []
    pilcrows = 0
    references = 0
    book_labels = 0
    for line in lines[position + 1 :]:
        cleaned, count = PILCROW.subn("", line.strip())
        pilcrows += count
        if jps:
            label = PSALM_BOOK.match(cleaned)
            if label:
                if book_code != "PSA" or PSALM_BOOK_LABELS.get(chapter) != label.group(1):
                    raise BuildError(f"Unexpected JPS BOOK section label: {source}")
                cleaned = cleaned[label.end() :]
                book_labels += 1
            # User-approved exact editorial markers. They also occur outside Psalms,
            # reflecting JPS versification; their numbers need not match the file chapter.
            cleaned, removed = INLINE_JPS_REFERENCE.subn("", cleaned)
            references += removed
        if cleaned.strip():
            prose.append(cleaned.strip())
    result = " ".join(prose)
    validate_prose(result, source)
    return result, {
        "source_lines": len(lines),
        "lines_discarded_before_heading": position,
        "nonempty_header_lines_removed": sum(bool(line.strip()) for line in lines[:position]),
        "chapter_headings_removed": 1,
        "pilcrow_markers_removed": pilcrows,
        "jps_inline_references_removed": references,
        "jps_book_labels_removed": book_labels,
        "scripture_characters_retained": len(result),
    }


def scan_ebible(root: Path, source: str, plan: BuildPlan) -> None:
    """Inspect every extracted file, explicitly excluding metadata and the KJV OT."""
    is_jps = source == "jps"
    prefix = "engjps" if is_jps else "eng-kjv2006"
    translation = "JPS 1917" if is_jps else "King James Version (eng-kjv2006)"
    family = "hebrew_bible" if is_jps else "new_testament"
    selected = HEBREW_BOOKS if is_jps else NT_BOOKS
    directory = safe_path(root, f"sources/ebible/{source}")
    pattern = re.compile(rf"{prefix}_([0-9]{{3}})_([A-Z0-9]{{3}})_([0-9]{{2,3}})_read\.txt")
    metadata = {f"{prefix}_000_000_000_read.txt", "copr.htm", "keys.asc"}
    discovered: dict[str, dict[int, str]] = {}
    cleaned: dict[str, dict[int, str]] = {}
    if not directory.is_dir():
        raise BuildError(f"Missing source directory: sources/ebible/{source}")
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        relative = path.relative_to(root).as_posix()
        try:
            if path.name in metadata:
                read_source(root, relative, translation, "excluded_metadata", plan)
                plan.counts["metadata_files_excluded"] += 1
                continue
            match = pattern.fullmatch(path.name)
            if not match or not path.is_file():
                raise BuildError(f"Unexpected eBible filename pattern: {relative}")
            index, code, number = match.groups()
            book = BOOK_BY_CODE.get(code)
            if book is None or book.index != int(index) or (is_jps and book not in HEBREW_BOOKS):
                raise BuildError(f"Unsupported archive book mapping: {relative}")
            chapter = int(number)
            chapters = discovered.setdefault(code, {})
            if chapter in chapters:
                raise BuildError(f"Duplicate chapter {code} {chapter}: {relative}")
            chapters[chapter] = relative
            include = book in selected
            role = family if include else "excluded_kjv_old_testament"
            raw = read_source(root, relative, translation, role, plan)
            doc_id = ("hb_" if is_jps else "nt_") + book.slug if include else None
            record: dict[str, Any] = {
                "source_filename": relative,
                "final_document_id": doc_id,
                "chapter_number": chapter,
                "surah_number": None,
                "role": role,
                "source_lines": len(raw.splitlines()),
                "lines_discarded_before_heading": 0,
                "nonempty_header_lines_removed": 0,
                "chapter_headings_removed": 0,
                "pilcrow_markers_removed": 0,
                "scripture_characters_retained": 0,
                "jps_inline_references_removed": 0,
                "jps_book_labels_removed": 0,
            }
            plan.units.append(record)
            prose, audit = clean_chapter(raw, chapter, relative, jps=is_jps, book_code=code)
            if not include:
                plan.counts["kjv_ot_chapters_excluded"] += 1
                continue
            cleaned.setdefault(code, {})[chapter] = prose
            record.update(audit)
        except (OSError, UnicodeError, BuildError) as exc:
            plan.errors.append(str(exc))
    label = "jps" if is_jps else "kjv_nt"
    plan.counts[f"{label}_books_discovered"] = sum(book.code in discovered for book in selected)
    plan.counts[f"{label}_chapters_discovered"] = sum(
        len(discovered.get(book.code, {})) for book in selected
    )
    for book in HEBREW_BOOKS if is_jps else (*HEBREW_BOOKS, *NT_BOOKS):
        actual = set(discovered.get(book.code, {}))
        if not actual and book not in selected:
            continue
        expected = set(range(1, book.chapters + 1))
        if actual != expected:
            plan.errors.append(
                f"Missing/unexpected chapters for {source}/{book.code}: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
        if book in selected and actual == expected and set(cleaned.get(book.code, {})) == expected:
            content = "\n\n".join(cleaned[book.code][n] for n in sorted(expected)) + "\n"
            doc_id = ("hb_" if is_jps else "nt_") + book.slug
            plan.documents.append(
                Document(doc_id, family, f"raw/{family}/{book.slug}.txt", content)
            )


def parse_pickthall(text: str) -> dict[int, dict[int, str]]:
    """Parse strict positive references; only blank lines and '#' comments are ignored.

    Surahs form unique contiguous blocks; ayahs may be out of order within a block.
    Exact observed ayah counts detect both internal gaps and truncated surahs.
    """
    surahs: dict[int, dict[int, str]] = {}
    current = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3 or not all(re.fullmatch(r"[0-9]+", part) for part in parts[:2]):
            raise BuildError(f"Malformed Pickthall row at line {line_number}")
        surah, ayah = int(parts[0]), int(parts[1])
        if not 1 <= surah <= 114 or ayah < 1:
            raise BuildError(f"Invalid Pickthall reference at line {line_number}")
        if current != surah:
            if surah in surahs:
                raise BuildError(f"Duplicate surah block: {surah}")
            current = surah
            surahs[surah] = {}
        if ayah in surahs[surah]:
            raise BuildError(f"Duplicate ayah: {surah}|{ayah}")
        scripture = parts[2].strip()
        validate_prose(scripture, f"Pickthall {surah}|{ayah}")
        surahs[surah][ayah] = scripture
    if set(surahs) != set(range(1, 115)):
        raise BuildError(f"Pickthall must contain exactly surahs 1..114; found {len(surahs)}")
    for surah, ayahs in surahs.items():
        if sorted(ayahs) != list(range(1, PICKTHALL_AYAH_COUNTS[surah - 1] + 1)):
            raise BuildError(f"Missing ayah references in surah {surah}")
    return surahs


def scan_pickthall(root: Path, plan: BuildPlan) -> None:
    relative = "sources/quran/pickthall.txt"
    text = read_source(root, relative, "Pickthall 1930", "quran", plan)
    surahs = parse_pickthall(text)
    plan.counts["pickthall_surahs_discovered"] = len(surahs)
    plan.counts["pickthall_ayahs_discovered"] = sum(len(ayahs) for ayahs in surahs.values())
    plan.counts["pickthall_comment_lines_excluded"] = sum(
        line.startswith("#") for line in text.splitlines()
    )
    for surah, ayahs in sorted(surahs.items()):
        prose = " ".join(ayahs[ayah] for ayah in sorted(ayahs))
        doc_id = f"quran_{surah:03d}"
        plan.documents.append(Document(doc_id, "quran", f"raw/quran/{surah:03d}.txt", prose + "\n"))
        plan.units.append(
            {
                "source_filename": relative,
                "final_document_id": doc_id,
                "chapter_number": None,
                "surah_number": surah,
                "role": "quran",
                "source_lines": len(ayahs),
                "lines_discarded_before_heading": 0,
                "nonempty_header_lines_removed": 0,
                "chapter_headings_removed": 0,
                "pilcrow_markers_removed": 0,
                "scripture_characters_retained": len(prose),
                "jps_inline_references_removed": 0,
                "jps_book_labels_removed": 0,
            }
        )


def inspect_sources(root: Path) -> BuildPlan:
    root = root.resolve()
    plan = BuildPlan()
    for source in ("jps", "kjv"):
        try:
            scan_ebible(root, source, plan)
        except (OSError, UnicodeError, BuildError) as exc:
            plan.errors.append(str(exc))
    try:
        scan_pickthall(root, plan)
    except (OSError, UnicodeError, BuildError) as exc:
        plan.errors.append(str(exc))
    for archive, translation in (
        ("engjps_readaloud.zip", "JPS 1917"),
        ("eng-kjv2006_readaloud.zip", "King James Version (eng-kjv2006)"),
    ):
        relative = f"sources/ebible/{archive}"
        try:
            path = safe_path(root, relative)
            if path.exists():
                raw = path.read_bytes()
                plan.sources.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "byte_size": len(raw),
                        "translation": translation,
                        "role": "archive_provenance_only",
                        "builder_version": BUILDER_VERSION,
                    }
                )
        except (OSError, BuildError) as exc:
            plan.errors.append(str(exc))
    plan.sources.sort(key=lambda record: record["path"])
    try:
        plan.validate()
    except BuildError as exc:
        plan.errors.append(str(exc))
    return plan


def manifest_text(plan: BuildPlan) -> str:
    lines = ['version = "1.0"', 'name = "core-canon-v1"', 'license_policy = "public-domain-only"']
    for document in plan.documents:
        lines.extend(["", "[[documents]]"])
        for key, value in (
            ("id", document.id),
            ("family", document.family),
            ("path", document.path),
            ("license", "public-domain"),
            ("sha256", document.sha256),
        ):
            lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def provenance_report(plan: BuildPlan) -> dict[str, Any]:
    return {
        "builder_version": BUILDER_VERSION,
        "translation_roles": {
            "JPS 1917": "Hebrew Bible",
            "KJV": "New Testament only",
            "Pickthall 1930": "Qur'an",
        },
        "kjv_old_testament_excluded": True,
        "exclusion_reason": "Avoid duplicating underlying Hebrew Bible / Old Testament scripture",
        "sources": plan.sources,
        "outputs": [
            {"path": "corpus/" + doc.path, "id": doc.id, "sha256": doc.sha256}
            for doc in plan.documents
        ],
    }


def cleaning_report(plan: BuildPlan) -> dict[str, Any]:
    aggregate: dict[str, Any] = dict(plan.counts)
    for key in (
        "chapter_headings_removed",
        "nonempty_header_lines_removed",
        "pilcrow_markers_removed",
        "jps_inline_references_removed",
        "jps_book_labels_removed",
    ):
        aggregate[key] = sum(unit[key] for unit in plan.units)
    for family in ("hebrew_bible", "new_testament", "quran"):
        aggregate[f"{family}_characters"] = sum(
            len(doc.text) for doc in plan.documents if doc.family == family
        )
    aggregate["total_characters"] = sum(len(doc.text) for doc in plan.documents)
    aggregate["total_documents"] = len(plan.documents)
    return {
        "builder_version": BUILDER_VERSION,
        "aggregate": aggregate,
        "units": plan.units,
        "character_count_definition": "Unicode codepoints; document totals include separators "
        "and the final newline, unit retained counts exclude generated separators.",
    }


def validate_destinations(root: Path, plan: BuildPlan, *, force: bool) -> None:
    expected = {"corpus/" + doc.path for doc in plan.documents}
    existing: set[str] = set()
    for family in ("hebrew_bible", "new_testament", "quran"):
        directory = safe_path(root, f"corpus/raw/{family}")
        for path in directory.rglob("*") if directory.exists() else ():
            relative = path.relative_to(root).as_posix()
            safe_path(root, relative)
            if path.suffix.lower() == ".txt":
                existing.add(relative)
    if existing and not force:
        raise BuildError("Existing corpus .txt files found; no overwrite without explicit --force")
    if existing - expected:
        raise BuildError(
            "Unknown corpus .txt files would contaminate corpus; never removed: "
            + ", ".join(sorted(existing - expected))
        )
    provenance_path = safe_path(root, "research/source_provenance.json")
    report_path = safe_path(root, "research/corpus_build_report.json")
    if existing:
        if not provenance_path.is_file():
            raise BuildError(
                "Existing files have no builder ownership record; --force cannot replace"
            )
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        owned = {entry["path"] for entry in previous.get("outputs", [])}
        if previous.get("builder_version") != BUILDER_VERSION or not existing <= owned:
            raise BuildError("Existing files are not owned by this corpus builder")
    for path in (provenance_path, report_path):
        if path.exists():
            if not force:
                raise BuildError(f"Existing builder report requires --force: {path.name}")
            if (
                json.loads(path.read_text(encoding="utf-8")).get("builder_version")
                != BUILDER_VERSION
            ):
                raise BuildError(f"Refusing to overwrite unrelated report: {path.name}")
    manifest = safe_path(root, "corpus/corpus_manifest.toml")
    if manifest.exists():
        old = tomllib.loads(manifest.read_text(encoding="utf-8"))
        if old.get("documents") and (not force or not provenance_path.is_file()):
            raise BuildError("Existing populated manifest cannot be replaced without owned --force")
        if any("corpus/" + doc["path"] not in expected for doc in old.get("documents", [])):
            raise BuildError(
                "Existing manifest contains documents outside this builder's ownership"
            )
    for relative in expected:
        target = safe_path(root, relative)
        if target.exists() and not target.is_file():
            raise BuildError(f"Output path is not a regular file: {relative}")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".core-canon-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(root: Path, plan: BuildPlan, *, force: bool = False) -> None:
    """Write only validated documents; publish the manifest last, after hash verification."""
    root = root.resolve()
    if plan.errors:
        raise BuildError("Source inspection failed; corpus construction refused")
    plan.validate()
    validate_destinations(root, plan, force=force)
    # Detect any input changed since inspection, before touching the final corpus.
    for source in plan.sources:
        current = safe_path(root, source["path"]).read_bytes()
        if hashlib.sha256(current).hexdigest() != source["sha256"]:
            raise BuildError(f"Source changed after inspection: {source['path']}")
    for document in plan.documents:
        atomic_write(safe_path(root, "corpus/" + document.path), document.payload)
    for document in plan.documents:
        data = safe_path(root, "corpus/" + document.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != document.sha256:
            raise BuildError(f"Written document hash mismatch: {document.path}")
    for relative, content in (
        ("research/source_provenance.json", provenance_report(plan)),
        ("research/corpus_build_report.json", cleaning_report(plan)),
    ):
        payload = json.dumps(content, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        atomic_write(safe_path(root, relative), payload.encode("utf-8"))
    atomic_write(
        safe_path(root, "corpus/corpus_manifest.toml"), manifest_text(plan).encode("utf-8")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Inspect/validate without writing")
    parser.add_argument(
        "--force", action="store_true", help="Replace only previously owned outputs"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: cwd)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plan = inspect_sources(root)
    for label, key in (
        ("JPS books discovered", "jps_books_discovered"),
        ("JPS chapters discovered", "jps_chapters_discovered"),
        ("KJV NT books discovered", "kjv_nt_books_discovered"),
        ("KJV NT chapters discovered", "kjv_nt_chapters_discovered"),
        ("KJV OT chapters deliberately excluded", "kjv_ot_chapters_excluded"),
        ("Pickthall surahs discovered", "pickthall_surahs_discovered"),
        ("Pickthall ayahs discovered", "pickthall_ayahs_discovered"),
        ("Metadata files excluded", "metadata_files_excluded"),
    ):
        print(f"{label}: {plan.counts[key]}")
    print(f"Validated final document count: {len(plan.documents)} (required: 180)")
    aggregate = cleaning_report(plan)["aggregate"]
    for label, key in (
        ("Chapter headings removed", "chapter_headings_removed"),
        ("Book-title/header lines removed", "nonempty_header_lines_removed"),
        ("Pilcrow markers removed", "pilcrow_markers_removed"),
        ("JPS inline references removed", "jps_inline_references_removed"),
        ("JPS BOOK section labels removed", "jps_book_labels_removed"),
        ("JPS characters", "hebrew_bible_characters"),
        ("NT characters", "new_testament_characters"),
        ("Qur'an characters", "quran_characters"),
        ("Total characters including document separators", "total_characters"),
    ):
        print(f"{label}: {aggregate[key]}")
    try:
        validate_destinations(root, plan, force=args.force)
    except (OSError, ValueError, KeyError) as exc:
        plan.errors.append(str(exc))
    if plan.errors:
        print(f"Validation failed ({len(plan.errors)} errors); no files written:")
        for error in plan.errors:
            print(f"  - {error}")
        return 1
    if args.dry_run:
        print(
            "Would create 180 scripture documents, corpus/corpus_manifest.toml, "
            "research/source_provenance.json, and research/corpus_build_report.json."
        )
        print("Dry-run complete; no files written.")
        return 0
    try:
        build(root, plan, force=args.force)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Build failed: {exc}")
        return 1
    print(
        "Hebrew Bible documents: 39\nNew Testament documents: 27\nQur'an documents: 114\nTotal: 180"
    )
    return 0
