"""Conservative scripture text normalization and provenance locking."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scripture_lm.corpus.manifest import (
    CorpusManifest,
    EditorialCleanupConfig,
    compute_file_sha256,
    compute_text_sha256,
)

NORMALIZATION_ALGORITHM = "conservative_v1"


class DocumentProvenance(BaseModel):
    """Immutable provenance record for a normalized scripture document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    family: str
    source_path: str
    source_sha256: str
    normalized_sha256: str
    raw_bytes: int
    raw_characters: int
    normalized_bytes: int
    normalized_characters: int
    lines_before: int
    lines_after: int
    cleanup_modifications: int
    cleanup_config: dict[str, Any] | None = None


class CorpusLock(BaseModel):
    """Provenance lock recording corpus fingerprints and document stats."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    normalization_algorithm: str = NORMALIZATION_ALGORITHM
    corpus_fingerprint: str
    normalization_fingerprint: str
    documents: list[DocumentProvenance] = Field(default_factory=list)


def conservative_normalize(
    text: str, cleanup: EditorialCleanupConfig | None = None
) -> tuple[str, int]:
    """Apply conservative, auditable normalization pipeline.

    Order of operations:
    1. Decode UTF-8 (assumed string input)
    2. Strip leading UTF-8 BOM
    3. Normalize newlines to Unix \n
    4. Unicode NFC normalization
    5. Explicit editorial regex cleanup (operating on canonical NFC + Unix \n)
    6. Trim line trailing whitespace
    7. Collapse ordinary horizontal whitespace within lines
    8. Normalize paragraph blank lines (collapse 3+ newlines to \n\n)
    9. Strip outer whitespace and append final \n

    Returns:
        tuple[normalized_text, cleanup_modifications_count]
    """
    cleanup_mods = 0

    # 2. Remove UTF-8 BOM
    text = text.removeprefix("\ufeff")

    # 3. Normalize newlines to Unix \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 5. Explicit editorial regex cleanup
    if cleanup:
        # 5a. strip_header_patterns: remove text only from the start of the document
        for pat in cleanup.strip_header_patterns:
            # Anchor at start of string (\A)
            compiled = re.compile(rf"\A(?:{pat})", re.DOTALL)
            new_text, count = compiled.subn("", text)
            if count > 0:
                cleanup_mods += count
                text = new_text

        # 5b. strip_footer_patterns: remove text only from the end of the document
        for pat in cleanup.strip_footer_patterns:
            # Anchor at end of string (\Z)
            compiled = re.compile(rf"(?:{pat})\Z", re.DOTALL)
            new_text, count = compiled.subn("", text)
            if count > 0:
                cleanup_mods += count
                text = new_text

        # 5c. line_prefix_patterns: applied independently to the beginning of each line
        if cleanup.line_prefix_patterns:
            lines = text.split("\n")
            modified_lines: list[str] = []
            for line in lines:
                current_line = line
                for pat in cleanup.line_prefix_patterns:
                    compiled = re.compile(rf"\A(?:{pat})")
                    current_line, count = compiled.subn("", current_line)
                    cleanup_mods += count
                modified_lines.append(current_line)
            text = "\n".join(modified_lines)

        # 5d. replace_patterns: global regex substitutions
        for rep in cleanup.replace_patterns:
            compiled = re.compile(rep.pattern)
            new_text, count = compiled.subn(rep.replacement, text)
            cleanup_mods += count
            text = new_text

    # 6. Trim trailing whitespace on lines & 7. Collapse ordinary horizontal whitespace within lines
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        # Collapse multiple horizontal whitespace characters (spaces/tabs) to a single space
        collapsed = re.sub(r"[^\S\n]+", " ", stripped)
        cleaned_lines.append(collapsed)
    text = "\n".join(cleaned_lines)

    # 8. Normalize paragraph blank lines (3+ consecutive newlines become \n\n)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 9. Strip outer whitespace and ensure single trailing newline
    text = text.strip()
    if not text:
        raise ValueError("Document is empty or contains only whitespace after normalization")

    text = text + "\n"

    return text, cleanup_mods


def compute_corpus_fingerprint(provenance_records: list[DocumentProvenance]) -> str:
    """Compute machine-independent corpus fingerprint from canonical sorted source records."""
    sorted_records = sorted(
        [[doc.document_id, doc.family, doc.source_sha256] for doc in provenance_records],
        key=lambda x: str(x[0]),
    )
    serialized = json.dumps(sorted_records, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_normalization_fingerprint(provenance_records: list[DocumentProvenance]) -> str:
    """Compute machine-independent normalization fingerprint from sorted normalized records."""
    sorted_records = sorted(
        [
            [
                doc.document_id,
                doc.normalized_sha256,
                doc.cleanup_config,
                NORMALIZATION_ALGORITHM,
            ]
            for doc in provenance_records
        ],
        key=lambda x: str(x[0]),
    )
    serialized = json.dumps(sorted_records, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_corpus(
    manifest: CorpusManifest,
    corpus_root: Path,
    output_dir: Path = Path("data/normalized"),
    lock_path: Path = Path("data/corpus_lock.json"),
) -> CorpusLock:
    """Normalize all manifest documents and write immutable data/normalized/ files and lockfile."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    provenance_records: list[DocumentProvenance] = []

    for doc in manifest.documents:
        source_file = (corpus_root / doc.path).resolve()
        if not source_file.is_file():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        # Raw file statistics
        raw_bytes = source_file.stat().st_size
        raw_text = source_file.read_text(encoding="utf-8")
        raw_chars = len(raw_text)
        lines_before = len(raw_text.splitlines())
        source_sha = compute_file_sha256(source_file)

        # Normalize
        normalized_text, cleanup_mods = conservative_normalize(raw_text, doc.cleanup)
        normalized_chars = len(normalized_text)
        normalized_bytes = len(normalized_text.encode("utf-8"))
        lines_after = len(normalized_text.splitlines())
        norm_sha = compute_text_sha256(normalized_text)

        # Write immutable normalized text
        family_dir = output_dir / doc.family
        family_dir.mkdir(parents=True, exist_ok=True)
        dest_file = family_dir / f"{doc.id}.txt"
        dest_file.write_text(normalized_text, encoding="utf-8", newline="\n")

        cleanup_dict = doc.cleanup.model_dump() if doc.cleanup else None

        record = DocumentProvenance(
            document_id=doc.id,
            family=doc.family,
            source_path=Path(doc.path).as_posix(),
            source_sha256=source_sha,
            normalized_sha256=norm_sha,
            raw_bytes=raw_bytes,
            raw_characters=raw_chars,
            normalized_bytes=normalized_bytes,
            normalized_characters=normalized_chars,
            lines_before=lines_before,
            lines_after=lines_after,
            cleanup_modifications=cleanup_mods,
            cleanup_config=cleanup_dict,
        )
        provenance_records.append(record)

    corpus_fp = compute_corpus_fingerprint(provenance_records)
    norm_fp = compute_normalization_fingerprint(provenance_records)

    lock = CorpusLock(
        version="1.0",
        normalization_algorithm=NORMALIZATION_ALGORITHM,
        corpus_fingerprint=corpus_fp,
        normalization_fingerprint=norm_fp,
        documents=provenance_records,
    )

    lock_path.write_text(json.dumps(lock.model_dump(), indent=2), encoding="utf-8", newline="\n")

    return lock
