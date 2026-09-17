"""Corpus manifest data models, validation, and SHA-256 checksum computation."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RegexReplacement(BaseModel):
    """Pair of pattern and replacement string for document cleanup."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    replacement: str = ""


class EditorialCleanupConfig(BaseModel):
    """Explicit, auditable regex rules for stripping editorial additions."""

    model_config = ConfigDict(extra="forbid")

    strip_header_patterns: list[str] = Field(default_factory=list)
    strip_footer_patterns: list[str] = Field(default_factory=list)
    line_prefix_patterns: list[str] = Field(default_factory=list)
    replace_patterns: list[RegexReplacement] = Field(default_factory=list)


class DocumentEntry(BaseModel):
    """Metadata entry for a single primary scripture document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    family: str
    path: str
    license: str
    sha256: str | None = None
    cleanup: EditorialCleanupConfig | None = None


class CorpusManifest(BaseModel):
    """Corpus manifest listing registered primary scripture documents."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    name: str = "core-canon-v1"
    license_policy: str = "public-domain"
    documents: list[DocumentEntry] = Field(default_factory=list)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file in binary mode."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_sha256(text: str) -> str:
    """Compute SHA-256 checksum of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_cleanup_regexes(cleanup: EditorialCleanupConfig, doc_id: str) -> list[str]:
    """Pre-compile all cleanup regexes to verify syntax validity at audit/load time."""
    errors: list[str] = []

    for idx, pat in enumerate(cleanup.strip_header_patterns):
        try:
            re.compile(pat)
        except re.error as e:
            errors.append(
                f"Invalid regex in document '{doc_id}' strip_header_patterns[{idx}] '{pat}': {e}"
            )

    for idx, pat in enumerate(cleanup.strip_footer_patterns):
        try:
            re.compile(pat)
        except re.error as e:
            errors.append(
                f"Invalid regex in document '{doc_id}' strip_footer_patterns[{idx}] '{pat}': {e}"
            )

    for idx, pat in enumerate(cleanup.line_prefix_patterns):
        try:
            re.compile(pat)
        except re.error as e:
            errors.append(
                f"Invalid regex in document '{doc_id}' line_prefix_patterns[{idx}] '{pat}': {e}"
            )

    for idx, rep in enumerate(cleanup.replace_patterns):
        try:
            re.compile(rep.pattern)
        except re.error as e:
            errors.append(
                f"Invalid regex in document '{doc_id}' replace_patterns[{idx}] '{rep.pattern}': {e}"
            )

    return errors


def validate_manifest(manifest: CorpusManifest, corpus_root: Path) -> list[str]:
    """Validate manifest integrity, path security, encoding, and checksums.

    Returns a list of error messages (empty list if manifest is fully valid).
    """
    errors: list[str] = []
    raw_dir = (corpus_root / "raw").resolve()

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()

    for doc in manifest.documents:
        # Check duplicate ID
        if doc.id in seen_ids:
            errors.append(f"Duplicate document ID: '{doc.id}'")
        seen_ids.add(doc.id)

        # Normalize relative path representation
        normalized_path_key = Path(doc.path).as_posix().lower()
        if normalized_path_key in seen_paths:
            errors.append(f"Duplicate document path: '{doc.path}'")
        seen_paths.add(normalized_path_key)

        # Path traversal check
        doc_full_path = (corpus_root / doc.path).resolve()
        try:
            doc_full_path.relative_to(raw_dir)
        except ValueError:
            errors.append(
                f"Path traversal detected for document '{doc.id}': '{doc.path}' "
                f"is outside corpus raw directory '{raw_dir}'"
            )
            continue

        # File existence check
        if not doc_full_path.is_file():
            errors.append(f"Document '{doc.id}' references non-existent file: '{doc.path}'")
            continue

        # File size and whitespace check
        try:
            with open(doc_full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError as e:
            errors.append(f"Document '{doc.id}' at '{doc.path}' is not valid UTF-8: {e}")
            continue
        except Exception as e:
            errors.append(f"Could not read document '{doc.id}' at '{doc.path}': {e}")
            continue

        if not content.strip():
            errors.append(
                f"Document '{doc.id}' at '{doc.path}' is empty or contains only whitespace"
            )
            continue

        # Checksum check if sha256 is present in manifest
        if doc.sha256:
            actual_sha = compute_file_sha256(doc_full_path)
            if actual_sha.lower() != doc.sha256.lower():
                errors.append(
                    f"SHA-256 mismatch for document '{doc.id}': "
                    f"expected {doc.sha256}, got {actual_sha}"
                )

        # Pre-compile cleanup regexes if configured
        if doc.cleanup:
            cleanup_errors = validate_cleanup_regexes(doc.cleanup, doc.id)
            errors.extend(cleanup_errors)

    return errors


def load_manifest(manifest_path: Path | str) -> CorpusManifest:
    """Load and parse corpus_manifest.toml into a validated CorpusManifest model."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found: {path}")

    with open(path, "rb") as f:
        raw_dict: dict[str, Any] = tomllib.load(f)

    return CorpusManifest.model_validate(raw_dict)
