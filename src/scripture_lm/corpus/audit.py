"""Corpus purity audit and integrity verification subsystem."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scripture_lm.corpus.manifest import (
    compute_file_sha256,
    load_manifest,
    validate_manifest,
)


class AuditResult(BaseModel):
    """Structured results from auditing a scripture corpus."""

    model_config = ConfigDict(extra="forbid")

    is_clean: bool = True
    num_documents: int = 0
    documents_per_family: dict[str, int] = Field(default_factory=dict)
    characters_per_family: dict[str, int] = Field(default_factory=dict)
    bytes_per_family: dict[str, int] = Field(default_factory=dict)
    sha256_verified: int = 0
    sha256_mismatches: list[str] = Field(default_factory=list)
    sha256_missing: list[str] = Field(default_factory=list)
    licenses: dict[str, int] = Field(default_factory=dict)
    untracked_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def scan_untracked_files(corpus_root: Path, registered_paths: set[Path]) -> list[str]:
    """Scan corpus/raw recursively for all regular files not in the manifest, ignoring .gitkeep."""
    raw_dir = (corpus_root / "raw").resolve()
    if not raw_dir.is_dir():
        return []

    untracked: list[str] = []
    for file_path in raw_dir.rglob("*"):
        if file_path.is_file():
            if file_path.name == ".gitkeep":
                continue
            resolved = file_path.resolve()
            if resolved not in registered_paths:
                rel_path = file_path.relative_to(corpus_root).as_posix()
                untracked.append(rel_path)

    return sorted(untracked)


def audit_corpus(
    manifest_path: Path | str = "corpus/corpus_manifest.toml",
    corpus_root: Path | str = "corpus",
) -> AuditResult:
    """Perform a comprehensive audit of the scripture corpus."""
    m_path = Path(manifest_path)
    c_root = Path(corpus_root)

    try:
        manifest = load_manifest(m_path)
    except Exception as e:
        return AuditResult(
            is_clean=False,
            errors=[f"Failed to load manifest '{manifest_path}': {e}"],
        )

    # Validate manifest documents (paths, uniqueness, traversal, utf-8, expected sha256)
    validation_errors = validate_manifest(manifest, c_root)

    # Collect registered paths
    registered_paths: set[Path] = set()
    docs_per_family: dict[str, int] = {}
    chars_per_family: dict[str, int] = {}
    bytes_per_family: dict[str, int] = {}
    licenses: dict[str, int] = {}
    sha_verified = 0
    sha_mismatches: list[str] = []
    sha_missing: list[str] = []
    warnings: list[str] = []

    if not manifest.documents:
        warnings.append("Corpus manifest contains 0 documents. Corpus is currently empty.")

    for doc in manifest.documents:
        full_path = (c_root / doc.path).resolve()
        registered_paths.add(full_path)

        family = doc.family
        docs_per_family[family] = docs_per_family.get(family, 0) + 1
        licenses[doc.license] = licenses.get(doc.license, 0) + 1

        if full_path.is_file():
            try:
                content = full_path.read_text(encoding="utf-8")
                char_count = len(content)
                byte_count = full_path.stat().st_size
                chars_per_family[family] = chars_per_family.get(family, 0) + char_count
                bytes_per_family[family] = bytes_per_family.get(family, 0) + byte_count

                if doc.sha256:
                    actual_sha = compute_file_sha256(full_path)
                    if actual_sha.lower() == doc.sha256.lower():
                        sha_verified += 1
                    else:
                        sha_mismatches.append(
                            f"{doc.id}: expected {doc.sha256[:12]}..., got {actual_sha[:12]}..."
                        )
                else:
                    sha_missing.append(doc.id)
            except Exception:
                pass  # Error already recorded in validation_errors

    # Scan for untracked regular files in corpus/raw
    untracked = scan_untracked_files(c_root, registered_paths)

    is_clean = len(validation_errors) == 0 and len(untracked) == 0 and len(sha_mismatches) == 0

    return AuditResult(
        is_clean=is_clean,
        num_documents=len(manifest.documents),
        documents_per_family=docs_per_family,
        characters_per_family=chars_per_family,
        bytes_per_family=bytes_per_family,
        sha256_verified=sha_verified,
        sha256_mismatches=sha_mismatches,
        sha256_missing=sha_missing,
        licenses=licenses,
        untracked_files=untracked,
        errors=validation_errors,
        warnings=warnings,
    )


def render_audit_report(result: AuditResult) -> None:
    """Display audit results in clear, formatted Rich tables and status panels."""
    console = Console()

    # Summary table
    table = Table(title="Corpus Audit Summary", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold yellow")
    table.add_column("Value", style="white")

    table.add_row("Total Registered Documents", str(result.num_documents))
    table.add_row("SHA-256 Verified", str(result.sha256_verified))
    table.add_row("SHA-256 Unspecified (to be locked)", str(len(result.sha256_missing)))
    table.add_row("SHA-256 Mismatches", str(len(result.sha256_mismatches)))
    table.add_row("Untracked Raw Files (Purity Violations)", str(len(result.untracked_files)))
    table.add_row("Total Errors", str(len(result.errors)))
    table.add_row("Total Warnings", str(len(result.warnings)))
    console.print(table)

    # Family Breakdown table
    if result.documents_per_family:
        fam_table = Table(
            title="Corpus Breakdown by Family", show_header=True, header_style="bold green"
        )
        fam_table.add_column("Scripture Family", style="bold white")
        fam_table.add_column("Documents", justify="right")
        fam_table.add_column("Raw Characters", justify="right")
        fam_table.add_column("Raw Bytes", justify="right")

        for fam in sorted(result.documents_per_family.keys()):
            fam_table.add_row(
                fam,
                f"{result.documents_per_family[fam]:,}",
                f"{result.characters_per_family.get(fam, 0):,}",
                f"{result.bytes_per_family.get(fam, 0):,}",
            )
        console.print(fam_table)

    # Purity Errors (untracked files)
    if result.untracked_files:
        purity_table = Table(
            title="PURITY ERRORS: Unregistered Files in corpus/raw",
            show_header=True,
            header_style="bold red",
        )
        purity_table.add_column("Untracked Path", style="red")
        for u in result.untracked_files:
            purity_table.add_row(u)
        console.print(purity_table)

    # Manifest Errors
    if result.errors:
        err_table = Table(
            title="MANIFEST VALIDATION ERRORS", show_header=True, header_style="bold red"
        )
        err_table.add_column("Error Description", style="red")
        for err in result.errors:
            err_table.add_row(err)
        console.print(err_table)

    # Warnings
    if result.warnings:
        warn_table = Table(title="Audit Warnings", show_header=True, header_style="bold yellow")
        warn_table.add_column("Warning Description", style="yellow")
        for w in result.warnings:
            warn_table.add_row(w)
        console.print(warn_table)

    # Status Banner
    if result.is_clean:
        console.print(
            Panel(
                "[bold green]Corpus audit passed with 0 errors and 0 purity violations.[/]",
                title="Audit Status: PASSED",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]Corpus audit failed. "
                "Fix manifest and purity errors before proceeding.[/]",
                title="Audit Status: FAILED",
                border_style="red",
            )
        )
