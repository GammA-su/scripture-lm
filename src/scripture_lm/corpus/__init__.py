"""Corpus management subsystem for Scripture-LM."""

from scripture_lm.corpus.audit import AuditResult, audit_corpus, render_audit_report
from scripture_lm.corpus.manifest import (
    CorpusManifest,
    DocumentEntry,
    EditorialCleanupConfig,
    RegexReplacement,
    compute_file_sha256,
    load_manifest,
    validate_manifest,
)
from scripture_lm.corpus.normalize import (
    CorpusLock,
    DocumentProvenance,
    conservative_normalize,
    normalize_corpus,
)
from scripture_lm.corpus.split import SplitManifest, generate_splits, load_or_generate_splits
from scripture_lm.corpus.statistics import compute_corpus_stats, render_stats_table

__all__ = [
    "AuditResult",
    "CorpusLock",
    "CorpusManifest",
    "DocumentEntry",
    "DocumentProvenance",
    "EditorialCleanupConfig",
    "RegexReplacement",
    "SplitManifest",
    "audit_corpus",
    "compute_corpus_stats",
    "compute_file_sha256",
    "conservative_normalize",
    "generate_splits",
    "load_manifest",
    "load_or_generate_splits",
    "normalize_corpus",
    "render_audit_report",
    "render_stats_table",
    "validate_manifest",
]
