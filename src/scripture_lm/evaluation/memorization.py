"""Memorization analysis and exact training-corpus copy detection for Scripture-LM.

Uses an efficient 50-character rolling-hash seed-and-extend algorithm stored in NumPy arrays
to avoid Python object overhead while guaranteeing exact substring match discovery.

Matches shorter than 50 characters (e.g. ordinary short liturgical phrases like "And the Lord")
are intentionally not searched and are not treated as evidence of memorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from scripture_lm.corpus.normalize import CorpusLock
from scripture_lm.corpus.split import SplitManifest
from scripture_lm.evaluation.generation_suite import GenerationSample

# Rolling hash parameters (64-bit unsigned polynomial rolling hash)
MASK_64 = (1 << 64) - 1
HASH_BASE = 1315423911


def compute_rolling_hashes_for_string(text: str, k: int = 50) -> np.ndarray:
    """Compute rolling 64-bit polynomial hashes for all length-k windows in text."""
    n = len(text)
    if n < k:
        return np.empty(0, dtype=np.uint64)

    hashes = np.empty(n - k + 1, dtype=np.uint64)
    # Compute initial window hash
    h = 0
    base_pow = 1
    base = HASH_BASE

    for i in range(k):
        h = (h * base + ord(text[i])) & MASK_64
        if i < k - 1:
            base_pow = (base_pow * base) & MASK_64

    hashes[0] = h

    # Slide window
    for i in range(1, n - k + 1):
        prev_char = ord(text[i - 1])
        next_char = ord(text[i + k - 1])
        h = ((h - prev_char * base_pow) * base + next_char) & MASK_64
        hashes[i] = h

    return hashes


class MatchSpan(BaseModel):
    """A maximal exact substring match between a generated continuation and the training corpus."""

    model_config = ConfigDict(extra="forbid")

    length: int
    matched_text: str
    gen_start: int
    gen_end: int
    source_document_id: str
    source_family: str
    source_start: int
    source_end: int


class SampleMemorizationResult(BaseModel):
    """Memorization metrics for a single generated continuation."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    continuation_length_chars: int
    longest_exact_match_chars: int = Field(
        ...,
        description=(
            "Longest exact match length discovered (>= 50 chars). 0 denotes no match >= 50 chars."
        ),
    )
    longest_match_source_doc: str | None = None
    longest_match_source_family: str | None = None
    longest_match_text: str = ""
    maximal_matching_spans_ge_50: int = Field(
        ..., description="Count of distinct maximal matching passages >= 50 characters"
    )
    maximal_matching_spans_ge_100: int = Field(
        ..., description="Count of distinct maximal matching passages >= 100 characters"
    )
    maximal_matching_spans_ge_200: int = Field(
        ..., description="Count of distinct maximal matching passages >= 200 characters"
    )
    matched_character_coverage_ge_50: float = Field(
        ..., description="Fraction of continuation characters covered by matches >= 50 characters"
    )
    matched_character_coverage_ge_100: float = Field(
        ..., description="Fraction of continuation characters covered by matches >= 100 characters"
    )
    matched_character_coverage_ge_200: float = Field(
        ..., description="Fraction of continuation characters covered by matches >= 200 characters"
    )
    sliding_50_char_windows: int = 0
    sliding_100_char_windows: int = 0
    sliding_200_char_windows: int = 0
    matches: list[MatchSpan] = Field(default_factory=list)


class MemorizationReport(BaseModel):
    """Aggregate memorization analysis across an entire generation benchmark suite."""

    model_config = ConfigDict(extra="forbid")

    num_samples: int
    total_continuation_chars: int
    max_longest_match_chars: int
    mean_longest_match_chars: float
    total_maximal_spans_ge_50: int
    total_maximal_spans_ge_100: int
    total_maximal_spans_ge_200: int
    mean_coverage_ge_50: float
    mean_coverage_ge_100: float
    mean_coverage_ge_200: float
    samples: list[SampleMemorizationResult]


@dataclass
class DocumentRecord:
    doc_id: str
    family: str
    text: str


class TrainingCorpusMatcher:
    """Efficient rolling-hash seed-and-extend exact matcher for training scriptures."""

    def __init__(
        self,
        documents: list[DocumentRecord],
        min_seed_length: int = 50,
    ) -> None:
        """Build seed index from training documents.

        Args:
            documents: List of DocumentRecord objects containing training scripture text.
            min_seed_length: Minimum seed length to index (default 50 characters).
        """
        self.min_seed_length = min_seed_length
        self.documents = documents

        all_hashes: list[np.ndarray] = []
        all_doc_indices: list[np.ndarray] = []
        all_offsets: list[np.ndarray] = []

        for doc_idx, doc in enumerate(documents):
            d_len = len(doc.text)
            if d_len >= min_seed_length:
                h = compute_rolling_hashes_for_string(doc.text, k=min_seed_length)
                n_windows = len(h)
                all_hashes.append(h)
                all_doc_indices.append(np.full(n_windows, doc_idx, dtype=np.uint32))
                all_offsets.append(np.arange(n_windows, dtype=np.uint32))

        if all_hashes:
            concatenated_hashes = np.concatenate(all_hashes)
            concatenated_docs = np.concatenate(all_doc_indices)
            concatenated_offsets = np.concatenate(all_offsets)

            # Sort by hash for binary search
            sort_order = np.argsort(concatenated_hashes)
            self._hashes = concatenated_hashes[sort_order]
            self._doc_indices = concatenated_docs[sort_order]
            self._offsets = concatenated_offsets[sort_order]
        else:
            self._hashes = np.empty(0, dtype=np.uint64)
            self._doc_indices = np.empty(0, dtype=np.uint32)
            self._offsets = np.empty(0, dtype=np.uint32)

    @classmethod
    def from_corpus_and_split(
        cls,
        data_root: Path | str = Path("data"),
        min_seed_length: int = 50,
    ) -> TrainingCorpusMatcher:
        """Construct matcher from normalized training documents defined in split_manifest.json."""
        data_dir = Path(data_root)
        split_path = data_dir / "splits" / "split_manifest.json"
        lock_path = data_dir / "corpus_lock.json"

        if not split_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {split_path}")
        if not lock_path.is_file():
            raise FileNotFoundError(f"Corpus lock not found: {lock_path}")

        split = SplitManifest.model_validate_json(split_path.read_text(encoding="utf-8"))
        lock = CorpusLock.model_validate_json(lock_path.read_text(encoding="utf-8"))

        doc_meta_map = {d.document_id: d for d in lock.documents}
        records: list[DocumentRecord] = []

        for doc_id in split.train:
            if doc_id in doc_meta_map:
                doc_prov = doc_meta_map[doc_id]
                norm_file = data_dir / "normalized" / doc_prov.family / f"{doc_id}.txt"
                if norm_file.is_file():
                    text = norm_file.read_text(encoding="utf-8")
                    records.append(DocumentRecord(doc_id=doc_id, family=doc_prov.family, text=text))

        return cls(records, min_seed_length=min_seed_length)

    def find_matches_in_continuation(
        self,
        continuation: str,
        sample_id: str = "sample_0",
    ) -> SampleMemorizationResult:
        """Find all maximal exact matches >= min_seed_length in the generated continuation.

        Extends matches left and right while strictly honoring document boundaries.
        """
        k = self.min_seed_length
        m = len(continuation)

        if m < k or len(self._hashes) == 0:
            return SampleMemorizationResult(
                sample_id=sample_id,
                continuation_length_chars=m,
                longest_exact_match_chars=0,
                maximal_matching_spans_ge_50=0,
                maximal_matching_spans_ge_100=0,
                maximal_matching_spans_ge_200=0,
                matched_character_coverage_ge_50=0.0,
                matched_character_coverage_ge_100=0.0,
                matched_character_coverage_ge_200=0.0,
            )

        # Compute rolling hashes of generated continuation
        gen_hashes = compute_rolling_hashes_for_string(continuation, k=k)
        discovered_spans: list[MatchSpan] = []

        # Sliding window counts
        sliding_50 = 0
        sliding_100 = 0
        sliding_200 = 0

        # Track which continuation indices are already covered by verified matches
        # to avoid redundant extension of identical overlapping seeds
        for gen_idx, q_hash in enumerate(gen_hashes):
            # Binary search in training index
            left = int(np.searchsorted(self._hashes, q_hash, side="left"))
            right = int(np.searchsorted(self._hashes, q_hash, side="right"))

            if left == right:
                continue

            # Candidate hits found
            sliding_50 += 1

            for cand_idx in range(left, right):
                doc_idx = int(self._doc_indices[cand_idx])
                doc_offset = int(self._offsets[cand_idx])
                doc = self.documents[doc_idx]
                doc_text = doc.text

                # 1. Exact string verification of 50-char seed
                if continuation[gen_idx : gen_idx + k] != doc_text[doc_offset : doc_offset + k]:
                    continue

                # 2. Extend left (strictly bounded by document start 0 and continuation start 0)
                ext_left = 0
                while (
                    gen_idx - ext_left - 1 >= 0
                    and doc_offset - ext_left - 1 >= 0
                    and continuation[gen_idx - ext_left - 1] == doc_text[doc_offset - ext_left - 1]
                ):
                    ext_left += 1

                # 3. Extend right (strictly bounded by document end and continuation end)
                ext_right = 0
                doc_len = len(doc_text)
                while (
                    gen_idx + k + ext_right < m
                    and doc_offset + k + ext_right < doc_len
                    and continuation[gen_idx + k + ext_right]
                    == doc_text[doc_offset + k + ext_right]
                ):
                    ext_right += 1

                actual_gen_start = gen_idx - ext_left
                actual_gen_end = gen_idx + k + ext_right
                actual_doc_start = doc_offset - ext_left
                actual_doc_end = doc_offset + k + ext_right
                match_len = actual_gen_end - actual_gen_start

                if match_len >= 100:
                    sliding_100 += 1
                if match_len >= 200:
                    sliding_200 += 1

                span = MatchSpan(
                    length=match_len,
                    matched_text=continuation[actual_gen_start:actual_gen_end],
                    gen_start=actual_gen_start,
                    gen_end=actual_gen_end,
                    source_document_id=doc.doc_id,
                    source_family=doc.family,
                    source_start=actual_doc_start,
                    source_end=actual_doc_end,
                )
                discovered_spans.append(span)

        # Deduplicate to find distinct maximal spans in continuation
        # Sort by (gen_start, -length)
        discovered_spans.sort(key=lambda s: (s.gen_start, -s.length))
        maximal_spans: list[MatchSpan] = []

        for s in discovered_spans:
            # Check if this span is completely subsumed by an existing maximal span
            is_subsumed = False
            for m_span in maximal_spans:
                if (
                    s.gen_start >= m_span.gen_start
                    and s.gen_end <= m_span.gen_end
                    and s.source_document_id == m_span.source_document_id
                ):
                    is_subsumed = True
                    break
            if not is_subsumed:
                maximal_spans.append(s)

        # Calculate metrics
        if maximal_spans:
            longest_span = max(maximal_spans, key=lambda s: s.length)
            longest_match_chars = longest_span.length
            longest_source_doc = longest_span.source_document_id
            longest_source_family = longest_span.source_family
            longest_text = longest_span.matched_text
        else:
            longest_match_chars = 0
            longest_source_doc = None
            longest_source_family = None
            longest_text = ""

        # Maximal matching spans counts
        spans_ge_50 = sum(1 for s in maximal_spans if s.length >= 50)
        spans_ge_100 = sum(1 for s in maximal_spans if s.length >= 100)
        spans_ge_200 = sum(1 for s in maximal_spans if s.length >= 200)

        # Character coverage
        cov_50_mask = np.zeros(m, dtype=bool)
        cov_100_mask = np.zeros(m, dtype=bool)
        cov_200_mask = np.zeros(m, dtype=bool)

        for s in maximal_spans:
            if s.length >= 50:
                cov_50_mask[s.gen_start : s.gen_end] = True
            if s.length >= 100:
                cov_100_mask[s.gen_start : s.gen_end] = True
            if s.length >= 200:
                cov_200_mask[s.gen_start : s.gen_end] = True

        cov_50 = float(cov_50_mask.mean()) if m > 0 else 0.0
        cov_100 = float(cov_100_mask.mean()) if m > 0 else 0.0
        cov_200 = float(cov_200_mask.mean()) if m > 0 else 0.0

        return SampleMemorizationResult(
            sample_id=sample_id,
            continuation_length_chars=m,
            longest_exact_match_chars=longest_match_chars,
            longest_match_source_doc=longest_source_doc,
            longest_match_source_family=longest_source_family,
            longest_match_text=longest_text,
            maximal_matching_spans_ge_50=spans_ge_50,
            maximal_matching_spans_ge_100=spans_ge_100,
            maximal_matching_spans_ge_200=spans_ge_200,
            matched_character_coverage_ge_50=cov_50,
            matched_character_coverage_ge_100=cov_100,
            matched_character_coverage_ge_200=cov_200,
            sliding_50_char_windows=sliding_50,
            sliding_100_char_windows=sliding_100,
            sliding_200_char_windows=sliding_200,
            matches=maximal_spans,
        )

    def analyze_generation_samples(
        self,
        samples: list[GenerationSample],
    ) -> MemorizationReport:
        """Analyze memorization across a collection of generation samples."""
        results: list[SampleMemorizationResult] = []

        for sample in samples:
            # Strictly evaluate the continuation (excluding the prompt)
            res = self.find_matches_in_continuation(
                continuation=sample.continuation,
                sample_id=sample.sample_id,
            )
            results.append(res)

        total_chars = sum(r.continuation_length_chars for r in results)
        longest_matches = [r.longest_exact_match_chars for r in results]
        max_longest = max(longest_matches) if longest_matches else 0
        mean_longest = float(np.mean(longest_matches)) if longest_matches else 0.0

        spans_50 = sum(r.maximal_matching_spans_ge_50 for r in results)
        spans_100 = sum(r.maximal_matching_spans_ge_100 for r in results)
        spans_200 = sum(r.maximal_matching_spans_ge_200 for r in results)

        cov_50 = (
            float(np.mean([r.matched_character_coverage_ge_50 for r in results]))
            if results
            else 0.0
        )
        cov_100 = (
            float(np.mean([r.matched_character_coverage_ge_100 for r in results]))
            if results
            else 0.0
        )
        cov_200 = (
            float(np.mean([r.matched_character_coverage_ge_200 for r in results]))
            if results
            else 0.0
        )

        return MemorizationReport(
            num_samples=len(samples),
            total_continuation_chars=total_chars,
            max_longest_match_chars=max_longest,
            mean_longest_match_chars=mean_longest,
            total_maximal_spans_ge_50=spans_50,
            total_maximal_spans_ge_100=spans_100,
            total_maximal_spans_ge_200=spans_200,
            mean_coverage_ge_50=cov_50,
            mean_coverage_ge_100=cov_100,
            mean_coverage_ge_200=cov_200,
            samples=results,
        )
