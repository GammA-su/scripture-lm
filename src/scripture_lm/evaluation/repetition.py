"""Repetition and diversity analysis for generated text continuations.

Measures:
    - repeated n-gram rate (fraction of duplicate n-grams)
    - unique n-gram ratio (distinct-1, distinct-2, distinct-3, distinct-4)
    - degenerate cycling loop detection (repeating periodic phrases)
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from scripture_lm.evaluation.generation_suite import GenerationSample


def tokenize_words(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_distinct_ngrams(tokens: list[str], n: int) -> tuple[float, int, int]:
    """Compute distinct-n ratio (unique n-grams / total n-grams).

    Returns:
        (ratio, unique_count, total_count)
    """
    if len(tokens) < n:
        return 1.0, len(tokens), len(tokens)

    total_ngrams = len(tokens) - n + 1
    unique_ngrams = len(set(tuple(tokens[i : i + n]) for i in range(total_ngrams)))
    ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 1.0
    return ratio, unique_ngrams, total_ngrams


def detect_degenerate_cycle(
    tokens: list[str], max_period: int = 16, min_repetitions: int = 3
) -> tuple[bool, int | None, list[str] | None]:
    """Detect if a periodic sequence of length p <= max_period repeats consecutively.

    Returns:
        (has_cycle, cycle_length, cycle_tokens)
    """
    n = len(tokens)
    if n < min_repetitions:
        return False, None, None

    max_p = min(max_period, n // min_repetitions)
    # Check periods from 1 to max_p
    for p in range(1, max_p + 1):
        min_len = p * min_repetitions
        # Slide search window
        for start in range(n - min_len + 1):
            pattern = tokens[start : start + p]
            # Check if repeats min_repetitions times consecutively
            repeats = True
            for rep in range(1, min_repetitions):
                offset = start + rep * p
                if tokens[offset : offset + p] != pattern:
                    repeats = False
                    break
            if repeats:
                return True, p, pattern

    return False, None, None


class SampleRepetitionMetrics(BaseModel):
    """Repetition and diversity metrics for a single continuation."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    num_words: int
    num_chars: int
    distinct_1: float = Field(..., description="Unique unigrams / total unigrams")
    distinct_2: float = Field(..., description="Unique bigrams / total bigrams")
    distinct_3: float = Field(..., description="Unique trigrams / total trigrams")
    distinct_4: float = Field(..., description="Unique 4-grams / total 4-grams")
    repeated_ngram_rate_4: float = Field(
        ..., description="Fraction of 4-grams that are duplicates (1.0 - distinct_4)"
    )
    has_degenerate_cycle: bool = False
    degenerate_cycle_period: int | None = None
    degenerate_cycle_snippet: str | None = None


class RepetitionReport(BaseModel):
    """Aggregated repetition analysis across an entire generation suite."""

    model_config = ConfigDict(extra="forbid")

    num_samples: int
    mean_distinct_1: float
    mean_distinct_2: float
    mean_distinct_3: float
    mean_distinct_4: float
    mean_repeated_ngram_rate_4: float
    samples_with_degenerate_cycle: int
    samples: list[SampleRepetitionMetrics]


def compute_continuation_repetition(
    continuation: str,
    sample_id: str = "sample_0",
) -> SampleRepetitionMetrics:
    """Compute repetition and diversity metrics for a generated text continuation."""
    words = tokenize_words(continuation)
    n_words = len(words)
    n_chars = len(continuation)

    if n_words == 0:
        return SampleRepetitionMetrics(
            sample_id=sample_id,
            num_words=0,
            num_chars=n_chars,
            distinct_1=1.0,
            distinct_2=1.0,
            distinct_3=1.0,
            distinct_4=1.0,
            repeated_ngram_rate_4=0.0,
            has_degenerate_cycle=False,
        )

    d1, _, _ = compute_distinct_ngrams(words, 1)
    d2, _, _ = compute_distinct_ngrams(words, 2)
    d3, _, _ = compute_distinct_ngrams(words, 3)
    d4, _, _ = compute_distinct_ngrams(words, 4)
    rep_4 = 1.0 - d4

    has_cycle, period, cycle_tokens = detect_degenerate_cycle(words)
    snippet = " ".join(cycle_tokens) if cycle_tokens else None

    return SampleRepetitionMetrics(
        sample_id=sample_id,
        num_words=n_words,
        num_chars=n_chars,
        distinct_1=d1,
        distinct_2=d2,
        distinct_3=d3,
        distinct_4=d4,
        repeated_ngram_rate_4=rep_4,
        has_degenerate_cycle=has_cycle,
        degenerate_cycle_period=period,
        degenerate_cycle_snippet=snippet,
    )


def analyze_generation_repetition(
    samples: list[GenerationSample],
) -> RepetitionReport:
    """Analyze repetition across all generated samples in a benchmark suite."""
    metrics: list[SampleRepetitionMetrics] = []

    for sample in samples:
        m = compute_continuation_repetition(sample.continuation, sample_id=sample.sample_id)
        metrics.append(m)

    if metrics:
        mean_d1 = float(np.mean([m.distinct_1 for m in metrics]))
        mean_d2 = float(np.mean([m.distinct_2 for m in metrics]))
        mean_d3 = float(np.mean([m.distinct_3 for m in metrics]))
        mean_d4 = float(np.mean([m.distinct_4 for m in metrics]))
        mean_rep4 = float(np.mean([m.repeated_ngram_rate_4 for m in metrics]))
        deg_count = sum(1 for m in metrics if m.has_degenerate_cycle)
    else:
        mean_d1, mean_d2, mean_d3, mean_d4, mean_rep4 = 1.0, 1.0, 1.0, 1.0, 0.0
        deg_count = 0

    return RepetitionReport(
        num_samples=len(samples),
        mean_distinct_1=mean_d1,
        mean_distinct_2=mean_d2,
        mean_distinct_3=mean_d3,
        mean_distinct_4=mean_d4,
        mean_repeated_ngram_rate_4=mean_rep4,
        samples_with_degenerate_cycle=deg_count,
        samples=metrics,
    )
