"""Deterministic, stratified document-level dataset splitting with fingerprint validation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scripture_lm.corpus.manifest import CorpusManifest
from scripture_lm.corpus.normalize import CorpusLock

SPLIT_ALGORITHM = "stratified_greedy_v1"
DEFAULT_SEED = 1337
DEFAULT_TARGETS = {"train": 0.90, "validation": 0.05, "test": 0.05}


class SplitManifest(BaseModel):
    """Machine-readable document split assignment manifest."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    algorithm: str = SPLIT_ALGORITHM
    seed: int = DEFAULT_SEED
    targets: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.90, "validation": 0.05, "test": 0.05}
    )
    corpus_fingerprint: str
    normalization_fingerprint: str
    train: list[str] = Field(default_factory=list)
    validation: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)
    family_breakdown: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    actual_characters: dict[str, int] = Field(default_factory=dict)
    actual_proportions: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def generate_splits(
    manifest: CorpusManifest,
    corpus_lock: CorpusLock,
    seed: int = DEFAULT_SEED,
    targets: dict[str, float] | None = None,
) -> SplitManifest:
    """Generate deterministic, stratified document splits optimizing for character targets."""
    effective_targets = targets or DEFAULT_TARGETS
    val_ratio = effective_targets.get("validation", 0.05)
    test_ratio = effective_targets.get("test", 0.05)

    # Map doc_id -> normalized character count
    doc_chars: dict[str, int] = {
        doc.document_id: doc.normalized_characters for doc in corpus_lock.documents
    }

    # Group document entries by family
    family_docs: dict[str, list[str]] = {}
    for doc in manifest.documents:
        family_docs.setdefault(doc.family, []).append(doc.id)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    family_breakdown: dict[str, dict[str, list[str]]] = {}
    warnings: list[str] = []

    # Sort families for deterministic processing
    for family in sorted(family_docs.keys()):
        raw_docs = family_docs[family]
        # Sort documents deterministically by ID
        sorted_docs = sorted(raw_docs)

        # Seeded deterministic shuffle per family
        # Hash family name into seed for independent family shuffling
        family_seed = (seed * 31 + sum(ord(c) for c in family)) & 0xFFFFFFFF
        rng = random.Random(family_seed)
        shuffled_docs = list(sorted_docs)
        rng.shuffle(shuffled_docs)

        fam_train: list[str] = []
        fam_val: list[str] = []
        fam_test: list[str] = []

        n_docs = len(shuffled_docs)
        fam_total_chars = sum(doc_chars.get(d, 0) for d in shuffled_docs)

        if n_docs >= 3:
            # Sort documents by size to allocate base representation intelligently
            # Smallest documents give finer control for small val/test targets
            docs_by_size = sorted(shuffled_docs, key=lambda d: doc_chars.get(d, 0))

            # Minimum 1 doc per split guarantee
            # Assign smallest to val, second smallest to test, largest to train
            fam_val.append(docs_by_size[0])
            fam_test.append(docs_by_size[1])
            fam_train.append(docs_by_size[-1])
            remaining = docs_by_size[2:-1]

            cur_val_chars = doc_chars.get(fam_val[0], 0)
            cur_test_chars = doc_chars.get(fam_test[0], 0)
            cur_train_chars = doc_chars.get(fam_train[0], 0)

            target_val_chars = fam_total_chars * val_ratio
            target_test_chars = fam_total_chars * test_ratio

            # Greedily allocate remaining documents
            # Shuffle remaining to prevent size bias
            rng.shuffle(remaining)
            for d in remaining:
                c = doc_chars.get(d, 0)
                # Compute current deficits
                def_val = target_val_chars - cur_val_chars
                def_test = target_test_chars - cur_test_chars

                # Pick split with largest positive deficit
                if def_val > 0 and (def_val >= def_test or def_test <= 0):
                    fam_val.append(d)
                    cur_val_chars += c
                elif def_test > 0:
                    fam_test.append(d)
                    cur_test_chars += c
                else:
                    fam_train.append(d)
                    cur_train_chars += c

        elif n_docs == 2:
            warnings.append(
                f"Family '{family}' contains only 2 documents. "
                "Representation in all 3 splits is impossible without subdividing documents."
            )
            # Allocate 1 to train, 1 to val
            fam_train.append(shuffled_docs[0])
            fam_val.append(shuffled_docs[1])
        elif n_docs == 1:
            warnings.append(
                f"Family '{family}' contains only 1 document. Allocated entirely to train split."
            )
            fam_train.append(shuffled_docs[0])

        fam_train.sort()
        fam_val.sort()
        fam_test.sort()

        train_ids.extend(fam_train)
        val_ids.extend(fam_val)
        test_ids.extend(fam_test)

        family_breakdown[family] = {
            "train": fam_train,
            "validation": fam_val,
            "test": fam_test,
        }

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()

    total_chars = sum(doc_chars.values()) if doc_chars else 1
    actual_chars = {
        "train": sum(doc_chars.get(d, 0) for d in train_ids),
        "validation": sum(doc_chars.get(d, 0) for d in val_ids),
        "test": sum(doc_chars.get(d, 0) for d in test_ids),
    }
    actual_props = {k: (v / total_chars) for k, v in actual_chars.items()}

    return SplitManifest(
        version="1.0",
        algorithm=SPLIT_ALGORITHM,
        seed=seed,
        targets=effective_targets,
        corpus_fingerprint=corpus_lock.corpus_fingerprint,
        normalization_fingerprint=corpus_lock.normalization_fingerprint,
        train=train_ids,
        validation=val_ids,
        test=test_ids,
        family_breakdown=family_breakdown,
        actual_characters=actual_chars,
        actual_proportions=actual_props,
        warnings=warnings,
    )


def load_or_generate_splits(
    manifest: CorpusManifest,
    corpus_lock: CorpusLock,
    split_path: Path = Path("data/splits/split_manifest.json"),
    seed: int = DEFAULT_SEED,
    targets: dict[str, float] | None = None,
    force_split: bool = False,
) -> tuple[SplitManifest, bool]:
    """Load existing split if fingerprints and parameters match; otherwise generate new.

    Returns:
        tuple[split_manifest, was_reused: bool]

    Raises:
        ValueError: If split parameters or fingerprints differ and force_split is False.
    """
    effective_targets = targets or DEFAULT_TARGETS

    if split_path.is_file() and not force_split:
        try:
            with open(split_path, "r", encoding="utf-8") as f:
                existing_dict: dict[str, Any] = json.load(f)
            existing = SplitManifest.model_validate(existing_dict)

            # Check matching criteria
            discrepancies: list[str] = []
            if existing.corpus_fingerprint != corpus_lock.corpus_fingerprint:
                discrepancies.append(
                    f"Corpus fingerprint mismatch ("
                    f"persisted: {existing.corpus_fingerprint[:10]}..., "
                    f"current: {corpus_lock.corpus_fingerprint[:10]}...)"
                )
            if existing.normalization_fingerprint != corpus_lock.normalization_fingerprint:
                discrepancies.append(
                    f"Normalization fingerprint mismatch ("
                    f"persisted: {existing.normalization_fingerprint[:10]}..., "
                    f"current: {corpus_lock.normalization_fingerprint[:10]}...)"
                )
            if existing.seed != seed:
                discrepancies.append(
                    f"Seed mismatch (persisted: {existing.seed}, requested: {seed})"
                )
            if existing.algorithm != SPLIT_ALGORITHM:
                discrepancies.append(
                    f"Algorithm mismatch (persisted: {existing.algorithm}, "
                    f"current: {SPLIT_ALGORITHM})"
                )
            if existing.targets != effective_targets:
                discrepancies.append(
                    f"Target ratios mismatch ("
                    f"persisted: {existing.targets}, requested: {effective_targets})"
                )

            if not discrepancies:
                return existing, True
            else:
                disc_str = "\n  - ".join(discrepancies)
                raise ValueError(
                    f"Existing split manifest '{split_path}' cannot be reused "
                    f"due to discrepancies:\n"
                    f"  - {disc_str}\n\n"
                    "Use --force-split to regenerate split assignments."
                )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(
                f"Existing split manifest '{split_path}' is corrupted: {e}. "
                "Use --force-split to regenerate."
            )

    # Generate new split
    new_split = generate_splits(manifest, corpus_lock, seed=seed, targets=effective_targets)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(new_split.model_dump(), f, indent=2)

    return new_split, False
