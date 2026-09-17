"""Standard generation benchmark definitions, data models, and protocols.

Note:
    The canonical autoregressive generation engine, KV cache, and CLI generate command
    are implemented in PROMPT 08.
    This module specifies the versioned generation benchmark suite ('standard_v1'),
    canonical evaluation prompts, data structures, and the GeneratorProtocol interface
    that evaluation and memorization analysis rely upon.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

BENCHMARK_VERSION = "standard_v1"


class GenerationPrompt(BaseModel):
    """A prompt specification in the evaluation suite."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    family: str
    prompt_text: str


class GenerationSettings(BaseModel):
    """Hyperparameters for sampling text from an autoregressive language model."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    max_new_tokens: int = 512
    max_new_characters: int | None = 1024
    seed: int = 42


class GenerationResult(BaseModel):
    """Structured result of an autoregressive generation pass."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    continuation: str
    full_text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    finish_reason: str = Field(
        ...,
        description="Termination reason: 'eos', 'character_limit', 'length', or 'context_limit'",
    )
    characters_generated: int


class GenerationSample(BaseModel):
    """A generated text sample split into prompt and model continuation."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    prompt: str
    continuation: str
    full_text: str
    settings: GenerationSettings
    finish_reason: str = "length"
    family: str | None = None
    suite_id: str = BENCHMARK_VERSION


class GenerationSuite(BaseModel):
    """A standardized suite of prompts and sampling settings for model comparison."""

    model_config = ConfigDict(extra="forbid")

    suite_id: str = BENCHMARK_VERSION
    description: str
    prompts: list[GenerationPrompt]
    canonical_settings: GenerationSettings
    seeds: list[int] = Field(default_factory=lambda: list(range(10)))


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Protocol for language model text generators implemented in PROMPT 08."""

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        *,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Generate text continuation given a prompt and sampling settings."""
        ...


CANONICAL_PROMPTS: list[GenerationPrompt] = [
    GenerationPrompt(
        prompt_id="unprompted",
        family="general",
        prompt_text="",
    ),
    GenerationPrompt(
        prompt_id="hebrew_bible_creation",
        family="hebrew_bible",
        prompt_text="In the beginning God created the heaven and the earth.",
    ),
    GenerationPrompt(
        prompt_id="hebrew_bible_law",
        family="hebrew_bible",
        prompt_text="And the LORD said unto Moses,",
    ),
    GenerationPrompt(
        prompt_id="new_testament_gospel",
        family="new_testament",
        prompt_text="The book of the generation of Jesus Christ, the son of David,",
    ),
    GenerationPrompt(
        prompt_id="new_testament_john",
        family="new_testament",
        prompt_text="In the beginning was the Word, and the Word was with God,",
    ),
    GenerationPrompt(
        prompt_id="quran_bismillah",
        family="quran",
        prompt_text="In the name of God, the Merciful, the Compassionate.",
    ),
    GenerationPrompt(
        prompt_id="quran_fatihah",
        family="quran",
        prompt_text="Praise belongs to God, the Lord of all Being,",
    ),
]


def get_canonical_generation_suite() -> GenerationSuite:
    """Return the fixed 'standard_v1' canonical generation benchmark suite."""
    return GenerationSuite(
        suite_id=BENCHMARK_VERSION,
        description=(
            "Canonical Scripture-LM generation benchmark suite "
            "(seeds 0..9, T=0.8, p=0.95, target=1024 chars)"
        ),
        prompts=CANONICAL_PROMPTS,
        canonical_settings=GenerationSettings(
            temperature=0.8,
            top_p=0.95,
            top_k=None,
            max_new_tokens=512,
            max_new_characters=1024,
            seed=0,
        ),
        seeds=list(range(10)),
    )


def sample_from_result(
    sample_id: str,
    result: GenerationResult,
    settings: GenerationSettings,
    family: str | None = None,
    suite_id: str = BENCHMARK_VERSION,
) -> GenerationSample:
    """Construct a GenerationSample from a GenerationResult."""
    return GenerationSample(
        sample_id=sample_id,
        prompt=result.prompt,
        continuation=result.continuation,
        full_text=result.full_text,
        settings=settings,
        finish_reason=result.finish_reason,
        family=family,
        suite_id=suite_id,
    )


def save_generation_results(
    path: Path | str,
    samples: list[GenerationSample],
    suite_id: str = BENCHMARK_VERSION,
) -> None:
    """Save generation samples to a machine-readable JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite_id": suite_id,
        "num_samples": len(samples),
        "samples": [s.model_dump() for s in samples],
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_generation_results(path: Path | str) -> list[GenerationSample]:
    """Load generation samples from a JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Generation results file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    samples_data = data.get("samples", [])
    return [GenerationSample.model_validate(s) for s in samples_data]
