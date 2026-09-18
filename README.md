# Scripture-LM

`Scripture-LM` is a research project for training small autoregressive language models **from random initialization using only Abrahamic scriptural text as linguistic training data**.

The project produces two models:

```text
Raw Abrahamic scripture
        │
        ├───────────────┐
        │               │
        ▼               ▼
  BPE tokenizer    Character tokenizer
        │               │
        ▼               ▼
 Scripture-BPE     Scripture-CHAR
        │               │
        └───────┬───────┘
                ▼
      controlled comparison
```

Both models learn only $P(x_t \mid x_1,\dots,x_{t-1})$ using causal next-token prediction under a strict closed-world regime:
- No pretrained weights, tokenizers, or embeddings
- No external text mixed into training
- No synthetic training data
- No instruction tuning or RLHF

## Project Structure

```text
scripture-lm/
├── configs/
│   ├── base.toml             # Shared model, training, and data configuration
│   ├── bpe.toml              # BPE-specific tokenizer overrides
│   ├── char.toml             # Character-specific tokenizer overrides
│   └── generation.toml       # Benchmark generation parameters
├── corpus/
│   ├── corpus_manifest.toml  # Strict audit manifest of registered scripture files
│   └── raw/
│       ├── hebrew_bible/     # Primary texts (e.g. JPS 1917)
│       ├── new_testament/    # Primary texts (e.g. KJV New Testament)
│       └── quran/            # Primary texts (e.g. Pickthall)
├── data/
│   ├── normalized/           # Cleaned scripture
│   ├── splits/               # Document-level train/val/test splits
│   └── encoded/              # Tokenized binary streams
├── artifacts/
│   └── tokenizers/           # BPE and Character vocabulary artifacts
├── runs/                     # Experiment checkpoints, metrics, and logs
└── src/
    └── scripture_lm/         # Core package
```

## Quick Start

### Installation

Manage dependencies using [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
```

### CLI Overview

```powershell
uv run scripture-lm --help
uv run scripture-lm corpus --help
uv run scripture-lm tokenizer --help
uv run scripture-lm encode --help
uv run scripture-lm train --help
uv run scripture-lm config show --help
uv run scripture-lm generate --help
uv run scripture-lm evaluate --help
uv run scripture-lm compare --help
uv run scripture-lm experiment --help
```

### Displaying Configuration

```powershell
uv run scripture-lm config show --config configs/bpe.toml
uv run scripture-lm config show --config configs/char.toml
```

### Training Configuration Check

```powershell
uv run scripture-lm train --config configs/bpe.toml --sampling-mode temperature --sampling-alpha 0.5 --dry-run
```

## Build the local scripture corpus

`tools/build_core_canon.py` converts the already-downloaded eBible JPS/KJV
read-aloud files and structured Pickthall text into 180 scripture documents.
It does not download inputs or execute any downstream pipeline.

Expected inputs are `sources/ebible/jps/`, `sources/ebible/kjv/`, and
`sources/quran/pickthall.txt`. The two ZIP files under `sources/ebible/`, when
present, are hashed for provenance; extraction is not performed by the builder.

```powershell
# Inspect all sources and validate without modifying corpus or research artifacts.
uv run python tools\build_core_canon.py --dry-run

# After reviewing the dry-run, construct the corpus explicitly.
uv run python tools\build_core_canon.py
```

The inspected eBible pattern is
`<translation>_<archive-index>_<book-code>_<chapter>_read.txt`:
Genesis is `002_GEN`, Matthew is `070_MAT`, and Revelation is `096_REV`.
Psalms uses three chapter digits; other books use two. Both archives contain
`*_000_000_000_read.txt`, `copr.htm`, and `keys.asc`, which are explicitly
excluded. Exact observed chapter counts validate all 39 JPS books and 27 KJV
New Testament books. KJV Old Testament chapters are inspected and excluded.

Cleaning removes the explicit `Chapter N.` line and all preceding title lines,
removes whitespace-delimited pilcrows, and joins scripture lines with one space.
Chapters are joined in numeric order with two newlines; documents end with one
Unix newline. Spelling, case, punctuation, and other scripture wording remain
unchanged. The supplied JPS files unexpectedly include `(chapter-verse)` markers
inside scripture lines, including outside Psalms, and five `BOOK I`–`BOOK V`
Psalms labels. Their exact source-specific removal was approved and is separately
counted in the audit; other parenthetical text and Psalm superscriptions remain.

The supplied Pickthall source uses `surah|ayah|text` with a trailing `#` comment
block. Only those comment lines and blank lines are ignored. References must be
positive integers, surahs must form unique blocks, and ayahs are sorted numerically.
The observed 114-surah / 6,236-ayah counts are pinned to detect internal gaps and
truncated final ayahs. Reference fields and comments never enter final documents.

The build writes:

- `corpus/raw/hebrew_bible/*.txt`: 39 JPS books.
- `corpus/raw/new_testament/*.txt`: 27 KJV New Testament books.
- `corpus/raw/quran/001.txt` through `114.txt`: Pickthall surahs.
- `corpus/corpus_manifest.toml`: deterministic IDs, order, licenses, and final-byte hashes.
- `research/source_provenance.json`: relative source paths, hashes, byte sizes,
  translation roles, archive hashes, and explicit KJV Old Testament exclusion.
- `research/corpus_build_report.json`: per-chapter/surah cleaning records and aggregate counts.

Existing corpus `.txt` files block construction by default. `--force` only
replaces the fixed outputs owned by a previous build, as recorded in its
provenance report. Unknown `.txt` files, unowned existing files, unrelated reports,
and symlinks/junctions are rejected. The builder never recursively deletes a
corpus directory. All input parsing and validation finish before writing; the
manifest is published after all 180 output hashes are verified. Writes are atomic
per file, not a transaction across the entire corpus, so an interrupted build
must be inspected before reuse.

Dry-run prints inventory and cleaning totals without writing reports. Build
reports contain no timestamps or absolute machine paths. Unit retained-character
counts exclude inserted separators; document totals include separators and the
final newline. All builder tests use temporary fixtures. Corpus preparation,
tokenization, encoding, training, and evaluation remain separate manual actions.

## Baseline experiments

The four canonical experiments share the same Transformer, seed, optimizer, and
early-stopping policy:

| Name | Tokenizer | Sampling | Alpha |
| --- | --- | --- | --- |
| `bpe-natural` | BPE | natural | — |
| `bpe-temperature-a05` | BPE | temperature | 0.5 |
| `char-natural` | character | natural | — |
| `char-temperature-a05` | character | temperature | 0.5 |

Every baseline explicitly specifies `max_effective_epochs = 20`,
`early_stopping_enabled = true`, and `early_stopping_patience = 8`.
An early-stopped baseline is a completed experiment. Natural sampling consumes
each chunk once per epoch without replacement; temperature sampling uses the
existing family sampler with replacement.

The versioned recipe lives in `src/scripture_lm/experiments/baseline_v1.toml`.
Editing user configs does not change what a canonical baseline name means.

```powershell
# Inspect without creating run files or starting training (no corpus required).
uv run scripture-lm experiment matrix
uv run scripture-lm experiment run --name bpe-natural --dry-run

# Run one experiment, a subset, or all four sequentially.
uv run scripture-lm experiment run --name bpe-natural
uv run scripture-lm experiment run-baseline --name bpe-natural --name char-natural
uv run scripture-lm experiment run-baseline

# Resume an interrupted run explicitly; runtime choices may change.
uv run scripture-lm experiment run --name bpe-natural --resume --device cuda --no-compile
```

Before executing, populate and register your own corpus, run `corpus prepare`,
train/build both tokenizers, and encode each tokenizer's dataset. No scripture
is downloaded automatically. Execution verifies existing data provenance.
The default directories are `runs/<experiment-name>/`; `--runs-root` selects
another output root.

Baseline commands accept `--device` and `--compile/--no-compile` but reject
scientific overrides such as `--effective-epochs`, `--seed`, and
`--sampling-alpha`. Use `run-custom` instead:

```powershell
uv run scripture-lm experiment run-custom --tokenizer bpe --sampling-mode temperature --sampling-alpha 0.25 --effective-epochs 20
uv run scripture-lm experiment run-custom --tokenizer char --sampling-mode natural --effective-epochs 5 --seed 42 --dry-run
```

Custom names use decimal labels such as `a0`, `a0p25`, `a0p5`, `a0p75`, and `a1`,
plus `-custom-<config-hash>` to distinguish scientific settings. `--config`
accepts a custom TOML for architecture and optimizer changes. Repeating the
same custom specification with `--resume` selects the same directory. Custom
experiments cannot use reserved baseline names.

Direct training remains available with the same scientific configurations:

```powershell
uv run scripture-lm train --config configs/bpe.toml --sampling-mode natural --effective-epochs 20
uv run scripture-lm train --config configs/bpe.toml --sampling-mode temperature --sampling-alpha 0.5 --effective-epochs 20
uv run scripture-lm train --config configs/char.toml --sampling-mode natural --effective-epochs 20
uv run scripture-lm train --config configs/char.toml --sampling-mode temperature --sampling-alpha 0.5 --effective-epochs 20
```

Direct training preserves its existing default directory names (for example,
`runs/bpe_natural`). Add `--run-dir runs/bpe-natural` to place a canonical run
where baseline comparison discovers it. Scientific settings are checked when
using a reserved name. For other direct experiments, choose a distinct
`--run-dir` rather than overwriting an existing run.

### Identity and execution status

Each new run writes `experiment_config.toml` and `experiment_config.sha256`
once. The hash covers canonical scientific configuration and encoded-data
provenance, including corpus, split, and tokenizer identity. Natural sampling
canonicalizes its irrelevant alpha to JSON null (omitted in TOML).
`config.toml` preserves the initial complete configuration for existing tools.
Device and compilation choices are excluded from scientific identity.

`environment.json` records runtime settings and execution history.
`run_status.json` tracks `planned`, `running`, `interrupted`, `failed`, and
`completed`, checkpoint references, and exposure progress. Baseline execution:

- Skips compatible completed runs, including runs that stopped early.
- Starts missing runs.
- Requires `--resume` and a complete latest checkpoint for interrupted runs.
- Retains failed runs and reports the failure instead of overwriting them.
- Rejects changed configurations or provenance in existing run directories.

A hard process termination can leave `running` status. Explicit `--resume`
recovers from its latest checkpoint after the process has stopped. An OS lock
prevents two experiment commands from writing the same run concurrently.
Runs interrupted before their first checkpoint are retained but cannot resume.
Use a new output root for a fresh attempt. Pre-orchestration runs without
immutable specifications/status are not automatically adopted or overwritten.
Runtime changes retain identity but need not reproduce identical floating-point
results across devices or compilation backends.

### Baseline comparison

```powershell
uv run scripture-lm experiment compare-baseline
uv run scripture-lm experiment compare-baseline --generate
uv run scripture-lm experiment compare-baseline --split test
```

Comparison evaluates each completed run's best checkpoint on validation by
default. Use `--split test` explicitly for the frozen final comparison.
Partial baselines are supported: all four names appear, while missing, failed,
and interrupted runs have statuses instead of metrics. Corpus/split and
checkpoint compatibility checks remain mandatory.

Reports are written to `runs/baseline_comparison/comparison.json`,
`comparison.csv`, and `README.md`; `--output-dir` overrides that destination.
They include family/macro/micro BPC, token loss/perplexity, exposure, and
available benchmark metrics, in canonical matrix order without a subjective
winner. Token perplexity is not comparable across tokenizers.

`--generate` runs the existing `standard_v1` prompt suite and records hashes
binding samples, repetition, and memorization reports to the checkpoint and
corpus. Generation metrics require the canonical prompts, seeds 0–9,
temperature 0.8, top-p 0.95, and 1024-character target. EOS and the suite's token
cap can end samples earlier. Ad-hoc, stale, incomplete, or incompatible
generation results appear as unavailable/incompatible. Without `--generate`,
only existing benchmark artifacts that pass these checks are included.
