"""Scripture-LM evaluation and cross-tokenizer comparison subsystem."""

from scripture_lm.evaluation.bpc import (
    BPC_EXCLUDED_IDS,
    BPCResult,
    compute_bpc,
)
from scripture_lm.evaluation.compare import (
    ComparisonReport,
    RunSummary,
    compare_runs,
    load_run_summary,
    render_comparison_table,
)
from scripture_lm.evaluation.generation_suite import (
    BENCHMARK_VERSION,
    GenerationPrompt,
    GenerationSample,
    GenerationSettings,
    GenerationSuite,
    GeneratorProtocol,
    get_canonical_generation_suite,
    load_generation_results,
    save_generation_results,
)
from scripture_lm.evaluation.memorization import (
    MatchSpan,
    MemorizationReport,
    SampleMemorizationResult,
    TrainingCorpusMatcher,
)
from scripture_lm.evaluation.perplexity import (
    PerplexityResult,
    compute_perplexity,
)
from scripture_lm.evaluation.repetition import (
    RepetitionReport,
    SampleRepetitionMetrics,
    analyze_generation_repetition,
    compute_continuation_repetition,
)

__all__ = [
    "BENCHMARK_VERSION",
    "BPC_EXCLUDED_IDS",
    "BPCResult",
    "ComparisonReport",
    "GenerationPrompt",
    "GenerationSample",
    "GenerationSettings",
    "GenerationSuite",
    "GeneratorProtocol",
    "MatchSpan",
    "MemorizationReport",
    "PerplexityResult",
    "RepetitionReport",
    "RunSummary",
    "SampleMemorizationResult",
    "SampleRepetitionMetrics",
    "TrainingCorpusMatcher",
    "analyze_generation_repetition",
    "compare_runs",
    "compute_bpc",
    "compute_continuation_repetition",
    "compute_perplexity",
    "get_canonical_generation_suite",
    "load_generation_results",
    "load_run_summary",
    "render_comparison_table",
    "save_generation_results",
]
