"""Scientific identity, execution lifecycle, and baseline reporting contracts."""

from __future__ import annotations

import csv
import json
import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from scripture_lm.cli import app
from scripture_lm.config import load_config
from scripture_lm.corpus.manifest import compute_file_sha256
from scripture_lm.data.chunk_index import EncodingProvenance
from scripture_lm.evaluation.generation_suite import (
    GenerationSample,
    get_canonical_generation_suite,
    save_generation_results,
)
from scripture_lm.experiments import reporting, runner
from scripture_lm.experiments.matrix import (
    alpha_label,
    baseline_matrix,
    config_hash,
    custom_experiment,
    get_baseline,
    scientific_config,
)
from scripture_lm.experiments.storage import (
    atomic_json,
    prepare_run,
    read_spec,
    read_status,
    serialize_toml_dict,
    update_status,
)

BASELINES = [
    "bpe-natural",
    "bpe-temperature-a05",
    "char-natural",
    "char-temperature-a05",
]
cli = CliRunner()


def provenance(tmp_path: Path, tokenizer: str = "bpe") -> EncodingProvenance:
    split = tmp_path / "source_split.json"
    split.write_text("{}", encoding="utf-8")
    return EncodingProvenance(
        tokenizer_type=tokenizer,
        context_length=512 if tokenizer == "bpe" else 2048,
        chunk_length=513 if tokenizer == "bpe" else 2049,
        corpus_fingerprint="corpus1",
        normalization_fingerprint="norm1",
        split_manifest_hash=compute_file_sha256(split),
        tokenizer_artifact_sha256=f"{tokenizer}123",
    )


def create_run(tmp_path: Path, name: str, status: str = "completed") -> Path:
    experiment = get_baseline(name)
    config = experiment.resolve()
    run = tmp_path / name
    prepare_run(run, config, provenance(tmp_path, config.tokenizer.type).model_dump(), resume=False)
    update_status(run, status)
    (run / "split_manifest.json").write_text("{}", encoding="utf-8")
    atomic_json(
        run / "corpus_lock.json",
        {
            "corpus_fingerprint": "corpus1",
            "normalization_fingerprint": "norm1",
        },
    )
    for checkpoint in ("best", "latest"):
        directory = run / "checkpoints" / checkpoint
        directory.mkdir(parents=True)
        (directory / "model.safetensors").write_bytes(b"fixture weights")
        (directory / "training_state.pt").write_bytes(b"fixture state")
        atomic_json(directory / "metadata.json", {"cumulative_raw_chars": 100})
    return run


def fake_evaluation(run: Path, split: str = "validation", **kwargs: Any) -> None:
    prov = read_spec(run)["provenance"]
    atomic_json(
        run / "evaluation" / f"{split}_metrics.json",
        {
            **{
                k: prov[k]
                for k in (
                    "corpus_fingerprint",
                    "normalization_fingerprint",
                    "split_manifest_hash",
                    "tokenizer_artifact_sha256",
                )
            },
            "split": split,
            "checkpoint_model_sha256": compute_file_sha256(
                run / "checkpoints" / "best" / "model.safetensors"
            ),
            "macro_bpc": 1.82,
            "micro_bpc": 1.83,
            "family_bpc": {
                "hebrew_bible": 1.80,
                "new_testament": 1.82,
                "quran": 1.84,
            },
            "token_cross_entropy": 2.0,
            "token_perplexity": 7.389,
            "evaluated_characters": 1000,
            "evaluated_chunks": 10,
        },
    )


def test_matrix_has_fixed_definitions_and_independent_configs() -> None:
    experiments = baseline_matrix()
    assert [e.name for e in experiments] == BASELINES
    for experiment in experiments:
        config = experiment.resolve()
        assert config.training.max_effective_epochs == 20
        assert config.training.early_stopping_enabled is True
        assert config.training.early_stopping_patience == 8
        assert config.model.layers == 6
        config.training.max_effective_epochs = 1
        assert experiment.resolve().training.max_effective_epochs == 20
    with pytest.raises(FrozenInstanceError):
        experiments[0].name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("name", BASELINES)
def test_direct_training_config_is_scientifically_equivalent(name: str) -> None:
    baseline = get_baseline(name).resolve()
    config = load_config(
        Path("configs/char.toml" if name.startswith("char") else "configs/bpe.toml"),
        cli_overrides={
            "data.sampling_mode": baseline.data.sampling_mode,
            "data.sampling_alpha": 0.5,
            "training.max_effective_epochs": 20,
        },
    )
    assert scientific_config(config) == scientific_config(baseline)


def test_runtime_device_override_does_not_change_experiment_identity() -> None:
    experiment = get_baseline("bpe-natural")
    assert config_hash(scientific_config(experiment.resolve())) == config_hash(
        scientific_config(experiment.resolve(device="cpu", compile_model=False))
    )


def test_natural_alpha_is_canonicalized_but_temperature_alpha_is_scientific() -> None:
    config = get_baseline("bpe-natural").resolve()
    config.data.sampling_alpha = 0.2
    original = config_hash(scientific_config(config))
    config.data.sampling_alpha = 0.9
    assert scientific_config(config)["data"]["sampling_alpha"] is None
    assert config_hash(scientific_config(config)) == original
    config.data.sampling_mode = "temperature"
    assert config_hash(scientific_config(config)) != original


@pytest.mark.parametrize(
    ("value", "label"),
    [
        ("0", "a0"),
        ("0.25", "a0p25"),
        ("0.50", "a0p5"),
        ("0.75", "a0p75"),
        ("1.00", "a1"),
        ("2.5e-1", "a0p25"),
    ],
)
def test_custom_alpha_names(value: str, label: str) -> None:
    assert alpha_label(value) == label
    experiment = custom_experiment("bpe", "temperature", value)
    assert experiment.name.startswith(f"bpe-temperature-{label}-custom-")
    assert experiment.name not in BASELINES


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1", "bad"])
def test_invalid_alpha_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        custom_experiment("bpe", "temperature", value)


def test_custom_identity_includes_epoch_and_seed_and_normalizes_decimal() -> None:
    first = custom_experiment("bpe", "temperature", "0.250")
    assert first.name == custom_experiment("bpe", "temperature", "0.25").name
    assert first.name != custom_experiment("bpe", "temperature", "0.25", 5).name
    assert first.name != custom_experiment("bpe", "temperature", "0.25", seed=42).name
    assert (
        custom_experiment("bpe", "natural", "0.2").name
        == custom_experiment("bpe", "natural", "0.9").name
    )


def test_baseline_rejects_effective_epoch_override() -> None:
    result = cli.invoke(
        app, ["experiment", "run", "--name", "bpe-natural", "--effective-epochs", "5", "--dry-run"]
    )
    assert result.exit_code != 0


def test_baseline_rejects_alpha_override() -> None:
    result = cli.invoke(
        app,
        [
            "experiment",
            "run",
            "--name",
            "bpe-temperature-a05",
            "--sampling-alpha",
            "0.25",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--seed", "42"),
        ("--sampling-mode", "temperature"),
        ("--tokenizer", "char"),
        ("--config", "configs/char.toml"),
    ],
)
def test_baseline_rejects_other_scientific_overrides(flag: str, value: str) -> None:
    result = cli.invoke(app, ["experiment", "run", "--name", "bpe-natural", flag, value])
    assert result.exit_code != 0


def test_matrix_and_dry_runs_have_no_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    constructor = Mock(side_effect=AssertionError("Training must not start"))
    monkeypatch.setattr(runner, "Trainer", constructor)
    for command in (
        ["matrix"],
        ["run", "--name", "bpe-natural", "--device", "cpu", "--dry-run"],
        ["run-baseline", "--dry-run"],
        [
            "run-custom",
            "--tokenizer",
            "char",
            "--sampling-mode",
            "temperature",
            "--sampling-alpha",
            "0.25",
            "--effective-epochs",
            "5",
            "--dry-run",
        ],
    ):
        result = cli.invoke(app, ["experiment", *command])
        assert result.exit_code == 0, result.output
    assert list(tmp_path.iterdir()) == []
    constructor.assert_not_called()


def test_immutable_snapshot_allows_runtime_resume_and_rejects_changes(tmp_path: Path) -> None:
    config = get_baseline("bpe-natural").resolve()
    prov = provenance(tmp_path).model_dump()
    run = tmp_path / "run"
    prepare_run(run, config, prov, resume=False)
    initial = {p.name: p.read_bytes() for p in run.iterdir()}
    config.training.device = "cpu"
    config.training.compile = False
    config.data.sampling_alpha = 0.2
    prepare_run(run, config, prov, resume=True)
    assert {p.name: p.read_bytes() for p in run.iterdir()} == initial
    assert read_spec(run)["configuration"]["training"]["early_stopping_enabled"] is True
    with pytest.raises(ValueError, match="explicit --resume"):
        prepare_run(run, config, prov, resume=False)
    config.training.seed = 42
    with pytest.raises(ValueError, match="Incompatible"):
        prepare_run(run, config, prov, resume=True)


def test_direct_train_cannot_use_reserved_name_with_custom_settings(tmp_path: Path) -> None:
    config = get_baseline("bpe-natural").resolve()
    config.training.max_effective_epochs = 5
    with pytest.raises(ValueError, match="Reserved baseline"):
        prepare_run(tmp_path / "bpe-natural", config, {}, resume=False)
    assert not (tmp_path / "bpe-natural").exists()


def test_immutable_hash_detects_tampering(tmp_path: Path) -> None:
    run = create_run(tmp_path, "bpe-natural")
    path = run / "experiment_config.toml"
    path.write_text(path.read_text("utf-8").replace("1337", "1338"), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_spec(run)


def test_config_serialization_roundtrips_escaped_strings() -> None:
    data = {"nested": {"value": 'quotes " and \\ slash\nnewline', "bools": [True, False]}}
    assert tomllib.loads(serialize_toml_dict(data)) == data


def test_process_lock_prevents_concurrent_execution(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with runner.execution_lock(path):
        with pytest.raises(ValueError, match="already running"):
            with runner.execution_lock(path):
                pytest.fail("Second execution acquired lock")
    with runner.execution_lock(path):
        pass


def test_completed_run_skips_and_incomplete_requires_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = create_run(tmp_path, "bpe-natural")
    prov = provenance(tmp_path)
    monkeypatch.setattr(runner, "verify_encoding_provenance", lambda cfg: (None, None, prov, 100))
    constructor = Mock()
    monkeypatch.setattr(runner, "Trainer", constructor)
    result = runner.run_experiment(get_baseline("bpe-natural"), runs_root=tmp_path)
    assert result["skipped"] is True
    constructor.assert_not_called()
    for status in ("running", "interrupted", "planned"):
        update_status(run, status)
        with pytest.raises(ValueError, match="explicit --resume"):
            runner.run_experiment(get_baseline("bpe-natural"), runs_root=tmp_path)
    update_status(run, "interrupted")
    constructor.return_value.train.return_value = {"effective_epoch": 20}
    runner.run_experiment(get_baseline("bpe-natural"), runs_root=tmp_path, resume=True)
    assert constructor.call_args.kwargs["resume_checkpoint_dir"] == run / "checkpoints" / "latest"
    update_status(run, "failed", error="test failure")
    with pytest.raises(ValueError, match="Failed run retained"):
        runner.run_experiment(get_baseline("bpe-natural"), runs_root=tmp_path, resume=True)


def test_resume_missing_checkpoint_fails_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = create_run(tmp_path, "bpe-natural", "interrupted")
    (run / "checkpoints" / "latest" / "training_state.pt").unlink()
    prov = provenance(tmp_path)
    monkeypatch.setattr(runner, "verify_encoding_provenance", lambda cfg: (None, None, prov, 100))
    with pytest.raises(ValueError, match="No complete resume checkpoint"):
        runner.run_experiment(get_baseline("bpe-natural"), runs_root=tmp_path, resume=True)
    assert read_status(run)["status"] == "interrupted"


def test_sequential_matrix_stops_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def execute(experiment: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(experiment.name)
        if len(seen) == 2:
            raise RuntimeError("failure")
        return {"status": "completed"}

    monkeypatch.setattr(runner, "run_experiment", execute)
    with pytest.raises(RuntimeError, match="failure"):
        runner.run_baseline()
    assert seen == BASELINES[:2]


@pytest.mark.parametrize(
    "error,status", [(RuntimeError("failed"), "failed"), (KeyboardInterrupt(), "interrupted")]
)
def test_initialization_failure_has_explicit_status(
    error: BaseException,
    status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prov = provenance(tmp_path)
    monkeypatch.setattr(runner, "verify_encoding_provenance", lambda cfg: (None, None, prov, 100))

    def fail(**kwargs: Any) -> None:
        prepare_run(kwargs["run_dir"], kwargs["config"], prov.model_dump(), resume=False)
        update_status(kwargs["run_dir"], "planned")
        raise error

    monkeypatch.setattr(runner, "Trainer", fail)
    with pytest.raises(type(error)):
        runner.run_experiment(get_baseline("bpe-natural"), runs_root=tmp_path)
    assert read_status(tmp_path / "bpe-natural")["status"] == status


def test_partial_comparison_preserves_statuses_and_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_run(tmp_path, "bpe-natural")
    create_run(tmp_path, "bpe-temperature-a05", "failed")
    create_run(tmp_path, "char-natural", "interrupted")
    evaluator = Mock(side_effect=fake_evaluation)
    monkeypatch.setattr(reporting, "evaluate_run", evaluator)
    payload = reporting.compare_baseline(runs_root=tmp_path)
    assert payload["split"] == "validation"
    rows = payload["runs"]
    assert [r["status"] for r in rows] == ["completed", "failed", "interrupted", "missing"]
    assert rows[0]["macro_bpc"] == 1.82
    assert all(r["macro_bpc"] is None for r in rows[1:])
    assert rows[0]["longest_match_chars"] is None
    assert evaluator.call_count == 1
    output = tmp_path / "baseline_comparison"
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[1]["macro_bpc"] == ""
    assert csv_rows[0]["hebrew_bible_bpc"] == "1.8"
    assert "winner" not in (output / "README.md").read_text("utf-8").lower()
    reporting.compare_baseline(runs_root=tmp_path, split="test")
    assert evaluator.call_args.kwargs["split"] == "test"


def test_all_missing_comparison_produces_status_report(tmp_path: Path) -> None:
    payload = reporting.compare_baseline(runs_root=tmp_path)
    assert [row["status"] for row in payload["runs"]] == ["missing"] * 4


def test_comparison_reuses_compatibility_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in BASELINES[:2]:
        create_run(tmp_path, name)

    def evaluate(run: Path, **kwargs: Any) -> None:
        fake_evaluation(run, **kwargs)
        if run.name.endswith("a05"):
            path = run / "evaluation" / "validation_metrics.json"
            data = json.loads(path.read_text("utf-8"))
            data["evaluated_characters"] = 999
            atomic_json(path, data)

    monkeypatch.setattr(reporting, "evaluate_run", evaluate)
    with pytest.raises(ValueError, match="Evaluated character count differs"):
        reporting.compare_baseline(runs_root=tmp_path)
    assert not (tmp_path / "baseline_comparison").exists()


def build_benchmark(run: Path) -> dict[str, Any]:
    fake_evaluation(run)
    evaluation: dict[str, Any] = json.loads(
        (run / "evaluation" / "validation_metrics.json").read_text("utf-8")
    )
    suite = get_canonical_generation_suite()
    samples = [
        GenerationSample(
            sample_id=f"{prompt.prompt_id}_s{seed}",
            prompt=prompt.prompt_text,
            continuation="generated fixture",
            full_text=prompt.prompt_text + "generated fixture",
            settings=suite.canonical_settings.model_copy(update={"seed": seed}),
            family=prompt.family,
        )
        for prompt in suite.prompts
        for seed in suite.seeds
    ]
    root = run / "generations"
    save_generation_results(root / "samples.json", samples)
    atomic_json(root / "memorization_report.json", {"max_longest_match_chars": 127})
    atomic_json(root / "repetition_report.json", {"mean_distinct_4": 0.8})
    atomic_json(
        root / "benchmark_provenance.json",
        {
            "suite": suite.model_dump(mode="json"),
            **{
                key: evaluation[key]
                for key in (
                    "checkpoint_model_sha256",
                    "tokenizer_artifact_sha256",
                    "corpus_fingerprint",
                    "split_manifest_hash",
                )
            },
            "files": {
                name: compute_file_sha256(root / name)
                for name in (
                    "samples.json",
                    "memorization_report.json",
                    "repetition_report.json",
                )
            },
        },
    )
    return evaluation


def test_only_bound_canonical_generation_is_comparable(tmp_path: Path) -> None:
    run = create_run(tmp_path, "bpe-natural")
    assert reporting.benchmark_metrics(run, {})["generation_status"] == "unavailable"
    evaluation = build_benchmark(run)
    metrics = reporting.benchmark_metrics(run, evaluation)
    assert metrics["generation_status"] == "compatible"
    assert metrics["longest_match_chars"] == 127
    evaluation["checkpoint_model_sha256"] = "other checkpoint"
    assert reporting.benchmark_metrics(run, evaluation)["generation_status"] == "incompatible"


@pytest.mark.parametrize("change", ["prompt", "seed", "temperature", "top_p", "length", "missing"])
def test_noncanonical_benchmark_samples_are_rejected(tmp_path: Path, change: str) -> None:
    run = create_run(tmp_path, "bpe-natural")
    evaluation = build_benchmark(run)
    root = run / "generations"
    path = root / "samples.json"
    data = json.loads(path.read_text("utf-8"))
    sample = data["samples"][0]
    if change == "missing":
        data["samples"].pop()
    elif change == "prompt":
        sample["prompt"] = "other prompt"
    else:
        key = "max_new_characters" if change == "length" else change
        sample["settings"][key] = {"seed": 100, "temperature": 1.0, "top_p": 0.8, "length": 100}[
            change
        ]
    atomic_json(path, data)
    meta_path = root / "benchmark_provenance.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta["files"]["samples.json"] = compute_file_sha256(path)
    atomic_json(meta_path, meta)
    assert reporting.benchmark_metrics(run, evaluation)["generation_status"] == "incompatible"


def test_training_wrapper_marks_early_stopping_as_completed(tmp_path: Path) -> None:
    from scripture_lm.training.trainer import Trainer

    run = create_run(tmp_path, "bpe-natural", "planned")
    trainer = object.__new__(Trainer)
    trainer.config = get_baseline("bpe-natural").resolve()
    trainer.run_dir = run
    trainer.N = 100
    trainer.cumulative_raw_chars = 742
    trainer.best_checkpoint_dir = run / "checkpoints" / "best"
    trainer.latest_checkpoint_dir = run / "checkpoints" / "latest"
    trainer.early_stopping = SimpleNamespace(should_stop=True)  # type: ignore[assignment]
    trainer.metrics_logger = Mock()
    trainer._train = Mock(return_value={"effective_epoch": 7.42})  # type: ignore[method-assign]
    trainer.train()
    status = read_status(run)
    assert status["status"] == "completed"
    assert status["completion_reason"] == "early_stopping"
    assert status["effective_epochs_completed"] == 7.42
