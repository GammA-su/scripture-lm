"""Unit tests for Scripture-LM CLI hierarchy, commands, and options."""

import pytest
from typer.testing import CliRunner

from scripture_lm.cli import app

runner = CliRunner()


def test_root_help() -> None:
    """Verify root scripture-lm --help works and shows subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Scripture-LM" in result.stdout
    assert "corpus" in result.stdout
    assert "tokenizer" in result.stdout
    assert "encode" in result.stdout
    assert "train" in result.stdout
    assert "config" in result.stdout
    assert "generate" in result.stdout
    assert "evaluate" in result.stdout
    assert "compare" in result.stdout
    assert "experiment" in result.stdout


@pytest.mark.parametrize(
    "subcommand",
    [
        ["corpus", "--help"],
        ["tokenizer", "--help"],
        ["encode", "--help"],
        ["train", "--help"],
        ["config", "--help"],
        ["config", "show", "--help"],
        ["generate", "--help"],
        ["evaluate", "--help"],
        ["compare", "--help"],
        ["experiment", "--help"],
    ],
)
def test_subcommand_help(subcommand: list[str]) -> None:
    """Verify all subcommands provide clean help messages."""
    result = runner.invoke(app, subcommand)
    assert result.exit_code == 0
    assert "Usage:" in result.stdout or "--help" in result.stdout


def test_train_cli_with_bpe_and_temperature() -> None:
    """Verify train CLI resolves BPE config and temperature sampling overrides."""
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            "configs/bpe.toml",
            "--sampling-mode",
            "temperature",
            "--sampling-alpha",
            "0.5",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Resolved Configuration" in result.stdout or "Training Configuration" in result.stdout
    assert "temperature" in result.stdout
    assert "0.5 (active)" in result.stdout
    assert "Configuration successfully validated!" in result.stdout


def test_train_cli_with_char_config() -> None:
    """Verify train CLI resolves Character config cleanly."""
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            "configs/char.toml",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "character" in result.stdout
    assert "2048" in result.stdout
    assert "Configuration successfully validated!" in result.stdout


def test_train_cli_invalid_alpha_fails() -> None:
    """Verify train CLI exits with non-zero code on invalid sampling alpha."""
    result = runner.invoke(
        app,
        [
            "train",
            "--sampling-alpha",
            "2.5",
        ],
    )
    assert result.exit_code != 0
    assert "Configuration validation error" in result.stdout


def test_train_cli_flag_overrides() -> None:
    """Verify train CLI flag overrides for seed, device, compile, and epochs."""
    result = runner.invoke(
        app,
        [
            "train",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--no-compile",
            "--effective-epochs",
            "10",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "42" in result.stdout
    assert "cpu" in result.stdout
    assert "False" in result.stdout
    assert "10" in result.stdout


def test_train_cli_without_data_fails() -> None:
    """Verify train CLI fails non-zero with helpful error when encoded data is missing."""
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            "configs/bpe.toml",
        ],
    )
    assert result.exit_code != 0
    assert "ERROR: encoded BPE dataset not found" in result.stdout


def test_config_show_command() -> None:
    """Verify scripture-lm config show displays resolved table."""
    result = runner.invoke(app, ["config", "show", "--config", "configs/bpe.toml"])
    assert result.exit_code == 0
    assert "bpe" in result.stdout
    assert "4096" in result.stdout

    result_char = runner.invoke(app, ["config", "show", "--config", "configs/char.toml"])
    assert result_char.exit_code == 0
    assert "character" in result_char.stdout
    assert "2048" in result_char.stdout


def test_corpus_audit_cli_default() -> None:
    """Verify scripture-lm corpus audit runs cleanly against default empty manifest."""
    result = runner.invoke(app, ["corpus", "audit"])
    assert result.exit_code == 0
    assert "Corpus Audit Summary" in result.stdout
    assert "PASSED" in result.stdout


def test_corpus_prepare_cli_empty_manifest() -> None:
    """Verify scripture-lm corpus prepare cleanly exits when manifest has no documents."""
    result = runner.invoke(app, ["corpus", "prepare"])
    assert result.exit_code == 0
    assert "Corpus manifest is empty" in result.stdout


def test_corpus_stats_missing_manifest_fails() -> None:
    """Verify scripture-lm corpus stats fails with helpful error when splits not prepared."""
    result = runner.invoke(
        app,
        ["corpus", "stats", "--split-manifest", "non_existent_path.json"],
    )
    assert result.exit_code != 0
    assert "Split manifest not found" in result.stdout
