"""Regression coverage for the encoder/trainer metadata filename contract."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from scripture_lm.cli import app
from scripture_lm.config import ScriptureLMConfig
from scripture_lm.data.chunk_index import (
    ENCODING_PROVENANCE_FILENAME,
    encoding_provenance_path,
)
from scripture_lm.data.encode import encode_dataset
from scripture_lm.tokenization.base import BaseTokenizer
from scripture_lm.tokenization.bpe import train_bpe_tokenizer
from scripture_lm.tokenization.character import build_character_tokenizer
from scripture_lm.training.trainer import verify_encoding_provenance
from tests.test_data import create_mini_corpus


@pytest.mark.parametrize("kind", ["bpe", "character"])
def test_real_encoder_output_passes_training_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Use actual encoder output, rather than manually inventing trainer metadata."""
    monkeypatch.chdir(tmp_path)
    corpus, normalized, lock, split = create_mini_corpus(tmp_path)
    artifacts = tmp_path / "artifacts" / "tokenizers"
    tokenizer: BaseTokenizer
    if kind == "bpe":
        tokenizer, _ = train_bpe_tokenizer(
            split_manifest_path=split,
            corpus_lock_path=lock,
            normalized_dir=normalized,
            output_dir=artifacts,
            vocab_size=350,
        )
    else:
        tokenizer, _ = build_character_tokenizer(
            split_manifest_path=split,
            corpus_lock_path=lock,
            normalized_dir=normalized,
            output_dir=artifacts,
        )
    _, encoded = encode_dataset(
        tokenizer=tokenizer,
        split_manifest_path=split,
        corpus_lock_path=lock,
        normalized_dir=normalized,
        output_base_dir=tmp_path / "encoded",
        context_length=32,
    )
    config = ScriptureLMConfig.model_validate({"tokenizer": {"type": kind, "context_length": 32}})
    # The helper fixture stores lock/split beside its encoded directory.
    data_root = tmp_path / "training_data"
    (data_root / "splits").mkdir(parents=True)
    (data_root / "corpus_lock.json").write_bytes(lock.read_bytes())
    (data_root / "splits" / "split_manifest.json").write_bytes(split.read_bytes())
    (tmp_path / "encoded").rename(data_root / "encoded")
    _, _, loaded, _ = verify_encoding_provenance(
        config, data_root=data_root, corpus_root=corpus, artifacts_root=artifacts.parent
    )
    assert loaded == encoded
    assert (data_root / "encoded" / kind / ENCODING_PROVENANCE_FILENAME).is_file()
    assert not (data_root / "encoded" / kind / "encoding_metadata.json").exists()


def test_provenance_path_supports_legacy_and_prefers_canonical(tmp_path: Path) -> None:
    canonical = tmp_path / ENCODING_PROVENANCE_FILENAME
    legacy = tmp_path / "encoding_metadata.json"
    assert encoding_provenance_path(tmp_path) == canonical
    legacy.write_text("{}", encoding="utf-8")
    assert encoding_provenance_path(tmp_path) == legacy
    canonical.write_text("{}", encoding="utf-8")
    assert encoding_provenance_path(tmp_path) == canonical


def test_train_cli_accepts_encoder_filename_without_starting_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "data" / "encoded" / "bpe"
    directory.mkdir(parents=True)
    (directory / ENCODING_PROVENANCE_FILENAME).write_text("{}", encoding="utf-8")
    config = tmp_path / "bpe.toml"
    config.write_text('[tokenizer]\ntype = "bpe"\n', encoding="utf-8")
    trainer = Mock()
    monkeypatch.setattr("scripture_lm.training.trainer.Trainer", trainer)
    result = CliRunner().invoke(app, ["train", "--config", str(config)])
    assert result.exit_code == 0, result.output
    trainer.return_value.train.assert_called_once_with()
