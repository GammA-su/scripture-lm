"""Write-once scientific artifacts and atomic execution status updates."""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from scripture_lm.config import ScriptureLMConfig
from scripture_lm.experiments.matrix import config_hash, get_baseline, scientific_config


def timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def serialize_toml_dict(data: dict[str, Any], prefix: str = "") -> str:
    """Serialize config dictionaries, omitting nulls unsupported by TOML."""
    lines = []
    for key, value in data.items():
        if value is not None and not isinstance(value, dict):
            lines.append(f"{json.dumps(key)} = {json.dumps(value, ensure_ascii=False)}")
    for key, value in data.items():
        if isinstance(value, dict):
            table = f"{prefix}.{json.dumps(key)}" if prefix else json.dumps(key)
            lines.extend([f"\n[{table}]", serialize_toml_dict(value, table)])
    return "\n".join(lines) + "\n"


def make_spec(config: ScriptureLMConfig, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "experiment_v1",
        "configuration": scientific_config(config),
        "provenance": provenance,
    }


def read_spec(run_dir: Path) -> dict[str, Any]:
    spec = tomllib.loads((run_dir / "experiment_config.toml").read_text(encoding="utf-8"))
    if spec.get("schema_version") != "experiment_v1":
        raise ValueError(f"Unsupported experiment specification schema: {run_dir}")
    config = ScriptureLMConfig.model_validate(spec["configuration"])
    spec["configuration"] = scientific_config(config)
    digest = (run_dir / "experiment_config.sha256").read_text(encoding="utf-8").strip()
    if config_hash(spec) != digest:
        raise ValueError(f"Immutable experiment configuration hash mismatch: {run_dir}")
    saved = ScriptureLMConfig.model_validate(
        tomllib.loads((run_dir / "config.toml").read_text(encoding="utf-8"))
    )
    if scientific_config(saved) != spec["configuration"]:
        raise ValueError(
            f"config.toml disagrees with immutable experiment specification: {run_dir}"
        )
    return spec


def check_reserved_name(run_dir: Path, config: ScriptureLMConfig) -> None:
    """Direct training cannot impersonate a baseline with changed scientific settings."""
    try:
        baseline = get_baseline(run_dir.name)
    except ValueError:
        return
    if scientific_config(config) != scientific_config(baseline.resolve()):
        raise ValueError(f"Reserved baseline name {run_dir.name} requires canonical settings")


def write_once(path: Path, text: str) -> None:
    """Never overwrite scientific artifacts, even with identical content."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to overwrite immutable artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def prepare_run(
    run_dir: Path, config: ScriptureLMConfig, provenance: dict[str, Any], *, resume: bool
) -> dict[str, Any]:
    """Check identity before logs/models are opened; persist the initial snapshot once."""
    check_reserved_name(run_dir, config)
    spec = make_spec(config, provenance)
    if run_dir.exists() and any(run_dir.iterdir()):
        if read_spec(run_dir) != spec:
            raise ValueError(f"Incompatible experiment configuration or provenance: {run_dir}")
        if not resume:
            raise ValueError(f"Run already exists; use explicit --resume: {run_dir}")
        return spec
    if resume:
        raise ValueError(f"Cannot resume missing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_once(run_dir / "config.toml", serialize_toml_dict(config.model_dump()))
    write_once(run_dir / "experiment_config.toml", serialize_toml_dict(spec))
    write_once(run_dir / "experiment_config.sha256", config_hash(spec) + "\n")
    return spec


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    if not run_dir.exists():
        return {"status": "missing"}
    if not path.is_file():
        raise ValueError(f"Run has no execution status; cannot infer completion: {run_dir}")
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") not in {"planned", "running", "interrupted", "failed", "completed"}:
        raise ValueError(f"Invalid execution status: {run_dir}")
    return result


def update_status(run_dir: Path, status: str, **details: Any) -> dict[str, Any]:
    if status not in {"planned", "running", "interrupted", "failed", "completed"}:
        raise ValueError(f"Invalid execution status: {status}")
    path = run_dir / "run_status.json"
    previous = read_status(run_dir) if path.exists() else {}
    previous.update(details)
    previous.update(status=status, updated_at=timestamp())
    atomic_json(path, previous)
    return previous
