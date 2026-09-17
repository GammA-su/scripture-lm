"""Sequential experiment execution using the shared training engine."""

from __future__ import annotations

import gc
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from scripture_lm.experiments.matrix import Experiment, baseline_matrix
from scripture_lm.experiments.storage import make_spec, read_spec, read_status, update_status
from scripture_lm.training.trainer import Trainer, verify_encoding_provenance


@contextmanager
def execution_lock(path: Path) -> Iterator[None]:
    """OS-released process lock: abrupt termination never leaves a held lock."""
    import sys

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f"Experiment is already running: {path.stem}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run_experiment(
    experiment: Experiment,
    *,
    runs_root: Path = Path("runs"),
    device: str | None = None,
    compile_model: bool | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one definition, never overwriting or implicitly resuming existing work."""
    config = experiment.resolve(device=device, compile_model=compile_model)
    run_dir = runs_root / experiment.name
    if dry_run:
        return {"name": experiment.name, "status": "planned", "config": config.model_dump()}
    with execution_lock(runs_root / ".locks" / f"{experiment.name}.lock"):
        _, _, provenance, _ = verify_encoding_provenance(config)
        status = read_status(run_dir)
        if status["status"] != "missing":
            if read_spec(run_dir) != make_spec(config, provenance.model_dump(mode="json")):
                raise ValueError(f"Incompatible experiment configuration or provenance: {run_dir}")
            if status["status"] == "completed":
                return {"name": experiment.name, "status": "completed", "skipped": True}
            if status["status"] == "failed":
                raise ValueError(f"Failed run retained at {run_dir}: {status.get('error', '')}")
            if not resume:
                raise ValueError(f"Run is {status['status']}; use explicit --resume: {run_dir}")
            checkpoint = run_dir / "checkpoints" / "latest"
            if not all(
                (checkpoint / name).is_file()
                for name in ("model.safetensors", "training_state.pt", "metadata.json")
            ):
                raise ValueError(f"No complete resume checkpoint at {checkpoint}; run retained")
        else:
            checkpoint = None
        trainer: Trainer | None = None
        try:
            trainer = Trainer(config=config, run_dir=run_dir, resume_checkpoint_dir=checkpoint)
            summary = trainer.train()
            return {"name": experiment.name, "status": "completed", "summary": summary}
        except BaseException as exc:
            if run_dir.is_dir() and (run_dir / "experiment_config.sha256").is_file():
                current = read_status(run_dir)
                if current["status"] not in {"failed", "interrupted"}:
                    update_status(
                        run_dir,
                        "interrupted"
                        if isinstance(exc, (KeyboardInterrupt, SystemExit))
                        else "failed",
                        error=str(exc),
                    )
            raise
        finally:
            if trainer is not None:
                trainer.metrics_logger.close()
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_baseline(
    experiments: Sequence[Experiment] | None = None,
    *,
    runs_root: Path = Path("runs"),
    device: str | None = None,
    compile_model: bool | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Execute in canonical order, propagating the first failure to the caller."""
    return [
        run_experiment(
            experiment,
            runs_root=runs_root,
            device=device,
            compile_model=compile_model,
            resume=resume,
            dry_run=dry_run,
        )
        for experiment in (experiments if experiments is not None else baseline_matrix())
    ]
