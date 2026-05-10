from __future__ import annotations

import fcntl
import json
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stereacl.registry import ExperimentSpec, get_experiment


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utcnow().isoformat()


def _safe_git_commit(project_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = proc.stdout.strip()
        return commit or None
    except Exception:
        return None


def _next_run_id(date_dir: Path) -> str:
    """Allocate and create a unique run directory atomically using a file lock."""
    date_dir.mkdir(parents=True, exist_ok=True)
    lock_path = date_dir.parent / ".run_id.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        max_index = 0
        for path in date_dir.iterdir():
            if not path.is_dir():
                continue
            suffix = path.name.split("run-", 1)[-1] if path.name.startswith("run-") else ""
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        run_id = f"run-{max_index + 1:03d}"
        (date_dir / run_id).mkdir()
    return run_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class RunContext:
    project_root: Path
    results_root: Path
    experiment: ExperimentSpec
    run_date: str
    run_id: str
    run_dir: Path
    manifest_path: Path
    artifacts_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path
    manifest: dict[str, Any]

    def save_json(self, relative_path: str | Path, payload: dict[str, Any]) -> Path:
        output_path = self.run_dir / relative_path
        _write_json(output_path, payload)
        return output_path

    def save_text(self, relative_path: str | Path, text: str) -> Path:
        output_path = self.run_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path

    def append_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        event_record = {
            "ts_utc": _iso_now(),
            "event": event,
            "payload": payload or {},
        }
        events_path = self.logs_dir / "events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_record, sort_keys=True) + "\n")

    def register_artifact(
        self,
        path: Path,
        artifact_type: str,
        description: str | None = None,
    ) -> None:
        rel = path.relative_to(self.run_dir).as_posix()
        artifact_entry = {
            "path": rel,
            "type": artifact_type,
            "description": description or "",
        }
        self.manifest.setdefault("artifacts", []).append(artifact_entry)
        self._persist_manifest()

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self.manifest["metrics"] = metrics
        self._persist_manifest()

    def _persist_manifest(self) -> None:
        _write_json(self.manifest_path, self.manifest)


def start_run(
    experiment_id: str | int,
    parameters: dict[str, Any],
    project_root: Path | None = None,
    notes: str | None = None,
) -> RunContext:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    experiment = get_experiment(experiment_id)
    run_date = _utcnow().date().isoformat()
    exp_folder = f"{experiment.id}_{experiment.slug}"
    date_dir = results_root / exp_folder / run_date
    date_dir.mkdir(parents=True, exist_ok=True)

    run_id = _next_run_id(date_dir)
    run_dir = date_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    logs_dir = run_dir / "logs"
    for folder in (run_dir, artifacts_dir, figures_dir, tables_dir, logs_dir):
        folder.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": experiment.id,
        "experiment_slug": experiment.slug,
        "experiment_title": experiment.title,
        "script": experiment.script,
        "run_date_utc": run_date,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": _iso_now(),
        "ended_at_utc": None,
        "duration_seconds": None,
        "project_root": str(root),
        "run_dir": str(run_dir),
        "git_commit": _safe_git_commit(root),
        "parameters": parameters,
        "metrics": {},
        "artifacts": [],
        "notes": notes or "",
        "error": None,
    }
    _write_json(manifest_path, manifest)

    ctx = RunContext(
        project_root=root,
        results_root=results_root,
        experiment=experiment,
        run_date=run_date,
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        artifacts_dir=artifacts_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        logs_dir=logs_dir,
        manifest=manifest,
    )
    ctx.append_event("run_started", {"experiment_id": experiment.id, "run_id": run_id})
    return ctx


def complete_run(ctx: RunContext, metrics: dict[str, Any]) -> None:
    ended = _utcnow()
    started = datetime.fromisoformat(ctx.manifest["started_at_utc"])
    ctx.manifest["status"] = "completed"
    ctx.manifest["ended_at_utc"] = ended.isoformat()
    ctx.manifest["duration_seconds"] = round((ended - started).total_seconds(), 3)
    ctx.manifest["metrics"] = metrics
    ctx.manifest["error"] = None
    ctx._persist_manifest()
    ctx.append_event("run_completed", {"duration_seconds": ctx.manifest["duration_seconds"]})


def fail_run(ctx: RunContext, exc: BaseException) -> None:
    ended = _utcnow()
    started = datetime.fromisoformat(ctx.manifest["started_at_utc"])
    ctx.manifest["status"] = "failed"
    ctx.manifest["ended_at_utc"] = ended.isoformat()
    ctx.manifest["duration_seconds"] = round((ended - started).total_seconds(), 3)
    ctx.manifest["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    ctx._persist_manifest()
    ctx.append_event(
        "run_failed",
        {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )

