#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write frozen artifact manifest for May ARR fixpack runs.")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--run-map", required=True)
    p.add_argument("--requirements-lock", default="requirements.lock.txt")
    p.add_argument("--output", default="")
    p.add_argument(
        "--checksum-glob",
        action="append",
        default=[],
        help="Additional glob (relative to project root) to checksum into manifest.",
    )
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def _python_version() -> str:
    try:
        proc = subprocess.run(["python", "--version"], check=True, capture_output=True, text=True)
        return (proc.stdout or proc.stderr).strip()
    except Exception:
        return ""


def _collect_run_dirs(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and (k.endswith("run_dir") or "/results/" in v):
                out.add(v)
            else:
                _collect_run_dirs(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_run_dirs(item, out)


def _read_run_manifest(run_dir: Path) -> dict[str, Any]:
    mp = run_dir / "manifest.json"
    if not mp.exists():
        return {
            "run_dir": str(run_dir),
            "exists": False,
        }
    payload = json.loads(mp.read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "exists": True,
        "experiment_id": payload.get("experiment_id"),
        "experiment_slug": payload.get("experiment_slug"),
        "run_id": payload.get("run_id"),
        "run_date_utc": payload.get("run_date_utc"),
        "status": payload.get("status"),
        "ended_at_utc": payload.get("ended_at_utc"),
        "git_commit": payload.get("git_commit"),
        "parameters": payload.get("parameters", {}),
    }


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    if not run_map_path.exists():
        raise FileNotFoundError(f"Run map not found: {run_map_path}")

    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "results" / args.run_tag / "artifact_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requirements_lock = Path(args.requirements_lock)
    if not requirements_lock.is_absolute():
        requirements_lock = PROJECT_ROOT / requirements_lock

    run_dir_strings: set[str] = set()
    _collect_run_dirs(run_map, run_dir_strings)
    run_dirs = sorted({Path(p) for p in run_dir_strings if p})

    run_records = [_read_run_manifest(rd) for rd in run_dirs]

    checksum_patterns = [
        f"results/{args.run_tag}/reports/*.csv",
        f"results/{args.run_tag}/reports/*.json",
        "paper/build/*.png",
        "paper/sections/*.tex",
        "tools/make_paper_figures.py",
        "requirements.lock.txt",
    ]
    checksum_patterns.extend(args.checksum_glob)

    checksum_records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in checksum_patterns:
        for fp in sorted(PROJECT_ROOT.glob(pattern)):
            if fp in seen or not fp.is_file():
                continue
            seen.add(fp)
            checksum_records.append(
                {
                    "path": str(fp.relative_to(PROJECT_ROOT)),
                    "sha256": _sha256(fp),
                    "size_bytes": fp.stat().st_size,
                }
            )

    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "project_root": str(PROJECT_ROOT),
        "git_commit_head": _git_head(),
        "python_version": _python_version(),
        "requirements_lock": {
            "path": str(requirements_lock.relative_to(PROJECT_ROOT)) if requirements_lock.exists() else str(requirements_lock),
            "exists": requirements_lock.exists(),
            "sha256": _sha256(requirements_lock) if requirements_lock.exists() else "",
        },
        "run_map_path": str(run_map_path),
        "run_map": run_map,
        "run_records": run_records,
        "checksums": checksum_records,
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
    }

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
