#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.registry import EXPERIMENTS


def discover_manifests(results_root: Path) -> list[Path]:
    return sorted(results_root.glob("*/*/run-*/manifest.json"))


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_keys(manifests: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for manifest in manifests:
        metrics = manifest.get("metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(key)
    return sorted(keys)


def write_index_json(summary_dir: Path, manifests: list[dict[str, Any]], grouped: dict[str, list[dict[str, Any]]]) -> None:
    experiments_payload: dict[str, Any] = {}
    for exp_id in sorted(EXPERIMENTS):
        runs = grouped.get(exp_id, [])
        durations = [run.get("duration_seconds") for run in runs if isinstance(run.get("duration_seconds"), (int, float))]
        completed = sum(1 for run in runs if run.get("status") == "completed")
        failed = sum(1 for run in runs if run.get("status") == "failed")
        latest = runs[-1] if runs else None

        metric_means: dict[str, float] = {}
        values_by_metric: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            for k, v in run.get("metrics", {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values_by_metric[k].append(float(v))
        for k in sorted(values_by_metric):
            metric_means[k] = round(statistics.fmean(values_by_metric[k]), 6)

        experiments_payload[exp_id] = {
            "title": EXPERIMENTS[exp_id].title,
            "slug": EXPERIMENTS[exp_id].slug,
            "runs": len(runs),
            "completed": completed,
            "failed": failed,
            "latest_run_id": latest.get("run_id") if latest else None,
            "latest_run_date_utc": latest.get("run_date_utc") if latest else None,
            "avg_duration_seconds": round(statistics.fmean(durations), 6) if durations else None,
            "metric_means": metric_means,
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(manifests),
        "experiments": experiments_payload,
    }
    out_path = summary_dir / "results_index.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_runs_json(summary_dir: Path, manifests: list[dict[str, Any]]) -> None:
    out_path = summary_dir / "all_runs.json"
    out_path.write_text(json.dumps(manifests, indent=2, sort_keys=True), encoding="utf-8")


def write_runs_csv(summary_dir: Path, manifests: list[dict[str, Any]]) -> None:
    numeric_metrics = metric_keys(manifests)
    fixed_fields = [
        "experiment_id",
        "experiment_slug",
        "experiment_title",
        "run_date_utc",
        "run_id",
        "status",
        "started_at_utc",
        "ended_at_utc",
        "duration_seconds",
        "git_commit",
        "run_dir",
    ]
    parameter_fields = sorted(
        {f"param_{k}" for m in manifests for k in m.get("parameters", {}).keys()}
    )
    metric_fields = [f"metric_{k}" for k in numeric_metrics]
    fieldnames = [*fixed_fields, *parameter_fields, *metric_fields]

    out_path = summary_dir / "all_runs.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for manifest in manifests:
            row = {key: manifest.get(key) for key in fixed_fields}
            for k, v in manifest.get("parameters", {}).items():
                row[f"param_{k}"] = v
            for metric_key in numeric_metrics:
                row[f"metric_{metric_key}"] = manifest.get("metrics", {}).get(metric_key)
            writer.writerow(row)


def write_overview_markdown(summary_dir: Path, manifests: list[dict[str, Any]], grouped: dict[str, list[dict[str, Any]]]) -> None:
    total = len(manifests)
    completed = sum(1 for m in manifests if m.get("status") == "completed")
    failed = sum(1 for m in manifests if m.get("status") == "failed")

    lines: list[str] = []
    lines.append("# StereACL Results Overview")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Total runs: {total}")
    lines.append(f"- Completed: {completed}")
    lines.append(f"- Failed: {failed}")
    lines.append("")
    lines.append("## Per-Experiment Summary")
    lines.append("")
    lines.append("| ID | Experiment | Runs | Completed | Failed | Latest Date | Latest Run |")
    lines.append("|---|---|---:|---:|---:|---|---|")

    for exp_id in sorted(EXPERIMENTS):
        runs = grouped.get(exp_id, [])
        comp = sum(1 for r in runs if r.get("status") == "completed")
        fail = sum(1 for r in runs if r.get("status") == "failed")
        latest = runs[-1] if runs else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    exp_id,
                    EXPERIMENTS[exp_id].title,
                    str(len(runs)),
                    str(comp),
                    str(fail),
                    str(latest.get("run_date_utc", "-")),
                    str(latest.get("run_id", "-")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Latest Runs")
    lines.append("")
    lines.append("| Ended UTC | ID | Run | Status | Duration (s) |")
    lines.append("|---|---|---|---|---:|")
    latest_runs = sorted(
        manifests,
        key=lambda m: m.get("ended_at_utc") or "",
        reverse=True,
    )[:25]
    for run in latest_runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("ended_at_utc") or "-"),
                    str(run.get("experiment_id")),
                    str(run.get("run_id")),
                    str(run.get("status")),
                    str(run.get("duration_seconds") or "-"),
                ]
            )
            + " |"
        )

    out_path = summary_dir / "overview.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results_root = PROJECT_ROOT / "results"
    summary_dir = results_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = discover_manifests(results_root)
    manifests = [load_manifest(path) for path in manifest_paths]
    manifests = sorted(
        manifests,
        key=lambda m: (
            m.get("run_date_utc") or "",
            m.get("experiment_id") or "",
            m.get("run_id") or "",
        ),
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        grouped[manifest.get("experiment_id", "unknown")].append(manifest)

    write_runs_json(summary_dir, manifests)
    write_runs_csv(summary_dir, manifests)
    write_index_json(summary_dir, manifests, grouped)
    write_overview_markdown(summary_dir, manifests, grouped)

    print(f"Compiled {len(manifests)} runs into {summary_dir}")


if __name__ == "__main__":
    main()

