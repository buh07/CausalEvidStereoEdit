#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/jumbo/lisp/f004ndc/StereACL")
EXP10_GLOB = PROJECT_ROOT / "results" / "10_path_mediation" / "*" / "run-*" / "manifest.json"
OUT_DIR = PROJECT_ROOT / "results" / "analysis"

MODELS = [
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B",
]
TARGETS = ["residual", "attention", "mlp"]


@dataclass
class SummaryRow:
    model: str
    ablation_target: str
    run_dir: str
    run_id: str
    ended_at_utc: str
    n_rows: int
    mean_abs_score_delta: float
    std_abs_score_delta: float
    mean_abs_margin_delta: float
    std_abs_margin_delta: float
    mean_score_delta: float
    mean_margin_delta: float
    max_abs_score_delta: float
    max_abs_score_axis: str
    max_abs_score_layer: int
    max_abs_margin_delta: float
    max_abs_margin_axis: str
    max_abs_margin_layer: int


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_runs() -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    for mpath in sorted(glob.glob(str(EXP10_GLOB))):
        manifest = _load_json(Path(mpath))
        if manifest.get("status") != "completed":
            continue
        params = manifest.get("parameters", {})
        model = params.get("model")
        if model not in MODELS:
            continue
        target = params.get("ablation_target", "residual")
        if target not in TARGETS:
            continue
        key = (model, target)
        ended = manifest.get("ended_at_utc", "")
        prev = latest.get(key)
        if prev is None or ended > prev.get("ended_at_utc", ""):
            latest[key] = manifest
    return latest


def _summarize_run(manifest: dict) -> SummaryRow:
    run_dir = Path(manifest["run_dir"])
    model = manifest.get("parameters", {}).get("model", "")
    target = manifest.get("parameters", {}).get("ablation_target", "residual")

    csv_path = run_dir / "tables" / "path_mediation.csv"
    df = pd.read_csv(csv_path)

    for col in ["stereotype_score_delta", "mean_margin_delta", "layer"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["stereotype_score_delta", "mean_margin_delta", "layer"]).copy()
    if df.empty:
        raise RuntimeError(f"No numeric rows in {csv_path}")

    df["abs_score"] = df["stereotype_score_delta"].abs()
    df["abs_margin"] = df["mean_margin_delta"].abs()

    max_score_row = df.loc[df["abs_score"].idxmax()]
    max_margin_row = df.loc[df["abs_margin"].idxmax()]

    return SummaryRow(
        model=model,
        ablation_target=target,
        run_dir=str(run_dir),
        run_id=manifest.get("run_id", ""),
        ended_at_utc=manifest.get("ended_at_utc", ""),
        n_rows=int(len(df)),
        mean_abs_score_delta=float(df["abs_score"].mean()),
        std_abs_score_delta=float(df["abs_score"].std(ddof=1)),
        mean_abs_margin_delta=float(df["abs_margin"].mean()),
        std_abs_margin_delta=float(df["abs_margin"].std(ddof=1)),
        mean_score_delta=float(df["stereotype_score_delta"].mean()),
        mean_margin_delta=float(df["mean_margin_delta"].mean()),
        max_abs_score_delta=float(max_score_row["abs_score"]),
        max_abs_score_axis=str(max_score_row["axis"]),
        max_abs_score_layer=int(max_score_row["layer"]),
        max_abs_margin_delta=float(max_margin_row["abs_margin"]),
        max_abs_margin_axis=str(max_margin_row["axis"]),
        max_abs_margin_layer=int(max_margin_row["layer"]),
    )


def _write_outputs(rows: list[SummaryRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = OUT_DIR / "exp10_decomposition_summary.csv"
    md_out = OUT_DIR / "exp10_decomposition_summary.md"

    fieldnames = [
        "model",
        "ablation_target",
        "n_rows",
        "mean_abs_score_delta",
        "std_abs_score_delta",
        "mean_abs_margin_delta",
        "std_abs_margin_delta",
        "mean_score_delta",
        "mean_margin_delta",
        "max_abs_score_delta",
        "max_abs_score_axis",
        "max_abs_score_layer",
        "max_abs_margin_delta",
        "max_abs_margin_axis",
        "max_abs_margin_layer",
        "run_id",
        "ended_at_utc",
        "run_dir",
    ]

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})

    lines = [
        "# Exp10 MLP-vs-Attention Decomposition Summary",
        "",
        "Latest completed run per model x ablation target.",
        "",
        "| Model | Target | Mean±SD |Δscore| | Mean±SD |Δmargin| | Max |Δscore| (axis@layer) | Max |Δmargin| (axis@layer) | Run |",
        "|---|---|---:|---:|---|---|---|",
    ]

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.model,
                    r.ablation_target,
                    f"{r.mean_abs_score_delta:.4f}±{r.std_abs_score_delta:.4f}",
                    f"{r.mean_abs_margin_delta:.4f}±{r.std_abs_margin_delta:.4f}",
                    f"{r.max_abs_score_delta:.4f} ({r.max_abs_score_axis}@L{r.max_abs_score_layer})",
                    f"{r.max_abs_margin_delta:.4f} ({r.max_abs_margin_axis}@L{r.max_abs_margin_layer})",
                    r.run_id,
                ]
            )
            + " |"
        )

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_out}")
    print(f"Wrote {md_out}")


def main() -> None:
    latest = _latest_runs()

    missing = [(m, t) for m in MODELS for t in TARGETS if (m, t) not in latest]
    if missing:
        for model, target in missing:
            print(f"Missing completed run for {model} / {target}")

    rows: list[SummaryRow] = []
    for model in MODELS:
        for target in TARGETS:
            manifest = latest.get((model, target))
            if manifest is None:
                continue
            rows.append(_summarize_run(manifest))

    if not rows:
        raise RuntimeError("No completed Exp10 runs found for the requested model/target set.")

    rows.sort(key=lambda r: (r.model, TARGETS.index(r.ablation_target)))
    _write_outputs(rows)


if __name__ == "__main__":
    main()
