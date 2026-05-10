#!/usr/bin/env python3
"""Summarize Exp04 outcomes by Exp01 direction extraction position.

Finds the latest completed Exp04 run per model for:
1) trait-position direction extraction in Exp01
2) prediction-position direction extraction in Exp01

Writes a comparison CSV and markdown note under results/analysis/.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = "/jumbo/lisp/f004ndc/StereACL"
EXP04_GLOB = os.path.join(
    PROJECT_ROOT, "results", "04_ablation_validation", "*", "run-*", "manifest.json"
)

MODELS = [
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B",
]


@dataclass
class Row:
    model: str
    direction_position: str
    run_dir: str
    exp1_run_dir: str
    ended_at_utc: str
    n_pairs: str
    score_delta: str
    score_ci: str
    score_q: str
    margin_delta: str
    margin_ci: str
    margin_q: str
    inject_delta: str
    inject_ci: str
    inject_q: str
    direction_norm_mean: str
    direction_norm_max: str
    direction_nonzero: str
    direction_total: str


def _read_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _direction_position_for_exp1(exp1_run_dir: str) -> str:
    if not exp1_run_dir:
        return "unknown"
    exp1_run_dir = exp1_run_dir.rstrip("/")
    manifest_path = os.path.join(exp1_run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        # Some manifests store relative paths.
        manifest_path = os.path.join(PROJECT_ROOT, exp1_run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return "unknown"
    try:
        m = _read_manifest(manifest_path)
        pos = m.get("parameters", {}).get("direction_position", "trait")
        return "prediction" if pos == "prediction" else "trait"
    except Exception:
        return "unknown"


def _read_ablation_table(run_dir: str) -> List[Dict[str, str]]:
    p = os.path.join(run_dir, "tables", "ablation_comparison.csv")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _direction_norm_stats(exp1_run_dir: str) -> Tuple[str, str, str, str]:
    if not exp1_run_dir:
        return "", "", "", ""
    exp1_run_dir = exp1_run_dir.rstrip("/")
    npz_path = os.path.join(exp1_run_dir, "artifacts", "directions_layerwise.npz")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(PROJECT_ROOT, exp1_run_dir, "artifacts", "directions_layerwise.npz")
    if not os.path.exists(npz_path):
        return "", "", "", ""
    try:
        import numpy as np  # local import to keep startup light

        data = np.load(npz_path)
        norms: List[float] = []
        for key in data.files:
            vec = data[key]
            norms.append(float(np.linalg.norm(vec)))
        if not norms:
            return "", "", "", ""
        nz = sum(1 for v in norms if v > 1e-8)
        return (
            f"{sum(norms)/len(norms):.6f}",
            f"{max(norms):.6f}",
            str(nz),
            str(len(norms)),
        )
    except Exception:
        return "", "", "", ""


def _get_condition(rows: List[Dict[str, str]], *names: str) -> Optional[Dict[str, str]]:
    for n in names:
        for r in rows:
            if r.get("condition") == n:
                return r
    return None


def _fmt_ci(lo: str, hi: str) -> str:
    if not lo or not hi:
        return ""
    try:
        return f"[{float(lo):.3f}, {float(hi):.3f}]"
    except Exception:
        return ""


def main() -> None:
    manifests = []
    for p in glob.glob(EXP04_GLOB):
        try:
            m = _read_manifest(p)
        except Exception:
            continue
        if m.get("status") != "completed":
            continue
        model = m.get("parameters", {}).get("model")
        if model not in MODELS:
            continue
        exp1 = m.get("parameters", {}).get("exp1_run_dir", "")
        position = _direction_position_for_exp1(exp1)
        manifests.append((model, position, m))

    latest: Dict[Tuple[str, str], dict] = {}
    for model, position, m in manifests:
        key = (model, position)
        if key not in latest or m.get("ended_at_utc", "") > latest[key].get("ended_at_utc", ""):
            latest[key] = m

    out_rows: List[Row] = []
    for model in MODELS:
        for position in ("trait", "prediction"):
            m = latest.get((model, position))
            if not m:
                continue
            run_dir = m["run_dir"]
            table_rows = _read_ablation_table(run_dir)
            ablation = _get_condition(
                table_rows, "direction_ablation_at_pred_pos", "direction_ablation"
            )
            inject = _get_condition(table_rows, "corrupt_to_clean")
            if not ablation:
                continue
            norm_mean, norm_max, norm_nonzero, norm_total = _direction_norm_stats(
                m.get("parameters", {}).get("exp1_run_dir", "")
            )

            out_rows.append(
                Row(
                    model=model,
                    direction_position=position,
                    run_dir=run_dir,
                    exp1_run_dir=m.get("parameters", {}).get("exp1_run_dir", ""),
                    ended_at_utc=m.get("ended_at_utc", ""),
                    n_pairs=ablation.get("n_pairs", ""),
                    score_delta=ablation.get("stereotype_score_delta", ""),
                    score_ci=_fmt_ci(
                        ablation.get("stereotype_score_delta_ci_low", ""),
                        ablation.get("stereotype_score_delta_ci_high", ""),
                    ),
                    score_q=ablation.get("q_score_sign", ""),
                    margin_delta=ablation.get("mean_margin_delta", ""),
                    margin_ci=_fmt_ci(
                        ablation.get("mean_margin_delta_ci_low", ""),
                        ablation.get("mean_margin_delta_ci_high", ""),
                    ),
                    margin_q=ablation.get("q_margin_wilcoxon", ""),
                    inject_delta=inject.get("stereotype_score_delta", "") if inject else "",
                    inject_ci=_fmt_ci(
                        inject.get("stereotype_score_delta_ci_low", "") if inject else "",
                        inject.get("stereotype_score_delta_ci_high", "") if inject else "",
                    ),
                    inject_q=inject.get("q_score_sign", "") if inject else "",
                    direction_norm_mean=norm_mean,
                    direction_norm_max=norm_max,
                    direction_nonzero=norm_nonzero,
                    direction_total=norm_total,
                )
            )

    analysis_dir = os.path.join(PROJECT_ROOT, "results", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    csv_path = os.path.join(analysis_dir, "direction_position_compare.csv")
    md_path = os.path.join(analysis_dir, "direction_position_compare.md")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "direction_position",
                "n_pairs",
                "score_delta",
                "score_ci",
                "score_q",
                "margin_delta",
                "margin_ci",
                "margin_q",
                "inject_delta",
                "inject_ci",
                "inject_q",
                "direction_norm_mean",
                "direction_norm_max",
                "direction_nonzero",
                "direction_total",
                "exp04_run_dir",
                "exp01_run_dir",
                "ended_at_utc",
            ]
        )
        for r in out_rows:
            w.writerow(
                [
                    r.model,
                    r.direction_position,
                    r.n_pairs,
                    r.score_delta,
                    r.score_ci,
                    r.score_q,
                    r.margin_delta,
                    r.margin_ci,
                    r.margin_q,
                    r.inject_delta,
                    r.inject_ci,
                    r.inject_q,
                    r.direction_norm_mean,
                    r.direction_norm_max,
                    r.direction_nonzero,
                    r.direction_total,
                    r.run_dir,
                    r.exp1_run_dir,
                    r.ended_at_utc,
                ]
            )

    by_model: Dict[str, Dict[str, Row]] = {}
    for r in out_rows:
        by_model.setdefault(r.model, {})[r.direction_position] = r

    lines = [
        "# Direction-Position Comparison",
        "",
        "Latest completed Exp04 runs split by Exp01 direction extraction position.",
        "",
        "| Model | Trait score Δ (q) | Prediction score Δ (q) | Trait inject Δ (q) | Prediction inject Δ (q) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        t = by_model.get(model, {}).get("trait")
        p = by_model.get(model, {}).get("prediction")

        def _cell(rr: Optional[Row], is_inject: bool = False) -> str:
            if not rr:
                return "n/a"
            if is_inject:
                return f"{rr.inject_delta} ({rr.inject_q or 'n/a'})"
            return f"{rr.score_delta} ({rr.score_q or 'n/a'})"

        lines.append(
            f"| {model} | {_cell(t)} | {_cell(p)} | {_cell(t, True)} | {_cell(p, True)} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
