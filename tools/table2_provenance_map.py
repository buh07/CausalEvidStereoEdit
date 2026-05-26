#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Draft family used for the earlier manuscript Table 2 values (+0.100/+0.050/+0.250).
OLD_RUNS_DEFAULT = {
    "google/gemma-2-2b": PROJECT_ROOT / "results/16_asymmetry_matrix/2026-05-08/run-005",
    "google/gemma-2-2b-it": PROJECT_ROOT / "results/16_asymmetry_matrix/2026-05-08/run-006",
    "meta-llama/Llama-3.2-3B": PROJECT_ROOT / "results/16_asymmetry_matrix/2026-05-08/run-007",
}


def _to_float(v: Any) -> float:
    try:
        if v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _bh(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    if m == 0:
        return []
    finite = [(i, float(p)) for i, p in enumerate(p_vals) if np.isfinite(p)]
    q = [float("nan")] * m
    if not finite:
        return q
    finite.sort(key=lambda x: x[1])
    running = 1.0
    k = len(finite)
    for rev_rank, (idx, p) in enumerate(reversed(finite), start=1):
        rank = k - rev_rank + 1
        candidate = min(running, (p * k) / rank)
        running = candidate
        q[idx] = max(0.0, min(1.0, float(candidate)))
    return q


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Map old Table-2 draft run family values to frozen run-family values with explicit run-dir provenance."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--primary-contrast", default="primary_inject_anti_minus_remove_stereo")
    return p.parse_args()


def _label(model: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model, model)


def _load_primary(path: Path, contrast: str) -> dict[str, float]:
    table = path / "tables" / "asymmetry_contrast.csv"
    if not table.exists():
        raise FileNotFoundError(table)
    df = pd.read_csv(table)
    row = df[df["contrast"] == contrast]
    if row.empty:
        raise ValueError(f"Missing contrast {contrast} in {table}")
    r = row.iloc[0]
    return {
        "n_pairs": _to_float(r.get("n_pairs", "")),
        "mean_score_contrast": _to_float(r.get("mean_score_contrast", "")),
        "score_ci_low": _to_float(r.get("mean_score_contrast_ci_low", "")),
        "score_ci_high": _to_float(r.get("mean_score_contrast_ci_high", "")),
        "p_score": _to_float(r.get("paired_p_score_sign", "")),
        "q_score_within_table": _to_float(r.get("q_score_sign", "")),
    }


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    mf = run_dir / "manifest.json"
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_dependencies(run_dir: Path) -> dict[str, Any]:
    dep = run_dir / "artifacts" / "dependencies.json"
    if not dep.exists():
        return {}
    try:
        return json.loads(dep.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _factorized_diff_note(old_manifest: dict[str, Any], old_dep: dict[str, Any], frozen_manifest: dict[str, Any], frozen_dep: dict[str, Any]) -> str:
    factors: list[str] = []
    old_seed = old_manifest.get("parameters", {}).get("seed", "")
    new_seed = frozen_manifest.get("parameters", {}).get("seed", "")
    if old_seed != new_seed:
        factors.append(f"seed {old_seed}->{new_seed}")
    old_held = old_manifest.get("parameters", {}).get("heldout_pairs", "")
    new_held = frozen_manifest.get("parameters", {}).get("heldout_pairs", "")
    if old_held != new_held:
        factors.append(f"heldout_pairs {old_held}->{new_held}")
    old_eval = old_dep.get("eval_exp1_run_dir", "")
    new_eval = frozen_dep.get("eval_exp1_run_dir", "")
    if old_eval != new_eval:
        factors.append("eval_exp1 lineage changed")
    old_dir = old_dep.get("directions_exp1_run_dir", "")
    new_dir = frozen_dep.get("directions_exp1_run_dir", "")
    if old_dir != new_dir:
        factors.append("directions_exp1 lineage changed")
    if not factors:
        factors.append("no major parameter lineage differences detected")
    return "; ".join(factors)


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    frozen_pvals: list[float] = []
    frozen_row_idx: list[int] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)

        old_dir = OLD_RUNS_DEFAULT.get(model_name)
        frozen_dir_raw = payload.get("exp16_canonical_run_dir", "")
        if old_dir is None or not old_dir.exists():
            missing.append(f"{model_name}:missing_old_run")
            continue
        if not frozen_dir_raw:
            missing.append(f"{model_name}:missing_frozen_run")
            continue
        frozen_dir = Path(str(frozen_dir_raw))
        if not frozen_dir.exists():
            missing.append(f"{model_name}:frozen_run_not_found")
            continue

        try:
            old = _load_primary(old_dir, args.primary_contrast)
            frozen = _load_primary(frozen_dir, args.primary_contrast)
        except Exception as exc:
            missing.append(f"{model_name}:{exc}")
            continue
        old_manifest = _load_manifest(old_dir)
        frozen_manifest = _load_manifest(frozen_dir)
        old_dep = _load_dependencies(old_dir)
        frozen_dep = _load_dependencies(frozen_dir)
        note = _factorized_diff_note(old_manifest, old_dep, frozen_manifest, frozen_dep)

        row = {
            "model": model_name,
            "model_label": label,
            "old_run_family": "2026-05-08_draft_family",
            "old_run_dir": str(old_dir),
            "frozen_run_family": run_map.get("run_tag", "frozen_run_map"),
            "frozen_run_dir": str(frozen_dir),
            "contrast": args.primary_contrast,
            "old_mean_score_contrast": old["mean_score_contrast"],
            "frozen_mean_score_contrast": frozen["mean_score_contrast"],
            "delta_frozen_minus_old": frozen["mean_score_contrast"] - old["mean_score_contrast"],
            "old_score_ci_low": old["score_ci_low"],
            "old_score_ci_high": old["score_ci_high"],
            "frozen_score_ci_low": frozen["score_ci_low"],
            "frozen_score_ci_high": frozen["score_ci_high"],
            "old_p_score": old["p_score"],
            "frozen_p_score": frozen["p_score"],
            "old_q_within_table": old["q_score_within_table"],
            "frozen_q_within_table": frozen["q_score_within_table"],
            "old_n_pairs": old["n_pairs"],
            "frozen_n_pairs": frozen["n_pairs"],
            "old_seed": old_manifest.get("parameters", {}).get("seed", ""),
            "frozen_seed": frozen_manifest.get("parameters", {}).get("seed", ""),
            "old_heldout_pairs": old_manifest.get("parameters", {}).get("heldout_pairs", ""),
            "frozen_heldout_pairs": frozen_manifest.get("parameters", {}).get("heldout_pairs", ""),
            "old_eval_exp1_run_dir": old_dep.get("eval_exp1_run_dir", ""),
            "frozen_eval_exp1_run_dir": frozen_dep.get("eval_exp1_run_dir", ""),
            "old_directions_exp1_run_dir": old_dep.get("directions_exp1_run_dir", ""),
            "frozen_directions_exp1_run_dir": frozen_dep.get("directions_exp1_run_dir", ""),
            "factorized_change_note": note,
            "provenance_note": (
                "Old values come from the 2026-05-08 draft run family; frozen values come from the May-ARR "
                "frozen run map. Table-2 inference in the current draft uses across-model BH on frozen p-values."
            ),
        }
        rows.append(row)
        frozen_pvals.append(frozen["p_score"])
        frozen_row_idx.append(len(rows) - 1)

    # Add across-model BH for frozen primary contrasts (current Table-2 convention).
    q_across = _bh(frozen_pvals)
    for idx, q in zip(frozen_row_idx, q_across, strict=False):
        rows[idx]["frozen_q_across_model_bh"] = q

    out_csv = out_dir / "table2_provenance_map.csv"
    columns = [
        "model",
        "model_label",
        "old_run_family",
        "old_run_dir",
        "frozen_run_family",
        "frozen_run_dir",
        "contrast",
        "old_mean_score_contrast",
        "frozen_mean_score_contrast",
        "delta_frozen_minus_old",
        "old_score_ci_low",
        "old_score_ci_high",
        "frozen_score_ci_low",
        "frozen_score_ci_high",
        "old_p_score",
        "frozen_p_score",
        "old_q_within_table",
        "frozen_q_within_table",
        "frozen_q_across_model_bh",
        "old_n_pairs",
        "frozen_n_pairs",
        "old_seed",
        "frozen_seed",
        "old_heldout_pairs",
        "frozen_heldout_pairs",
        "old_eval_exp1_run_dir",
        "frozen_eval_exp1_run_dir",
        "old_directions_exp1_run_dir",
        "frozen_directions_exp1_run_dir",
        "factorized_change_note",
        "provenance_note",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)

    meta = {
        "run_map": str(run_map_path),
        "output": str(out_csv),
        "primary_contrast": args.primary_contrast,
        "old_runs_default": {k: str(v) for k, v in OLD_RUNS_DEFAULT.items()},
        "missing": missing,
    }
    meta_path = out_dir / "table2_provenance_map_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(out_csv)
    print(meta_path)


if __name__ == "__main__":
    main()
