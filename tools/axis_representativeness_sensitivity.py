#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _label(model_name: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model_name, model_name)


def _sign_p(vals: np.ndarray) -> float:
    arr = vals[np.isfinite(vals)]
    nz = arr[arr != 0]
    n = int(nz.size)
    if n == 0:
        return float("nan")
    k = int(np.sum(nz > 0))
    probs = np.array([comb(n, i) * (0.5 ** n) for i in range(n + 1)], dtype=float)
    p_obs = probs[k]
    p = float(np.sum(probs[probs <= p_obs + 1e-15]))
    return min(1.0, p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Axis representativeness sensitivity for Exp16: leave-one-axis-out and equal-axis reweighted "
            "aggregates for primary matched contrast."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    loao_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)
        exp16_dir_raw = payload.get("exp16_canonical_run_dir", "")
        if not exp16_dir_raw:
            missing.append(f"{model_name}:missing_exp16")
            continue
        p16 = Path(str(exp16_dir_raw)) / "tables" / "asymmetry_pair_deltas.csv"
        if not p16.exists():
            missing.append(f"{model_name}:missing_pair_table")
            continue

        df = pd.read_csv(p16)
        if df.empty or "axis" not in df.columns or "primary_score_contrast" not in df.columns:
            missing.append(f"{model_name}:missing_axis_or_contrast")
            continue
        df["primary_score_contrast"] = pd.to_numeric(df["primary_score_contrast"], errors="coerce")
        df = df.dropna(subset=["primary_score_contrast", "axis"]).copy()
        if df.empty:
            continue

        all_vals = df["primary_score_contrast"].to_numpy(dtype=float)
        all_mean = float(np.mean(all_vals))
        all_p = _sign_p(all_vals)
        axes = sorted(df["axis"].astype(str).unique().tolist())

        axis_means = df.groupby("axis")["primary_score_contrast"].mean()
        equal_axis_mean = float(axis_means.mean()) if not axis_means.empty else float("nan")

        agg_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "n_pairs": int(len(df)),
                "n_axes": int(len(axes)),
                "pair_weighted_mean": all_mean,
                "pair_weighted_sign_p": all_p,
                "equal_axis_mean": equal_axis_mean,
                "equal_axis_minus_pair_weighted": equal_axis_mean - all_mean,
            }
        )

        for axis in axes:
            sub = df[df["axis"] != axis]
            vals = sub["primary_score_contrast"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            loao_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "dropped_axis": axis,
                    "n_pairs": int(vals.size),
                    "mean_score_contrast": float(np.mean(vals)),
                    "delta_vs_full": float(np.mean(vals) - all_mean),
                    "sign_p": _sign_p(vals),
                }
            )

    loao_path = out_dir / "axis_leave_one_out_sensitivity.csv"
    agg_path = out_dir / "axis_reweighted_aggregate_summary.csv"
    meta_path = out_dir / "axis_representativeness_sensitivity_meta.json"

    pd.DataFrame(
        loao_rows,
        columns=[
            "model",
            "model_label",
            "dropped_axis",
            "n_pairs",
            "mean_score_contrast",
            "delta_vs_full",
            "sign_p",
        ],
    ).to_csv(loao_path, index=False)

    pd.DataFrame(
        agg_rows,
        columns=[
            "model",
            "model_label",
            "n_pairs",
            "n_axes",
            "pair_weighted_mean",
            "pair_weighted_sign_p",
            "equal_axis_mean",
            "equal_axis_minus_pair_weighted",
        ],
    ).to_csv(agg_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "loao": str(loao_path),
        "aggregate": str(agg_path),
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(loao_path)
    print(agg_path)
    print(meta_path)


if __name__ == "__main__":
    main()
