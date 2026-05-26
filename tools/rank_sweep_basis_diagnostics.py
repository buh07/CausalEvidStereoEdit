#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rank-sweep basis diagnostics: compare raw-direction and SVD-orth basis sweep monotonicity "
            "and minima from Exp07 outputs."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def _summarize_mode(df: pd.DataFrame, basis_mode: str) -> dict[str, Any] | None:
    g = df[df["basis_mode"] == basis_mode].copy()
    if g.empty:
        return None
    k_means = g.groupby("k", as_index=False)["stereotype_score"].mean().sort_values("k")
    if k_means.empty:
        return None
    ks = k_means["k"].to_numpy(dtype=float)
    vals = k_means["stereotype_score"].to_numpy(dtype=float)
    diffs = np.diff(vals)
    violations = int(np.sum(diffs > 0))
    min_idx = int(np.argmin(vals))
    return {
        "basis_mode": basis_mode,
        "n_k": int(vals.size),
        "min_score": float(vals[min_idx]),
        "k_at_min": int(ks[min_idx]),
        "mean_score": float(np.mean(vals)),
        "monotonicity_violations": violations,
    }


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)
        exp07_dir_raw = payload.get("exp07_run_dir", "")
        if not exp07_dir_raw:
            missing.append(f"{model_name}:missing_exp07")
            continue
        rank_path = Path(str(exp07_dir_raw)) / "tables" / "rank_sweep.csv"
        if not rank_path.exists():
            missing.append(f"{model_name}:missing_rank_sweep")
            continue

        df = pd.read_csv(rank_path)
        if df.empty:
            continue
        if "basis_mode" not in df.columns:
            df["basis_mode"] = "svd"

        svd = _summarize_mode(df, "svd")
        raw = _summarize_mode(df, "raw")
        for rec in [svd, raw]:
            if rec is None:
                continue
            mode_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "basis_mode": rec["basis_mode"],
                    "n_k": rec["n_k"],
                    "min_score": rec["min_score"],
                    "k_at_min": rec["k_at_min"],
                    "mean_score": rec["mean_score"],
                    "monotonicity_violations": rec["monotonicity_violations"],
                }
            )

        if svd is not None and raw is not None:
            compare_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "raw_minus_svd_min_score": float(raw["min_score"] - svd["min_score"]),
                    "raw_minus_svd_mean_score": float(raw["mean_score"] - svd["mean_score"]),
                    "raw_minus_svd_monotonicity_violations": int(raw["monotonicity_violations"] - svd["monotonicity_violations"]),
                }
            )

    mode_path = out_dir / "rank_sweep_basis_mode_summary.csv"
    cmp_path = out_dir / "rank_sweep_basis_mode_comparison.csv"
    meta_path = out_dir / "rank_sweep_basis_diagnostics_meta.json"

    pd.DataFrame(
        mode_rows,
        columns=[
            "model",
            "model_label",
            "basis_mode",
            "n_k",
            "min_score",
            "k_at_min",
            "mean_score",
            "monotonicity_violations",
        ],
    ).to_csv(mode_path, index=False)

    pd.DataFrame(
        compare_rows,
        columns=[
            "model",
            "model_label",
            "raw_minus_svd_min_score",
            "raw_minus_svd_mean_score",
            "raw_minus_svd_monotonicity_violations",
        ],
    ).to_csv(cmp_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "mode_summary": str(mode_path),
        "comparison": str(cmp_path),
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(mode_path)
    print(cmp_path)
    print(meta_path)


if __name__ == "__main__":
    main()
