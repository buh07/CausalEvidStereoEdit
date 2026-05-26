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


def _to_float_series(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df.get(col), errors="coerce").to_numpy(dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Proxy plausibility-confound stratification: bin pairs by a plausibility-gap proxy and "
            "compare intervention effect concentration across bins."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--bins", type=int, default=3)
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

    rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
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
        if df.empty:
            continue

        b_st = _to_float_series(df, "baseline_margin_stereo")
        b_an = _to_float_series(df, "baseline_margin_anti")
        inj = _to_float_series(df, "inject_on_anti_score_delta")
        rem = _to_float_series(df, "remove_on_stereo_score_delta")
        prim = _to_float_series(df, "primary_score_contrast")

        # Proxy: stronger stereo-vs-anti baseline separation may reflect plausibility/fluency imbalance.
        proxy = np.abs(b_st - b_an)
        anti_unlikelihood = -b_an

        work = pd.DataFrame(
            {
                "pair_id": df.get("pair_id", pd.Series(range(len(df)))).astype(str),
                "proxy_gap": proxy,
                "anti_unlikelihood": anti_unlikelihood,
                "inject_score_delta": inj,
                "remove_score_delta": rem,
                "primary_score_contrast": prim,
            }
        )
        work = work.dropna(subset=["proxy_gap", "primary_score_contrast"])
        if work.empty:
            missing.append(f"{model_name}:no_valid_rows")
            continue

        # Quantile bins (low/mid/high by default).
        q = np.linspace(0.0, 1.0, args.bins + 1)
        edges = np.quantile(work["proxy_gap"].to_numpy(dtype=float), q)
        edges[0] = -np.inf
        edges[-1] = np.inf
        # Ensure strictly increasing edges for pd.cut robustness.
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9

        labels = [f"bin_{i+1}" for i in range(args.bins)]
        work["proxy_bin"] = pd.cut(work["proxy_gap"], bins=edges, labels=labels, include_lowest=True)

        global_primary = float(np.nanmean(work["primary_score_contrast"].to_numpy(dtype=float)))
        global_inject = float(np.nanmean(work["inject_score_delta"].to_numpy(dtype=float)))

        for b in labels:
            g = work[work["proxy_bin"] == b]
            if g.empty:
                continue
            prim_arr = g["primary_score_contrast"].to_numpy(dtype=float)
            inj_arr = g["inject_score_delta"].to_numpy(dtype=float)
            rem_arr = g["remove_score_delta"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "proxy_bin": b,
                    "n_pairs": int(len(g)),
                    "proxy_gap_mean": float(np.mean(g["proxy_gap"].to_numpy(dtype=float))),
                    "anti_unlikelihood_mean": float(np.mean(g["anti_unlikelihood"].to_numpy(dtype=float))),
                    "primary_score_contrast_mean": float(np.nanmean(prim_arr)),
                    "inject_score_delta_mean": float(np.nanmean(inj_arr)),
                    "remove_score_delta_mean": float(np.nanmean(rem_arr)),
                    "primary_minus_model_mean": float(np.nanmean(prim_arr) - global_primary),
                    "inject_minus_model_mean": float(np.nanmean(inj_arr) - global_inject),
                }
            )

        model_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "n_pairs": int(len(work)),
                "proxy_gap_q25": float(np.quantile(work["proxy_gap"].to_numpy(dtype=float), 0.25)),
                "proxy_gap_q50": float(np.quantile(work["proxy_gap"].to_numpy(dtype=float), 0.50)),
                "proxy_gap_q75": float(np.quantile(work["proxy_gap"].to_numpy(dtype=float), 0.75)),
                "global_primary_score_contrast_mean": global_primary,
                "global_inject_score_delta_mean": global_inject,
            }
        )

    rows_path = out_dir / "plausibility_proxy_bin_summary.csv"
    model_path = out_dir / "plausibility_proxy_model_summary.csv"
    meta_path = out_dir / "plausibility_proxy_stratification_meta.json"

    pd.DataFrame(
        rows,
        columns=[
            "model",
            "model_label",
            "proxy_bin",
            "n_pairs",
            "proxy_gap_mean",
            "anti_unlikelihood_mean",
            "primary_score_contrast_mean",
            "inject_score_delta_mean",
            "remove_score_delta_mean",
            "primary_minus_model_mean",
            "inject_minus_model_mean",
        ],
    ).to_csv(rows_path, index=False)

    pd.DataFrame(
        model_rows,
        columns=[
            "model",
            "model_label",
            "n_pairs",
            "proxy_gap_q25",
            "proxy_gap_q50",
            "proxy_gap_q75",
            "global_primary_score_contrast_mean",
            "global_inject_score_delta_mean",
        ],
    ).to_csv(model_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "bin_rows": str(rows_path),
        "model_rows": str(model_path),
        "proxy_definition": "abs(baseline_margin_stereo - baseline_margin_anti)",
        "bins": args.bins,
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(rows_path)
    print(model_path)
    print(meta_path)


if __name__ == "__main__":
    main()
