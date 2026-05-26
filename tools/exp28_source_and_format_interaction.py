#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_float(v: Any) -> float:
    try:
        if v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _label(model_name: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model_name, model_name)


def _sign_pvalue(arr: np.ndarray) -> float:
    nz = arr[arr != 0]
    n = int(nz.size)
    if n == 0:
        return float("nan")
    k = int(np.sum(nz > 0))
    from math import comb

    probs = np.array([comb(n, i) * (0.5**n) for i in range(n + 1)], dtype=float)
    p_obs = probs[k]
    p = float(np.sum(probs[probs <= p_obs + 1e-15]))
    return min(1.0, p)


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, values.size, size=values.size)
        means.append(float(np.mean(values[idx])))
    arr = np.array(means, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build Exp28 source-stratified summaries and single-vs-span format interaction "
            "tables from run-map artifacts."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--bootstrap-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=17)
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

    source_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)

        exp28_dir_raw = payload.get("exp28_run_dir", "")
        exp16_dir_raw = payload.get("exp16_canonical_run_dir", "")
        if not exp28_dir_raw or not exp16_dir_raw:
            missing.append(f"{model_name}:missing_exp16_or_exp28")
            continue
        exp28_dir = Path(str(exp28_dir_raw))
        exp16_dir = Path(str(exp16_dir_raw))
        pair28_path = exp28_dir / "tables" / "multitoken_asymmetry_pair_deltas.csv"
        pair16_path = exp16_dir / "tables" / "asymmetry_pair_deltas.csv"
        contrast16_path = exp16_dir / "tables" / "asymmetry_contrast.csv"
        contrast28_path = exp28_dir / "tables" / "multitoken_asymmetry_contrast.csv"
        if any(not p.exists() for p in [pair28_path, pair16_path, contrast16_path, contrast28_path]):
            missing.append(f"{model_name}:missing_required_tables")
            continue

        df28 = pd.read_csv(pair28_path)
        if "source" not in df28.columns or "primary_score_contrast" not in df28.columns:
            missing.append(f"{model_name}:exp28_missing_columns")
            continue

        # Source-stratified Exp28 contrasts.
        for source_name, g in df28.groupby("source"):
            vals = pd.to_numeric(g["primary_score_contrast"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            ci_low, ci_high = _bootstrap_mean_ci(vals, n_boot=args.bootstrap_n, seed=args.seed)
            source_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "source": source_name,
                    "n_pairs": int(vals.size),
                    "mean_score_contrast": float(np.mean(vals)),
                    "score_ci_low": ci_low,
                    "score_ci_high": ci_high,
                    "p_score_sign": _sign_pvalue(vals),
                }
            )

        # Single-token vs span-level interaction (unpaired difference in means).
        c16 = pd.read_csv(contrast16_path)
        c28 = pd.read_csv(contrast28_path)
        r16 = c16[c16["contrast"] == "primary_inject_anti_minus_remove_stereo"]
        r28 = c28[c28["contrast"] == "primary_inject_anti_minus_remove_stereo"]
        if r16.empty or r28.empty:
            missing.append(f"{model_name}:missing_primary_contrast")
            continue

        single_mean = _to_float(r16.iloc[0].get("mean_score_contrast", ""))
        span_mean = _to_float(r28.iloc[0].get("mean_score_contrast", ""))

        d16 = pd.read_csv(pair16_path)
        v16 = pd.to_numeric(d16.get("primary_score_contrast"), errors="coerce").to_numpy(dtype=float)
        v28 = pd.to_numeric(df28.get("primary_score_contrast"), errors="coerce").to_numpy(dtype=float)
        v16 = v16[np.isfinite(v16)]
        v28 = v28[np.isfinite(v28)]
        if v16.size == 0 or v28.size == 0:
            missing.append(f"{model_name}:empty_pair_vectors")
            continue

        # Unpaired bootstrap for span-minus-single mean contrast.
        rng = np.random.default_rng(args.seed + 101)
        diffs = []
        for _ in range(args.bootstrap_n):
            idx16 = rng.integers(0, v16.size, size=v16.size)
            idx28 = rng.integers(0, v28.size, size=v28.size)
            diffs.append(float(np.mean(v28[idx28]) - np.mean(v16[idx16])))
        darr = np.array(diffs, dtype=float)
        interaction_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "n_single_pairs": int(v16.size),
                "n_span_pairs": int(v28.size),
                "single_mean_score_contrast": single_mean,
                "span_mean_score_contrast": span_mean,
                "span_minus_single": float(span_mean - single_mean),
                "span_minus_single_ci_low": float(np.quantile(darr, 0.025)),
                "span_minus_single_ci_high": float(np.quantile(darr, 0.975)),
                "p_directional": float(
                    2 * min(np.mean(darr <= 0.0), np.mean(darr >= 0.0))
                ),
            }
        )

    source_path = out_dir / "exp28_source_stratified_summary.csv"
    inter_path = out_dir / "exp28_single_vs_span_interaction.csv"
    meta_path = out_dir / "exp28_source_and_format_meta.json"

    pd.DataFrame(
        source_rows,
        columns=[
            "model",
            "model_label",
            "source",
            "n_pairs",
            "mean_score_contrast",
            "score_ci_low",
            "score_ci_high",
            "p_score_sign",
        ],
    ).to_csv(source_path, index=False)
    pd.DataFrame(
        interaction_rows,
        columns=[
            "model",
            "model_label",
            "n_single_pairs",
            "n_span_pairs",
            "single_mean_score_contrast",
            "span_mean_score_contrast",
            "span_minus_single",
            "span_minus_single_ci_low",
            "span_minus_single_ci_high",
            "p_directional",
        ],
    ).to_csv(inter_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "source_summary": str(source_path),
        "single_vs_span_interaction": str(inter_path),
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(source_path)
    print(inter_path)
    print(meta_path)


if __name__ == "__main__":
    main()
