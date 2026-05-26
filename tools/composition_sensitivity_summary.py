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
        "Qwen/Qwen2.5-3B": "Qwen-3B",
        "Qwen/Qwen2.5-3B-Instruct": "Qwen-3B-Instruct",
        "/jumbo/lisp/f004ndc/models/mistral-7b-v0.1": "Mistral-7B",
        "/jumbo/lisp/f004ndc/models/olmo-2-7b": "OLMo-2-7B",
    }.get(model_name, model_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize Exp16 composition sensitivity across direction pools "
            "(mixed vs StereoSet-only vs CrowS-only) under identical heldout slices."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--primary-contrast", default="primary_inject_anti_minus_remove_stereo")
    return p.parse_args()


def _load_primary(exp16_dir: Path, primary_contrast: str) -> dict[str, Any]:
    contrast_path = exp16_dir / "tables" / "asymmetry_contrast.csv"
    pair_path = exp16_dir / "tables" / "asymmetry_pair_deltas.csv"
    if not contrast_path.exists() or not pair_path.exists():
        raise FileNotFoundError(f"Missing contrast/pair tables in {exp16_dir}")
    cdf = pd.read_csv(contrast_path)
    row = cdf[cdf["contrast"] == primary_contrast]
    if row.empty:
        raise ValueError(f"Primary contrast {primary_contrast} not found in {contrast_path}")
    r0 = row.iloc[0]
    pdf = pd.read_csv(pair_path)
    if "pair_id" not in pdf.columns or "primary_score_contrast" not in pdf.columns:
        raise ValueError(f"Missing pair-level columns in {pair_path}")
    pair_series = pd.to_numeric(pdf["primary_score_contrast"], errors="coerce")
    pairs = {
        str(pid): float(val)
        for pid, val in zip(pdf["pair_id"].astype(str).tolist(), pair_series.tolist())
        if np.isfinite(val)
    }
    return {
        "n_pairs": int(_to_float(r0.get("n_pairs", 0))),
        "mean_score_contrast": _to_float(r0.get("mean_score_contrast", "")),
        "score_ci_low": _to_float(r0.get("mean_score_contrast_ci_low", "")),
        "score_ci_high": _to_float(r0.get("mean_score_contrast_ci_high", "")),
        "p_score": _to_float(r0.get("paired_p_score_sign", "")),
        "q_score": _to_float(r0.get("q_score_sign", "")),
        "pairs": pairs,
    }


def _signed_p_twosided(deltas: np.ndarray) -> float:
    # Two-sided sign-test p-value on non-zero deltas; normal approximation is sufficient for n>=20.
    nz = deltas[deltas != 0]
    n = nz.size
    if n == 0:
        return float("nan")
    k = int(np.sum(nz > 0))
    # Exact binomial two-sided via tail doubling around 0.5.
    from math import comb

    probs = np.array([comb(n, i) * (0.5**n) for i in range(n + 1)], dtype=float)
    p_obs = probs[k]
    p = float(np.sum(probs[probs <= p_obs + 1e-15]))
    return min(1.0, p)


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)

        run_dirs = {
            "mixed": payload.get("exp16_canonical_run_dir", ""),
            "stereoset_only": payload.get("exp16_source_stereoset_run_dir", ""),
            "crows_only": payload.get("exp16_source_crows_run_dir", ""),
        }

        stats_by_pool: dict[str, dict[str, Any]] = {}
        for pool, rd in run_dirs.items():
            if not rd:
                missing.append(f"{model_name}:{pool}:missing_run_dir")
                continue
            exp16_dir = Path(str(rd))
            try:
                vals = _load_primary(exp16_dir, args.primary_contrast)
            except Exception as exc:
                missing.append(f"{model_name}:{pool}:{exc}")
                continue
            stats_by_pool[pool] = vals
            pool_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "direction_pool": pool,
                    "exp16_run_dir": str(exp16_dir),
                    "n_pairs": vals["n_pairs"],
                    "mean_score_contrast": vals["mean_score_contrast"],
                    "score_ci_low": vals["score_ci_low"],
                    "score_ci_high": vals["score_ci_high"],
                    "p_score_sign": vals["p_score"],
                    "q_score_within_run": vals["q_score"],
                }
            )

        if "mixed" not in stats_by_pool:
            continue
        mixed_pairs = stats_by_pool["mixed"]["pairs"]
        for pool in ("stereoset_only", "crows_only"):
            if pool not in stats_by_pool:
                continue
            other_pairs = stats_by_pool[pool]["pairs"]
            common = sorted(set(mixed_pairs) & set(other_pairs))
            if not common:
                continue
            deltas = np.array([other_pairs[p] - mixed_pairs[p] for p in common], dtype=float)
            interaction_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "comparison": f"{pool}_minus_mixed",
                    "n_overlap_pairs": int(len(common)),
                    "mean_delta_score_contrast": float(np.mean(deltas)),
                    "delta_ci_low": float(np.quantile(deltas, 0.025)),
                    "delta_ci_high": float(np.quantile(deltas, 0.975)),
                    "p_sign": _signed_p_twosided(deltas),
                }
            )

    pool_path = out_dir / "composition_sensitivity_exp16_pool_summary.csv"
    inter_path = out_dir / "composition_sensitivity_exp16_interaction.csv"
    meta_path = out_dir / "composition_sensitivity_exp16_meta.json"

    pd.DataFrame(
        pool_rows,
        columns=[
            "model",
            "model_label",
            "direction_pool",
            "exp16_run_dir",
            "n_pairs",
            "mean_score_contrast",
            "score_ci_low",
            "score_ci_high",
            "p_score_sign",
            "q_score_within_run",
        ],
    ).to_csv(pool_path, index=False)
    pd.DataFrame(
        interaction_rows,
        columns=[
            "model",
            "model_label",
            "comparison",
            "n_overlap_pairs",
            "mean_delta_score_contrast",
            "delta_ci_low",
            "delta_ci_high",
            "p_sign",
        ],
    ).to_csv(inter_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "primary_contrast": args.primary_contrast,
        "pool_summary": str(pool_path),
        "interaction": str(inter_path),
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(pool_path)
    print(inter_path)
    print(meta_path)


if __name__ == "__main__":
    main()
