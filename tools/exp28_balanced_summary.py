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


def _bootstrap_ci(arr: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, vals.size, size=vals.size)
        means[i] = float(np.mean(vals[idx]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize balanced/quota Exp28 outputs with source-span composition and "
            "single-vs-span interaction diagnostics."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--bootstrap-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=29)
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

    summary_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)
        exp28_dir_raw = payload.get("exp28_balanced_run_dir", "") or payload.get("exp28_run_dir", "")
        exp16_dir_raw = payload.get("exp16_canonical_run_dir", "")
        if not exp28_dir_raw:
            missing.append(f"{model_name}:missing_exp28")
            continue

        exp28_dir = Path(str(exp28_dir_raw))
        pair_path = exp28_dir / "tables" / "multitoken_asymmetry_pair_deltas.csv"
        contrast_path = exp28_dir / "tables" / "multitoken_asymmetry_contrast.csv"
        strata_path = exp28_dir / "tables" / "multitoken_matched_contrast_by_span.csv"
        comp_path = exp28_dir / "tables" / "multitoken_eval_composition.csv"
        dep_path = exp28_dir / "artifacts" / "dependencies.json"
        if any(not p.exists() for p in [pair_path, contrast_path, strata_path]):
            missing.append(f"{model_name}:missing_exp28_tables")
            continue

        d_pair = pd.read_csv(pair_path)
        d_contrast = pd.read_csv(contrast_path)
        d_strata = pd.read_csv(strata_path)
        deps = json.loads(dep_path.read_text(encoding="utf-8")) if dep_path.exists() else {}

        primary = d_contrast[d_contrast["contrast"] == "primary_inject_anti_minus_remove_stereo"]
        if primary.empty:
            missing.append(f"{model_name}:missing_primary_contrast")
            continue
        r = primary.iloc[0]
        summary_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "exp28_run_dir": str(exp28_dir),
                "balance_mode": deps.get("balance_mode", "unknown"),
                "source_quotas": json.dumps(deps.get("source_quotas", {}), sort_keys=True),
                "n_pairs": int(_to_float(r.get("n_pairs", 0))),
                "mean_score_contrast": _to_float(r.get("mean_score_contrast", "")),
                "score_ci_low": _to_float(r.get("mean_score_contrast_ci_low", "")),
                "score_ci_high": _to_float(r.get("mean_score_contrast_ci_high", "")),
                "p_score": _to_float(r.get("paired_p_score_sign", "")),
                "q_score": _to_float(r.get("q_score_sign", "")),
                "mean_margin_contrast": _to_float(r.get("mean_margin_contrast", "")),
                "margin_ci_low": _to_float(r.get("mean_margin_contrast_ci_low", "")),
                "margin_ci_high": _to_float(r.get("mean_margin_contrast_ci_high", "")),
                "mde_score_approx": _to_float(d_strata[d_strata["stratum"] == "all"]["mde_score_approx"].iloc[0])
                if not d_strata[d_strata["stratum"] == "all"].empty
                else float("nan"),
                "power_vs_sesoi": d_strata[d_strata["stratum"] == "all"]["power_vs_sesoi"].iloc[0]
                if not d_strata[d_strata["stratum"] == "all"].empty
                else "",
            }
        )

        if comp_path.exists():
            d_comp = pd.read_csv(comp_path)
            for _, rr in d_comp.iterrows():
                composition_rows.append(
                    {
                        "model": model_name,
                        "model_label": label,
                        "row_type": rr.get("row_type", ""),
                        "source": rr.get("source", ""),
                        "span_len": rr.get("span_len", ""),
                        "n_pairs": _to_float(rr.get("n_pairs", 0)),
                        "balance_mode": rr.get("balance_mode", deps.get("balance_mode", "")),
                    }
                )

        if exp16_dir_raw:
            exp16_dir = Path(str(exp16_dir_raw))
            p16 = exp16_dir / "tables" / "asymmetry_pair_deltas.csv"
            if p16.exists():
                d16 = pd.read_csv(p16)
                m = d16[["pair_id", "primary_score_contrast"]].merge(
                    d_pair[["pair_id", "primary_score_contrast"]],
                    on="pair_id",
                    suffixes=("_single", "_span"),
                    how="inner",
                )
                if not m.empty:
                    single = pd.to_numeric(m["primary_score_contrast_single"], errors="coerce").to_numpy(dtype=float)
                    span = pd.to_numeric(m["primary_score_contrast_span"], errors="coerce").to_numpy(dtype=float)
                    diff = span - single
                    ci_lo, ci_hi = _bootstrap_ci(diff, n_boot=args.bootstrap_n, seed=args.seed)
                    interaction_rows.append(
                        {
                            "model": model_name,
                            "model_label": label,
                            "n_overlap_pairs": int(len(m)),
                            "single_mean": float(np.nanmean(single)),
                            "span_mean": float(np.nanmean(span)),
                            "span_minus_single": float(np.nanmean(diff)),
                            "span_minus_single_ci_low": ci_lo,
                            "span_minus_single_ci_high": ci_hi,
                        }
                    )

    summary_path = out_dir / "exp28_balanced_summary.csv"
    comp_path_out = out_dir / "exp28_balanced_source_span_counts.csv"
    interaction_path = out_dir / "exp28_balanced_format_interaction.csv"
    meta_path = out_dir / "exp28_balanced_summary_meta.json"

    pd.DataFrame(
        summary_rows,
        columns=[
            "model",
            "model_label",
            "exp28_run_dir",
            "balance_mode",
            "source_quotas",
            "n_pairs",
            "mean_score_contrast",
            "score_ci_low",
            "score_ci_high",
            "p_score",
            "q_score",
            "mean_margin_contrast",
            "margin_ci_low",
            "margin_ci_high",
            "mde_score_approx",
            "power_vs_sesoi",
        ],
    ).to_csv(summary_path, index=False)

    pd.DataFrame(
        composition_rows,
        columns=["model", "model_label", "row_type", "source", "span_len", "n_pairs", "balance_mode"],
    ).to_csv(comp_path_out, index=False)

    pd.DataFrame(
        interaction_rows,
        columns=[
            "model",
            "model_label",
            "n_overlap_pairs",
            "single_mean",
            "span_mean",
            "span_minus_single",
            "span_minus_single_ci_low",
            "span_minus_single_ci_high",
        ],
    ).to_csv(interaction_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "summary": str(summary_path),
        "composition": str(comp_path_out),
        "interaction": str(interaction_path),
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(summary_path)
    print(comp_path_out)
    print(interaction_path)
    print(meta_path)


if __name__ == "__main__":
    main()
