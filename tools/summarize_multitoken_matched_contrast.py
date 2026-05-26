#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _to_float(v: Any) -> float:
    try:
        if v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate Exp28 multi-token matched inject/remove summaries from a run-map "
            "into manuscript-ready report tables."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--primary-contrast", default="primary_inject_anti_minus_remove_stereo")
    return p.parse_args()


def _model_label(model_name: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model_name, model_name)


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map is missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _model_label(model_name, payload)
        exp28_dir_raw = payload.get("exp28_run_dir", "")
        if not exp28_dir_raw:
            missing.append(f"{model_name}:missing_exp28_run_dir")
            continue
        exp28_dir = Path(str(exp28_dir_raw))

        contrast_path = exp28_dir / "tables" / "multitoken_asymmetry_contrast.csv"
        strata_path = exp28_dir / "tables" / "multitoken_matched_contrast_by_span.csv"
        if not contrast_path.exists():
            missing.append(f"{model_name}:missing:{contrast_path}")
            continue

        contrast_df = pd.read_csv(contrast_path)
        row_primary = contrast_df[contrast_df["contrast"] == args.primary_contrast]
        if row_primary.empty:
            missing.append(f"{model_name}:missing_primary_contrast")
            continue
        primary = row_primary.iloc[0]

        row: dict[str, Any] = {
            "model": model_name,
            "model_label": label,
            "exp28_run_dir": str(exp28_dir),
            "contrast": args.primary_contrast,
            "n_pairs": int(_to_float(primary.get("n_pairs", 0))),
            "mean_score_contrast": _to_float(primary.get("mean_score_contrast", "")),
            "score_ci_low": _to_float(primary.get("mean_score_contrast_ci_low", "")),
            "score_ci_high": _to_float(primary.get("mean_score_contrast_ci_high", "")),
            "p_score": _to_float(primary.get("paired_p_score_sign", "")),
            "q_score_within_table_bh": _to_float(primary.get("q_score_sign", "")),
            "mean_margin_contrast": _to_float(primary.get("mean_margin_contrast", "")),
            "margin_ci_low": _to_float(primary.get("mean_margin_contrast_ci_low", "")),
            "margin_ci_high": _to_float(primary.get("mean_margin_contrast_ci_high", "")),
            "p_margin": _to_float(primary.get("paired_p_margin_wilcoxon", "")),
            "q_margin_within_table_bh": _to_float(primary.get("q_margin_wilcoxon", "")),
        }

        if strata_path.exists():
            strata_df = pd.read_csv(strata_path)
            all_row = strata_df[strata_df["stratum"].astype(str) == "all"]
            if not all_row.empty:
                sr = all_row.iloc[0]
                row.update(
                    {
                        "sesoi": _to_float(sr.get("sesoi", "")),
                        "power_alpha": _to_float(sr.get("power_alpha", "")),
                        "target_power": _to_float(sr.get("target_power", "")),
                        "mde_score_approx": _to_float(sr.get("mde_score_approx", "")),
                        "power_vs_sesoi": str(sr.get("power_vs_sesoi", "")),
                    }
                )

        exp27_dir_raw = payload.get("exp27_run_dir", "")
        if exp27_dir_raw:
            exp27_path = Path(str(exp27_dir_raw)) / "tables" / "multitoken_span_summary.csv"
            if exp27_path.exists():
                exp27_df = pd.read_csv(exp27_path)
                if not exp27_df.empty:
                    r0 = exp27_df.iloc[0]
                    row["exp27_ablation_delta"] = _to_float(r0.get("stereotype_score_delta", ""))
                    row["exp27_ablation_q"] = _to_float(r0.get("q_score_sign", ""))

        rows.append(row)

    out_csv = out_dir / "multitoken_matched_contrast_summary.csv"
    columns = [
        "model",
        "model_label",
        "exp28_run_dir",
        "contrast",
        "n_pairs",
        "mean_score_contrast",
        "score_ci_low",
        "score_ci_high",
        "p_score",
        "q_score_within_table_bh",
        "mean_margin_contrast",
        "margin_ci_low",
        "margin_ci_high",
        "p_margin",
        "q_margin_within_table_bh",
        "sesoi",
        "power_alpha",
        "target_power",
        "mde_score_approx",
        "power_vs_sesoi",
        "exp27_ablation_delta",
        "exp27_ablation_q",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)

    meta = {
        "run_map": str(run_map_path),
        "output": str(out_csv),
        "n_models": len(rows),
        "missing": missing,
        "primary_contrast": args.primary_contrast,
    }
    meta_path = out_dir / "multitoken_matched_contrast_summary_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(out_csv)
    print(meta_path)


if __name__ == "__main__":
    main()
