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


def _label(model_name: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model_name, model_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Exp29 prompt matched contrasts from run-map.")
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

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)
        exp29_dir_raw = payload.get("exp29_run_dir", "")
        if not exp29_dir_raw:
            missing.append(f"{model_name}:missing_exp29")
            continue
        exp29_dir = Path(str(exp29_dir_raw))
        cpath = exp29_dir / "tables" / "prompt_calibration_contrast.csv"
        if not cpath.exists():
            missing.append(f"{model_name}:missing_prompt_calibration_contrast")
            continue
        cdf = pd.read_csv(cpath)
        row = cdf[cdf["contrast"] == "prompt_stereo_minus_prompt_anti"]
        if row.empty:
            missing.append(f"{model_name}:missing_primary_prompt_contrast")
            continue
        r0 = row.iloc[0]
        rows.append(
            {
                "model": model_name,
                "model_label": label,
                "exp29_run_dir": str(exp29_dir),
                "n_pairs": int(_to_float(r0.get("n_pairs", 0))),
                "mean_score_contrast": _to_float(r0.get("mean_score_contrast", "")),
                "score_ci_low": _to_float(r0.get("mean_score_contrast_ci_low", "")),
                "score_ci_high": _to_float(r0.get("mean_score_contrast_ci_high", "")),
                "p_score_sign": _to_float(r0.get("paired_p_score_sign", "")),
                "q_score_within_table_bh": _to_float(r0.get("q_score_sign", "")),
                "mean_margin_contrast": _to_float(r0.get("mean_margin_contrast", "")),
                "margin_ci_low": _to_float(r0.get("mean_margin_contrast_ci_low", "")),
                "margin_ci_high": _to_float(r0.get("mean_margin_contrast_ci_high", "")),
                "p_margin_wilcoxon": _to_float(r0.get("paired_p_margin_wilcoxon", "")),
                "q_margin_within_table_bh": _to_float(r0.get("q_margin_wilcoxon", "")),
            }
        )

    out_csv = out_dir / "prompt_matched_contrast_summary.csv"
    pd.DataFrame(
        rows,
        columns=[
            "model",
            "model_label",
            "exp29_run_dir",
            "n_pairs",
            "mean_score_contrast",
            "score_ci_low",
            "score_ci_high",
            "p_score_sign",
            "q_score_within_table_bh",
            "mean_margin_contrast",
            "margin_ci_low",
            "margin_ci_high",
            "p_margin_wilcoxon",
            "q_margin_within_table_bh",
        ],
    ).to_csv(out_csv, index=False)

    meta = {
        "run_map": str(run_map_path),
        "output": str(out_csv),
        "missing": missing,
    }
    meta_path = out_dir / "prompt_matched_contrast_summary_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(out_csv)
    print(meta_path)


if __name__ == "__main__":
    main()
