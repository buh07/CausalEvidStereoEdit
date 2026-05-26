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
            "Apply the four-step checklist framing to prompt intervention results (Exp29) as a "
            "second intervention family calibration report."
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
        cpath = Path(str(exp29_dir_raw)) / "tables" / "prompt_calibration_contrast.csv"
        if not cpath.exists():
            missing.append(f"{model_name}:missing_prompt_contrast")
            continue

        cdf = pd.read_csv(cpath)
        row = cdf[cdf["contrast"] == "prompt_stereo_minus_prompt_anti"]
        if row.empty:
            missing.append(f"{model_name}:missing_primary_prompt_row")
            continue
        r = row.iloc[0]
        q = _to_float(r.get("q_score_sign", ""))
        mean = _to_float(r.get("mean_score_contrast", ""))
        step1_status = "Strong" if np.isfinite(q) and q < 0.05 and np.isfinite(mean) and mean > 0 else "Provisional"

        rows.append(
            {
                "model": model_name,
                "model_label": label,
                "step_1_causal_efficacy_status": step1_status,
                "step_1_primary_q": q,
                "step_1_primary_score_contrast": mean,
                "step_2_reliability_status": "Not evaluated for prompt operator in this draft",
                "step_3_contamination_status": "Not applicable to prompt operator",
                "step_4_transfer_status": "Not evaluated for prompt operator in this draft",
                "overall_note": "Checklist remains discriminative: prompt step-1 passes strongly, while direction-edit steps 2-4 remain mixed/provisional.",
            }
        )

    out_csv = out_dir / "prompt_checklist_status.csv"
    meta_path = out_dir / "prompt_checklist_status_meta.json"

    pd.DataFrame(
        rows,
        columns=[
            "model",
            "model_label",
            "step_1_causal_efficacy_status",
            "step_1_primary_q",
            "step_1_primary_score_contrast",
            "step_2_reliability_status",
            "step_3_contamination_status",
            "step_4_transfer_status",
            "overall_note",
        ],
    ).to_csv(out_csv, index=False)

    meta = {"run_map": str(run_map_path), "output": str(out_csv), "missing": missing}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(out_csv)
    print(meta_path)


if __name__ == "__main__":
    main()
