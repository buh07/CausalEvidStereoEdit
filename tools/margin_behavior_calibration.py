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
            "Calibrate margin movement to behavioral flips: flip probability by |margin shift| bins "
            "for remove-on-stereo ablation."
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
    model_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    bins = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf")]
    bin_labels = ["[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,inf)"]

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)

        exp16_dir_raw = payload.get("exp16_canonical_run_dir", "")
        if not exp16_dir_raw:
            missing.append(f"{model_name}:missing_exp16")
            continue
        pair_path = Path(str(exp16_dir_raw)) / "tables" / "asymmetry_pair_deltas.csv"
        if not pair_path.exists():
            missing.append(f"{model_name}:missing_pair_table")
            continue

        df = pd.read_csv(pair_path)
        margin_delta = pd.to_numeric(df.get("remove_on_stereo_margin_delta"), errors="coerce").to_numpy(dtype=float)
        score_delta = pd.to_numeric(df.get("remove_on_stereo_score_delta"), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(margin_delta) & np.isfinite(score_delta)
        md = margin_delta[valid]
        sd = score_delta[valid]
        if md.size == 0:
            missing.append(f"{model_name}:no_valid_rows")
            continue

        abs_md = np.abs(md)
        flipped = (sd != 0).astype(float)
        idx = np.digitize(abs_md, bins, right=False) - 1

        for bi, lab in enumerate(bin_labels):
            mask = idx == bi
            if not np.any(mask):
                continue
            rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "abs_margin_shift_bin": lab,
                    "n_pairs": int(np.sum(mask)),
                    "mean_abs_margin_shift": float(np.mean(abs_md[mask])),
                    "flip_rate": float(np.mean(flipped[mask])),
                    "mean_signed_margin_shift": float(np.mean(md[mask])),
                }
            )

        model_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "n_pairs": int(md.size),
                "overall_flip_rate": float(np.mean(flipped)),
                "overall_mean_abs_margin_shift": float(np.mean(abs_md)),
            }
        )

    rows_path = out_dir / "margin_behavior_calibration_bins.csv"
    model_path = out_dir / "margin_behavior_calibration_model_summary.csv"
    meta_path = out_dir / "margin_behavior_calibration_meta.json"

    pd.DataFrame(
        rows,
        columns=[
            "model",
            "model_label",
            "abs_margin_shift_bin",
            "n_pairs",
            "mean_abs_margin_shift",
            "flip_rate",
            "mean_signed_margin_shift",
        ],
    ).to_csv(rows_path, index=False)

    pd.DataFrame(
        model_rows,
        columns=["model", "model_label", "n_pairs", "overall_flip_rate", "overall_mean_abs_margin_shift"],
    ).to_csv(model_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "bins": bins,
        "bin_rows": str(rows_path),
        "model_rows": str(model_path),
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(rows_path)
    print(model_path)
    print(meta_path)


if __name__ == "__main__":
    main()
