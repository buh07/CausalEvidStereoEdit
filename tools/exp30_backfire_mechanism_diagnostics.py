#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import load_directions_npz


def _label(model_name: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model_name, model_name)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exp30 mechanism diagnostics: axis-level decomposition of transfer/backfire rows and "
            "source-direction cosine/sign alignment between StereoSet and CrowS direction pools."
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

    axis_rows: list[dict[str, Any]] = []
    align_rows: list[dict[str, Any]] = []
    align_axis_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)

        exp30_dir_raw = payload.get("exp30_run_dir", "")
        exp1_ss_raw = payload.get("exp01_stereoset_run_dir", "")
        exp1_cr_raw = payload.get("exp01_crows_run_dir", "")
        if not exp30_dir_raw:
            missing.append(f"{model_name}:missing_exp30")
            continue

        exp30_dir = Path(str(exp30_dir_raw))
        axis_path = exp30_dir / "tables" / "cross_dataset_injection_transfer_axis_summary.csv"
        pair_path = exp30_dir / "tables" / "cross_dataset_injection_transfer_pairs.csv"
        if axis_path.exists():
            dax = pd.read_csv(axis_path)
        elif pair_path.exists():
            dp = pd.read_csv(pair_path)
            dax = (
                dp.groupby(["condition", "rank_source", "target_source", "axis"], as_index=False)
                .agg(n_pairs=("pair_id", "count"), stereotype_score_delta=("score_delta", "mean"), mean_margin_delta=("margin_delta", "mean"))
            )
        else:
            missing.append(f"{model_name}:missing_exp30_axis_or_pair_tables")
            continue

        for _, r in dax.iterrows():
            axis_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "condition": r.get("condition", ""),
                    "rank_source": r.get("rank_source", ""),
                    "target_source": r.get("target_source", ""),
                    "axis": r.get("axis", ""),
                    "n_pairs": float(r.get("n_pairs", np.nan)),
                    "stereotype_score_delta": float(r.get("stereotype_score_delta", np.nan)),
                    "mean_margin_delta": float(r.get("mean_margin_delta", np.nan)),
                    "q_score_sign": float(r.get("q_score_sign", np.nan)) if "q_score_sign" in dax.columns else float("nan"),
                }
            )

        if not exp1_ss_raw or not exp1_cr_raw:
            missing.append(f"{model_name}:missing_exp01_source_dirs")
            continue
        dss_path = Path(str(exp1_ss_raw)) / "artifacts" / "directions_layerwise.npz"
        dcr_path = Path(str(exp1_cr_raw)) / "artifacts" / "directions_layerwise.npz"
        if not dss_path.exists() or not dcr_path.exists():
            missing.append(f"{model_name}:missing_source_direction_npz")
            continue

        dss = load_directions_npz(dss_path)
        dcr = load_directions_npz(dcr_path)
        common = sorted(set(dss) & set(dcr))
        cos_vals: list[float] = []
        axis_to_vals: dict[str, list[float]] = {}
        for key in common:
            c = _cos(dss[key], dcr[key])
            if not np.isfinite(c):
                continue
            cos_vals.append(c)
            axis_to_vals.setdefault(key[0], []).append(c)

        if not cos_vals:
            continue
        cos_arr = np.array(cos_vals, dtype=float)
        align_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "n_shared_axis_layers": int(cos_arr.size),
                "mean_cosine_stereoset_vs_crows": float(np.mean(cos_arr)),
                "median_cosine_stereoset_vs_crows": float(np.median(cos_arr)),
                "neg_cosine_fraction": float(np.mean((cos_arr < 0).astype(float))),
            }
        )
        for axis, vals in sorted(axis_to_vals.items()):
            arr = np.array(vals, dtype=float)
            align_axis_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "axis": axis,
                    "n_shared_layers": int(arr.size),
                    "mean_cosine": float(np.mean(arr)),
                    "median_cosine": float(np.median(arr)),
                    "neg_cosine_fraction": float(np.mean((arr < 0).astype(float))),
                }
            )

    axis_out = out_dir / "exp30_backfire_axis_decomposition.csv"
    align_out = out_dir / "exp30_direction_alignment_summary.csv"
    align_axis_out = out_dir / "exp30_direction_alignment_by_axis.csv"
    meta_out = out_dir / "exp30_backfire_mechanism_meta.json"

    pd.DataFrame(
        axis_rows,
        columns=[
            "model",
            "model_label",
            "condition",
            "rank_source",
            "target_source",
            "axis",
            "n_pairs",
            "stereotype_score_delta",
            "mean_margin_delta",
            "q_score_sign",
        ],
    ).to_csv(axis_out, index=False)

    pd.DataFrame(
        align_rows,
        columns=[
            "model",
            "model_label",
            "n_shared_axis_layers",
            "mean_cosine_stereoset_vs_crows",
            "median_cosine_stereoset_vs_crows",
            "neg_cosine_fraction",
        ],
    ).to_csv(align_out, index=False)

    pd.DataFrame(
        align_axis_rows,
        columns=[
            "model",
            "model_label",
            "axis",
            "n_shared_layers",
            "mean_cosine",
            "median_cosine",
            "neg_cosine_fraction",
        ],
    ).to_csv(align_axis_out, index=False)

    meta = {
        "run_map": str(run_map_path),
        "axis_decomposition": str(axis_out),
        "direction_alignment": str(align_out),
        "direction_alignment_by_axis": str(align_axis_out),
        "missing": missing,
    }
    meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(axis_out)
    print(align_out)
    print(align_axis_out)
    print(meta_out)


if __name__ == "__main__":
    main()
