#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
            "Summarize score-margin dissociation for Exp16 ablation: boundary crossings, "
            "baseline-margin distribution, and heterogeneity by |margin|."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def _label(model_name: str, payload: dict[str, Any]) -> str:
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
        raise ValueError("Run map missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    bins = [0.0, 0.25, 0.5, 1.0, 2.0, float("inf")]
    bin_labels = ["[0,0.25)", "[0.25,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,inf)"]

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
        needed = ["baseline_margin_stereo", "remove_on_stereo_margin_delta"]
        if any(c not in df.columns for c in needed):
            missing.append(f"{model_name}:missing_required_columns")
            continue

        base = pd.to_numeric(df["baseline_margin_stereo"], errors="coerce").to_numpy(dtype=float)
        delta = pd.to_numeric(df["remove_on_stereo_margin_delta"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(base) & np.isfinite(delta)
        base = base[ok]
        delta = delta[ok]
        if base.size == 0:
            missing.append(f"{model_name}:no_valid_rows")
            continue

        edited = base + delta
        abs_base = np.abs(base)

        cross_down = (base > 0) & (edited <= 0)
        cross_up = (base <= 0) & (edited > 0)
        any_cross = cross_down | cross_up

        near_boundary = abs_base <= 0.25
        moderate_boundary = (abs_base > 0.25) & (abs_base <= 1.0)
        far_boundary = abs_base > 1.0

        rho_abs_delta = spearmanr(abs_base, np.abs(delta), nan_policy="omit")
        rho_abs_cross = spearmanr(abs_base, any_cross.astype(float), nan_policy="omit")

        crossers = any_cross
        non_crossers = ~any_cross
        mean_abs_delta_cross = float(np.mean(np.abs(delta[crossers]))) if np.any(crossers) else float("nan")
        mean_abs_delta_non = float(np.mean(np.abs(delta[non_crossers]))) if np.any(non_crossers) else float("nan")
        mean_delta_cross = float(np.mean(delta[crossers])) if np.any(crossers) else float("nan")
        mean_delta_non = float(np.mean(delta[non_crossers])) if np.any(non_crossers) else float("nan")
        post_margin_cross_mean = float(np.mean(edited[crossers])) if np.any(crossers) else float("nan")
        post_margin_non_mean = float(np.mean(edited[non_crossers])) if np.any(non_crossers) else float("nan")

        # Permutation test: are crossing |delta| magnitudes larger than non-crossing?
        perm_p = float("nan")
        if np.any(crossers) and np.any(non_crossers):
            rng = np.random.default_rng(17)
            obs = mean_abs_delta_cross - mean_abs_delta_non
            labels = crossers.astype(int).copy()
            vals = np.abs(delta).copy()
            reps = 5000
            hits = 0
            for _ in range(reps):
                rng.shuffle(labels)
                g1 = vals[labels == 1]
                g0 = vals[labels == 0]
                if g1.size == 0 or g0.size == 0:
                    continue
                stat = float(np.mean(g1) - np.mean(g0))
                if abs(stat) >= abs(obs):
                    hits += 1
            perm_p = float((hits + 1) / (reps + 1))

        summary_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "exp16_run_dir": str(exp16_dir_raw),
                "n_pairs": int(base.size),
                "mean_margin_delta_remove_on_stereo": float(np.mean(delta)),
                "median_abs_baseline_margin": float(np.median(abs_base)),
                "crossings_down_count": int(np.sum(cross_down)),
                "crossings_up_count": int(np.sum(cross_up)),
                "crossings_any_count": int(np.sum(any_cross)),
                "crossings_any_rate": float(np.mean(any_cross.astype(float))),
                "near_boundary_count_abs_le_0p25": int(np.sum(near_boundary)),
                "near_boundary_rate_abs_le_0p25": float(np.mean(near_boundary.astype(float))),
                "moderate_boundary_count_0p25_to_1": int(np.sum(moderate_boundary)),
                "far_boundary_count_gt_1": int(np.sum(far_boundary)),
                "spearman_abs_baseline_vs_abs_delta": float(rho_abs_delta.correlation)
                if np.isfinite(rho_abs_delta.correlation)
                else float("nan"),
                "spearman_abs_baseline_vs_crossing": float(rho_abs_cross.correlation)
                if np.isfinite(rho_abs_cross.correlation)
                else float("nan"),
                "spearman_abs_baseline_vs_abs_delta_p": float(rho_abs_delta.pvalue)
                if np.isfinite(rho_abs_delta.pvalue)
                else float("nan"),
                "spearman_abs_baseline_vs_crossing_p": float(rho_abs_cross.pvalue)
                if np.isfinite(rho_abs_cross.pvalue)
                else float("nan"),
                "mean_abs_delta_crossers": mean_abs_delta_cross,
                "mean_abs_delta_noncrossers": mean_abs_delta_non,
                "mean_delta_crossers": mean_delta_cross,
                "mean_delta_noncrossers": mean_delta_non,
                "mean_post_margin_crossers": post_margin_cross_mean,
                "mean_post_margin_noncrossers": post_margin_non_mean,
                "perm_p_abs_delta_crossers_vs_noncrossers": perm_p,
            }
        )

        quality_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "n_crossers": int(np.sum(crossers)),
                "n_noncrossers": int(np.sum(non_crossers)),
                "mean_abs_delta_crossers": mean_abs_delta_cross,
                "mean_abs_delta_noncrossers": mean_abs_delta_non,
                "mean_delta_crossers": mean_delta_cross,
                "mean_delta_noncrossers": mean_delta_non,
                "mean_post_margin_crossers": post_margin_cross_mean,
                "mean_post_margin_noncrossers": post_margin_non_mean,
                "perm_p_abs_delta_crossers_vs_noncrossers": perm_p,
            }
        )

        idx = np.digitize(abs_base, bins, right=False) - 1
        for bi, label_bin in enumerate(bin_labels):
            mask = idx == bi
            if not np.any(mask):
                continue
            bin_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "abs_margin_bin": label_bin,
                    "n_pairs": int(np.sum(mask)),
                    "mean_abs_baseline_margin": float(np.mean(abs_base[mask])),
                    "mean_margin_delta": float(np.mean(delta[mask])),
                    "cross_down_rate": float(np.mean(cross_down[mask].astype(float))),
                    "cross_any_rate": float(np.mean(any_cross[mask].astype(float))),
                }
            )

    summary_path = out_dir / "boundary_crossing_summary.csv"
    bins_path = out_dir / "boundary_crossing_bins.csv"
    quality_path = out_dir / "boundary_crossing_quality.csv"
    summary_cols = [
        "model",
        "model_label",
        "exp16_run_dir",
        "n_pairs",
        "mean_margin_delta_remove_on_stereo",
        "median_abs_baseline_margin",
        "crossings_down_count",
        "crossings_up_count",
        "crossings_any_count",
        "crossings_any_rate",
        "near_boundary_count_abs_le_0p25",
        "near_boundary_rate_abs_le_0p25",
        "moderate_boundary_count_0p25_to_1",
        "far_boundary_count_gt_1",
        "spearman_abs_baseline_vs_abs_delta",
        "spearman_abs_baseline_vs_crossing",
        "spearman_abs_baseline_vs_abs_delta_p",
        "spearman_abs_baseline_vs_crossing_p",
        "mean_abs_delta_crossers",
        "mean_abs_delta_noncrossers",
        "mean_delta_crossers",
        "mean_delta_noncrossers",
        "mean_post_margin_crossers",
        "mean_post_margin_noncrossers",
        "perm_p_abs_delta_crossers_vs_noncrossers",
    ]
    bin_cols = [
        "model",
        "model_label",
        "abs_margin_bin",
        "n_pairs",
        "mean_abs_baseline_margin",
        "mean_margin_delta",
        "cross_down_rate",
        "cross_any_rate",
    ]
    pd.DataFrame(summary_rows, columns=summary_cols).to_csv(summary_path, index=False)
    pd.DataFrame(bin_rows, columns=bin_cols).to_csv(bins_path, index=False)
    pd.DataFrame(
        quality_rows,
        columns=[
            "model",
            "model_label",
            "n_crossers",
            "n_noncrossers",
            "mean_abs_delta_crossers",
            "mean_abs_delta_noncrossers",
            "mean_delta_crossers",
            "mean_delta_noncrossers",
            "mean_post_margin_crossers",
            "mean_post_margin_noncrossers",
            "perm_p_abs_delta_crossers_vs_noncrossers",
        ],
    ).to_csv(quality_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "summary": str(summary_path),
        "bins": str(bins_path),
        "quality": str(quality_path),
        "missing": missing,
        "definition": {
            "cross_down": "baseline margin > 0 and edited margin <= 0 under remove_on_stereo",
            "cross_up": "baseline margin <= 0 and edited margin > 0 under remove_on_stereo",
        },
    }
    meta_path = out_dir / "boundary_crossing_summary_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(summary_path)
    print(bins_path)
    print(quality_path)
    print(meta_path)


if __name__ == "__main__":
    main()
