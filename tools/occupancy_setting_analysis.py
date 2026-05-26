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


def _label(model_name: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model_name, model_name)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan"), float("nan")
    r = spearmanr(x[mask], y[mask], nan_policy="omit")
    return float(r.correlation), float(r.pvalue)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare prediction-position direction occupancy between Exp16 (cross-position) and "
            "Exp26 (same-position AR), and test occupancy-to-effect coupling."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--base-kind", choices=["stereo", "anti", "all"], default="stereo")
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

    setting_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)

        exp16_dir_raw = payload.get("exp16_canonical_run_dir", "")
        exp26_dir_raw = payload.get("exp26_run_dir", "")
        if not exp16_dir_raw or not exp26_dir_raw:
            missing.append(f"{model_name}:missing_exp16_or_exp26")
            continue

        p16 = Path(str(exp16_dir_raw)) / "tables" / "asymmetry_occupancy_pair_summary.csv"
        p26 = Path(str(exp26_dir_raw)) / "tables" / "ar_same_position_occupancy_pair_summary.csv"
        if not p16.exists() or not p26.exists():
            missing.append(f"{model_name}:missing_occupancy_outputs")
            continue

        d16 = pd.read_csv(p16)
        d26 = pd.read_csv(p26)
        if args.base_kind != "all":
            d16 = d16[d16["base_kind"] == args.base_kind].copy()
            d26 = d26[d26["base_kind"] == args.base_kind].copy()

        for dset_name, df in [("exp16_cross", d16), ("exp26_same", d26)]:
            if df.empty:
                continue
            true_abs = pd.to_numeric(df.get("mean_abs_true_proj"), errors="coerce").to_numpy(dtype=float)
            rand_abs = pd.to_numeric(df.get("mean_abs_random_proj"), errors="coerce").to_numpy(dtype=float)
            remove_margin = pd.to_numeric(df.get("remove_on_stereo_margin_delta"), errors="coerce").to_numpy(dtype=float)
            remove_score = pd.to_numeric(df.get("remove_on_stereo_score_delta"), errors="coerce").to_numpy(dtype=float)

            mask = np.isfinite(true_abs) & np.isfinite(rand_abs)
            if int(np.sum(mask)) == 0:
                continue
            true_abs_m = true_abs[mask]
            rand_abs_m = rand_abs[mask]
            remove_margin_m = remove_margin[mask] if remove_margin.shape[0] == mask.shape[0] else np.array([], dtype=float)
            remove_score_m = remove_score[mask] if remove_score.shape[0] == mask.shape[0] else np.array([], dtype=float)

            r1, p1 = _safe_spearman(true_abs_m, np.abs(remove_margin_m)) if remove_margin_m.size else (float("nan"), float("nan"))
            r2, p2 = _safe_spearman(true_abs_m, np.abs(remove_score_m)) if remove_score_m.size else (float("nan"), float("nan"))

            coupling_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "setting": dset_name,
                    "base_kind": args.base_kind,
                    "n_pairs": int(true_abs_m.size),
                    "mean_abs_true_proj": float(np.mean(true_abs_m)),
                    "mean_abs_random_proj": float(np.mean(rand_abs_m)),
                    "mean_abs_true_minus_random": float(np.mean(true_abs_m) - np.mean(rand_abs_m)),
                    "spearman_abs_true_vs_abs_remove_margin": r1,
                    "spearman_abs_true_vs_abs_remove_margin_p": p1,
                    "spearman_abs_true_vs_abs_remove_score": r2,
                    "spearman_abs_true_vs_abs_remove_score_p": p2,
                }
            )

        # Paired setting difference on overlapping pair ids.
        if d16.empty or d26.empty:
            continue
        m = d16[["pair_id", "mean_abs_true_proj", "mean_abs_random_proj"]].merge(
            d26[["pair_id", "mean_abs_true_proj", "mean_abs_random_proj"]],
            on="pair_id",
            suffixes=("_exp16", "_exp26"),
            how="inner",
        )
        if m.empty:
            missing.append(f"{model_name}:no_overlap_pairs")
            continue
        t16 = pd.to_numeric(m["mean_abs_true_proj_exp16"], errors="coerce").to_numpy(dtype=float)
        t26 = pd.to_numeric(m["mean_abs_true_proj_exp26"], errors="coerce").to_numpy(dtype=float)
        r16 = pd.to_numeric(m["mean_abs_random_proj_exp16"], errors="coerce").to_numpy(dtype=float)
        r26 = pd.to_numeric(m["mean_abs_random_proj_exp26"], errors="coerce").to_numpy(dtype=float)
        dt = t26 - t16
        dr = r26 - r16
        setting_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "base_kind": args.base_kind,
                "n_overlap_pairs": int(len(m)),
                "mean_abs_true_proj_exp16": float(np.nanmean(t16)),
                "mean_abs_true_proj_exp26": float(np.nanmean(t26)),
                "mean_abs_true_proj_same_minus_cross": float(np.nanmean(dt)),
                "mean_abs_random_proj_exp16": float(np.nanmean(r16)),
                "mean_abs_random_proj_exp26": float(np.nanmean(r26)),
                "mean_abs_random_proj_same_minus_cross": float(np.nanmean(dr)),
            }
        )

    setting_path = out_dir / "occupancy_setting_comparison.csv"
    coupling_path = out_dir / "occupancy_effect_coupling.csv"
    meta_path = out_dir / "occupancy_setting_analysis_meta.json"

    pd.DataFrame(
        setting_rows,
        columns=[
            "model",
            "model_label",
            "base_kind",
            "n_overlap_pairs",
            "mean_abs_true_proj_exp16",
            "mean_abs_true_proj_exp26",
            "mean_abs_true_proj_same_minus_cross",
            "mean_abs_random_proj_exp16",
            "mean_abs_random_proj_exp26",
            "mean_abs_random_proj_same_minus_cross",
        ],
    ).to_csv(setting_path, index=False)

    pd.DataFrame(
        coupling_rows,
        columns=[
            "model",
            "model_label",
            "setting",
            "base_kind",
            "n_pairs",
            "mean_abs_true_proj",
            "mean_abs_random_proj",
            "mean_abs_true_minus_random",
            "spearman_abs_true_vs_abs_remove_margin",
            "spearman_abs_true_vs_abs_remove_margin_p",
            "spearman_abs_true_vs_abs_remove_score",
            "spearman_abs_true_vs_abs_remove_score_p",
        ],
    ).to_csv(coupling_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "setting_comparison": str(setting_path),
        "coupling": str(coupling_path),
        "base_kind": args.base_kind,
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(setting_path)
    print(coupling_path)
    print(meta_path)


if __name__ == "__main__":
    main()
