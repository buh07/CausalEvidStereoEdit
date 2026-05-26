#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate fixed-seed Exp16 core asymmetry contrasts for Core-3 models."
    )
    p.add_argument("--run-map", required=True, help="JSON run map produced by may_arr tmux orchestrator.")
    p.add_argument("--output-dir", default="", help="Directory for output CSV/JSON files.")
    p.add_argument("--contrast", default="primary_inject_anti_minus_remove_stereo")
    p.add_argument("--require-complete", action="store_true", help="Fail if any model is missing configured seeds.")
    return p.parse_args()


def _to_float(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _load_primary_row(exp16_run_dir: Path, contrast: str) -> dict[str, Any]:
    path = exp16_run_dir / "tables" / "asymmetry_contrast.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing asymmetry contrast table: {path}")
    df = pd.read_csv(path)
    row = df[df["contrast"] == contrast]
    if row.empty:
        raise ValueError(f"Contrast '{contrast}' not found in {path}")
    return row.iloc[0].to_dict()


def _model_label(model: str, fallback_label: str | None = None) -> str:
    if fallback_label:
        return fallback_label
    mapping = {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }
    return mapping.get(model, model)


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    if not run_map_path.exists():
        raise FileNotFoundError(f"Run map not found: {run_map_path}")

    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map must contain a non-empty 'models' object.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload in sorted(models.items()):
        if not isinstance(payload, dict):
            continue
        label = _model_label(model_name, payload.get("label"))
        seed_runs = payload.get("exp16_seed_runs", {})
        if not isinstance(seed_runs, dict):
            seed_runs = {}

        for seed_str, run_dir_raw in sorted(seed_runs.items(), key=lambda x: int(str(x[0]))):
            if not run_dir_raw:
                missing.append(f"{model_name}:seed={seed_str}")
                continue
            run_dir = Path(str(run_dir_raw))
            try:
                row = _load_primary_row(run_dir, contrast=args.contrast)
            except Exception as exc:
                missing.append(f"{model_name}:seed={seed_str}:{exc}")
                continue
            seed_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "seed": int(seed_str),
                    "exp16_run_dir": str(run_dir),
                    "contrast": args.contrast,
                    "n_pairs": int(_to_float(row.get("n_pairs", 0))) if np.isfinite(_to_float(row.get("n_pairs", 0))) else 0,
                    "score_contrast": _to_float(row.get("mean_score_contrast", "")),
                    "score_contrast_ci_low": _to_float(row.get("mean_score_contrast_ci_low", "")),
                    "score_contrast_ci_high": _to_float(row.get("mean_score_contrast_ci_high", "")),
                    "score_p": _to_float(row.get("paired_p_score_sign", "")),
                    "score_q_within_table": _to_float(row.get("q_score_sign", "")),
                    "margin_contrast": _to_float(row.get("mean_margin_contrast", "")),
                    "margin_contrast_ci_low": _to_float(row.get("mean_margin_contrast_ci_low", "")),
                    "margin_contrast_ci_high": _to_float(row.get("mean_margin_contrast_ci_high", "")),
                    "margin_p": _to_float(row.get("paired_p_margin_wilcoxon", "")),
                    "margin_q_within_table": _to_float(row.get("q_margin_wilcoxon", "")),
                }
            )

    if args.require_complete and missing:
        raise RuntimeError("Missing seed runs:\n" + "\n".join(missing))

    seed_df = pd.DataFrame(seed_rows)
    seed_csv = out_dir / "seed_core_asymmetry_rows.csv"
    if seed_df.empty:
        seed_df = pd.DataFrame(
            columns=[
                "model",
                "model_label",
                "seed",
                "exp16_run_dir",
                "contrast",
                "n_pairs",
                "score_contrast",
                "score_contrast_ci_low",
                "score_contrast_ci_high",
                "score_p",
                "score_q_within_table",
                "margin_contrast",
                "margin_contrast_ci_low",
                "margin_contrast_ci_high",
                "margin_p",
                "margin_q_within_table",
            ]
        )
    seed_df.to_csv(seed_csv, index=False)

    agg_rows: list[dict[str, Any]] = []
    if not seed_df.empty:
        for model, group in seed_df.groupby("model", sort=True):
            s = pd.to_numeric(group["score_contrast"], errors="coerce")
            m = pd.to_numeric(group["margin_contrast"], errors="coerce")
            n_pairs = pd.to_numeric(group["n_pairs"], errors="coerce")
            agg_rows.append(
                {
                    "model": model,
                    "model_label": str(group["model_label"].iloc[0]),
                    "n_seeds": int(group["seed"].nunique()),
                    "seeds": ",".join(str(int(x)) for x in sorted(group["seed"].dropna().astype(int).tolist())),
                    "n_pairs_min": int(np.nanmin(n_pairs)) if np.isfinite(np.nanmin(n_pairs)) else 0,
                    "n_pairs_max": int(np.nanmax(n_pairs)) if np.isfinite(np.nanmax(n_pairs)) else 0,
                    "pooled_score_contrast_mean": float(np.nanmean(s)),
                    "between_seed_score_sd": float(np.nanstd(s, ddof=1)) if np.sum(np.isfinite(s)) > 1 else float("nan"),
                    "between_seed_score_var": float(np.nanvar(s, ddof=1)) if np.sum(np.isfinite(s)) > 1 else float("nan"),
                    "pooled_margin_contrast_mean": float(np.nanmean(m)),
                    "between_seed_margin_sd": float(np.nanstd(m, ddof=1)) if np.sum(np.isfinite(m)) > 1 else float("nan"),
                    "between_seed_margin_var": float(np.nanvar(m, ddof=1)) if np.sum(np.isfinite(m)) > 1 else float("nan"),
                }
            )

    agg_df = pd.DataFrame(agg_rows)
    agg_csv = out_dir / "seed_core_asymmetry_aggregate.csv"
    if agg_df.empty:
        agg_df = pd.DataFrame(
            columns=[
                "model",
                "model_label",
                "n_seeds",
                "seeds",
                "n_pairs_min",
                "n_pairs_max",
                "pooled_score_contrast_mean",
                "between_seed_score_sd",
                "between_seed_score_var",
                "pooled_margin_contrast_mean",
                "between_seed_margin_sd",
                "between_seed_margin_var",
            ]
        )
    agg_df.to_csv(agg_csv, index=False)

    meta = {
        "run_map": str(run_map_path),
        "output_dir": str(out_dir),
        "contrast": args.contrast,
        "n_seed_rows": int(len(seed_rows)),
        "n_models": int(agg_df["model"].nunique()) if not agg_df.empty else 0,
        "missing": missing,
    }
    meta_path = out_dir / "seed_core_asymmetry_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {seed_csv}")
    print(f"Wrote {agg_csv}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
