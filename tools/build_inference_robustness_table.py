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
        description=(
            "Recompute key Exp16 q-values under multiple correction families: "
            "within-table BH, across-model BH, and prereg Bonferroni."
        )
    )
    p.add_argument("--run-map", required=True, help="JSON run map from may_arr tmux orchestrator.")
    p.add_argument("--output-dir", default="", help="Output directory for robustness tables.")
    p.add_argument("--primary-contrast", default="primary_inject_anti_minus_remove_stereo")
    return p.parse_args()


def _to_float(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _bh(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    if m == 0:
        return []
    finite = [(i, float(p)) for i, p in enumerate(p_vals) if np.isfinite(p)]
    if not finite:
        return [float("nan")] * m
    finite.sort(key=lambda x: x[1])
    q = [float("nan")] * m
    k = len(finite)
    running = 1.0
    for rev_rank, (idx, p) in enumerate(reversed(finite), start=1):
        rank = k - rev_rank + 1
        val = min(running, (p * k) / rank)
        running = val
        q[idx] = max(0.0, min(1.0, float(val)))
    return q


def _bonf(p_vals: list[float], m_tests: int) -> list[float]:
    out: list[float] = []
    for p in p_vals:
        if not np.isfinite(p):
            out.append(float("nan"))
        else:
            out.append(float(min(1.0, max(0.0, p * m_tests))))
    return out


def _model_label(model: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model, model)


def _canonical_exp16_dir(payload: dict[str, Any]) -> Path | None:
    if payload.get("exp16_canonical_run_dir"):
        return Path(str(payload["exp16_canonical_run_dir"]))
    seed_runs = payload.get("exp16_seed_runs", {})
    if isinstance(seed_runs, dict):
        if "11" in seed_runs and seed_runs["11"]:
            return Path(str(seed_runs["11"]))
        for seed_key in sorted(seed_runs.keys(), key=lambda x: int(str(x))):
            val = seed_runs.get(seed_key)
            if val:
                return Path(str(val))
    return None


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map must include models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _model_label(model_name, payload)
        exp16_dir = _canonical_exp16_dir(payload)
        if exp16_dir is None:
            missing.append(f"{model_name}:missing_canonical_exp16")
            continue
        table_path = exp16_dir / "tables" / "asymmetry_contrast.csv"
        if not table_path.exists():
            missing.append(f"{model_name}:missing_table:{table_path}")
            continue
        df = pd.read_csv(table_path)
        if df.empty:
            missing.append(f"{model_name}:empty_table")
            continue

        for _, row in df.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "exp16_run_dir": str(exp16_dir),
                    "contrast": str(row.get("contrast", "")),
                    "n_pairs": int(_to_float(row.get("n_pairs", 0))) if np.isfinite(_to_float(row.get("n_pairs", 0))) else 0,
                    "p_score": _to_float(row.get("paired_p_score_sign", "")),
                    "q_score_within_table_bh": _to_float(row.get("q_score_sign", "")),
                    "p_margin": _to_float(row.get("paired_p_margin_wilcoxon", "")),
                    "q_margin_within_table_bh": _to_float(row.get("q_margin_wilcoxon", "")),
                    "mean_score_contrast": _to_float(row.get("mean_score_contrast", "")),
                    "mean_margin_contrast": _to_float(row.get("mean_margin_contrast", "")),
                }
            )

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        out = out_dir / "inference_robustness_core.csv"
        pd.DataFrame().to_csv(out, index=False)
        print(f"Wrote empty table: {out}")
        return

    # Across-model BH and Bonferroni are defined per contrast family (e.g., primary contrast across models).
    q_score_across: list[float] = [float("nan")] * len(df_all)
    q_margin_across: list[float] = [float("nan")] * len(df_all)
    q_score_bonf: list[float] = [float("nan")] * len(df_all)
    q_margin_bonf: list[float] = [float("nan")] * len(df_all)

    for contrast, group in df_all.groupby("contrast", sort=True):
        idxs = group.index.tolist()
        score_p = [float(df_all.loc[i, "p_score"]) for i in idxs]
        margin_p = [float(df_all.loc[i, "p_margin"]) for i in idxs]
        bh_s = _bh(score_p)
        bh_m = _bh(margin_p)
        bonf_s = _bonf(score_p, m_tests=max(1, len(score_p)))
        bonf_m = _bonf(margin_p, m_tests=max(1, len(margin_p)))
        for ii, i in enumerate(idxs):
            q_score_across[i] = bh_s[ii]
            q_margin_across[i] = bh_m[ii]
            q_score_bonf[i] = bonf_s[ii]
            q_margin_bonf[i] = bonf_m[ii]

    df_all["q_score_across_model_bh"] = q_score_across
    df_all["q_margin_across_model_bh"] = q_margin_across
    df_all["q_score_prereg_bonferroni"] = q_score_bonf
    df_all["q_margin_prereg_bonferroni"] = q_margin_bonf

    primary_df = df_all[df_all["contrast"] == args.primary_contrast].copy()
    primary_df = primary_df.sort_values("model_label")

    full_out = out_dir / "inference_robustness_all_contrasts.csv"
    primary_out = out_dir / "inference_robustness_core.csv"
    df_all.to_csv(full_out, index=False)
    primary_df.to_csv(primary_out, index=False)

    summary = {
        "run_map": str(run_map_path),
        "output_dir": str(out_dir),
        "primary_contrast": args.primary_contrast,
        "n_rows_all": int(len(df_all)),
        "n_rows_primary": int(len(primary_df)),
        "missing": missing,
        "policy_notes": {
            "within_table": "As reported in Exp16 rows (BH within-run contrast family).",
            "across_model_bh": "Recomputed BH across models within each named contrast family.",
            "prereg_bonferroni": "Recomputed Bonferroni across models within each named contrast family.",
        },
    }
    summary_path = out_dir / "inference_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {full_out}")
    print(f"Wrote {primary_out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
