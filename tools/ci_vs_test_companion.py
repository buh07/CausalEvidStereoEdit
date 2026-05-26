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


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False
    return (lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Companion report comparing bootstrap-mean CI calls vs paired sign-test q-value calls "
            "on transfer rows, with a compact worked-example table."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def _read_transfer_condition(exp15_dir: Path) -> pd.DataFrame:
    fp = exp15_dir / "tables" / "cross_dataset_transfer_condition_summary.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp)
    if df.empty:
        raise ValueError(f"Empty file: {fp}")
    return df


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
        label = str(payload.get("label") or model_name)
        exp15_raw = payload.get("exp15_run_dir", "")
        if not exp15_raw:
            missing.append(f"{model_name}:missing_exp15")
            continue
        exp15_dir = Path(str(exp15_raw))
        try:
            df = _read_transfer_condition(exp15_dir)
        except Exception as exc:
            missing.append(f"{model_name}:{exc}")
            continue

        for _, r in df.iterrows():
            delta = _to_float(r.get("stereotype_score_delta", ""))
            ci_low = _to_float(r.get("stereotype_score_delta_ci_low", ""))
            ci_high = _to_float(r.get("stereotype_score_delta_ci_high", ""))
            q = _to_float(r.get("q_score_sign", ""))
            n_pairs = _to_float(r.get("n_pairs", ""))
            ci_zero = _ci_excludes_zero(ci_low, ci_high)
            sig = bool(np.isfinite(q) and q < args.alpha)
            mismatch = ci_zero != sig
            mismatch_type = ""
            if mismatch:
                if ci_zero and not sig:
                    mismatch_type = "CI_excludes_zero_but_q_nonsig"
                elif sig and not ci_zero:
                    mismatch_type = "q_sig_but_CI_crosses_zero"
            rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "condition": str(r.get("condition", "")),
                    "rank_source": str(r.get("rank_source", "")),
                    "target_source": str(r.get("target_source", "")),
                    "n_pairs": n_pairs,
                    "delta_score": delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "q_score_sign": q,
                    "ci_excludes_zero": ci_zero,
                    "q_significant": sig,
                    "mismatch": mismatch,
                    "mismatch_type": mismatch_type,
                }
            )

    detail_df = pd.DataFrame(rows)
    detail_path = out_dir / "ci_vs_test_companion_rows.csv"
    summary_path = out_dir / "ci_vs_test_companion_summary.csv"
    worked_path = out_dir / "ci_vs_test_worked_example.csv"
    meta_path = out_dir / "ci_vs_test_companion_meta.json"

    if detail_df.empty:
        detail_df = pd.DataFrame(
            columns=[
                "model",
                "model_label",
                "condition",
                "rank_source",
                "target_source",
                "n_pairs",
                "delta_score",
                "ci_low",
                "ci_high",
                "q_score_sign",
                "ci_excludes_zero",
                "q_significant",
                "mismatch",
                "mismatch_type",
            ]
        )
    detail_df.to_csv(detail_path, index=False)

    if detail_df.empty:
        summary_df = pd.DataFrame(columns=["model", "model_label", "rows", "mismatch_rows", "ci_only_rows", "q_only_rows"])
    else:
        gb = detail_df.groupby(["model", "model_label"], sort=True)
        out = []
        for (model, label), g in gb:
            ci_only = int(np.sum(g["mismatch_type"] == "CI_excludes_zero_but_q_nonsig"))
            q_only = int(np.sum(g["mismatch_type"] == "q_sig_but_CI_crosses_zero"))
            out.append(
                {
                    "model": model,
                    "model_label": label,
                    "rows": int(len(g)),
                    "mismatch_rows": int(np.sum(g["mismatch"])),
                    "ci_only_rows": ci_only,
                    "q_only_rows": q_only,
                }
            )
        summary_df = pd.DataFrame(out)
    summary_df.to_csv(summary_path, index=False)

    worked_cols = [
        "model",
        "model_label",
        "condition",
        "rank_source",
        "target_source",
        "n_pairs",
        "delta_score",
        "ci_low",
        "ci_high",
        "q_score_sign",
        "mismatch_type",
    ]
    worked_df = detail_df[detail_df["mismatch"] == True] if not detail_df.empty else pd.DataFrame(columns=worked_cols)
    if not worked_df.empty:
        worked_df = worked_df.sort_values(["mismatch_type", "model_label", "condition"]).head(1)
    else:
        worked_df = pd.DataFrame(columns=worked_cols)
    worked_df[worked_cols].to_csv(worked_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "alpha": args.alpha,
        "rows": str(detail_path),
        "summary": str(summary_path),
        "worked_example": str(worked_path),
        "note": (
            "Bootstrap CI here summarizes the mean paired delta with resampling; paired sign-test q-values "
            "test directional sign consistency and can disagree under skew/ties/discrete deltas."
        ),
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(detail_path)
    print(summary_path)
    print(worked_path)
    print(meta_path)


if __name__ == "__main__":
    main()
