#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import write_csv, write_json
from stereacl.run_context import complete_run, fail_run, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 21: transfer equivalence framing with SESOI bounds from Exp15 summaries."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--sesoi", type=float, default=0.10, help="Equivalence margin for Δscore.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha for MDE approximation.")
    parser.add_argument("--target-power", type=float, default=0.80, help="Target power for MDE approximation.")
    parser.add_argument("--exp15-run-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _latest_exp15_run(model: str) -> Path:
    root = PROJECT_ROOT / "results" / "15_cross_dataset_component_transfer"
    candidates = sorted(root.glob("*/*/manifest.json"))
    best: tuple[str, Path] | None = None
    for mp in candidates:
        payload = json.loads(mp.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        if payload.get("parameters", {}).get("model") != model:
            continue
        run_dir = Path(payload["run_dir"])
        if not (run_dir / "tables" / "cross_dataset_transfer_condition_summary.csv").exists():
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        raise FileNotFoundError(f"No completed Exp15 run found for model={model}")
    return best[1]


def _to_float(x: Any) -> float:
    try:
        if x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _classify(ci_low: float, ci_high: float, sesoi: float) -> str:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "missing"
    if ci_low > -sesoi and ci_high < sesoi:
        return "equivalent_within_sesoi"
    if ci_low > sesoi:
        return "positive_effect_outside_sesoi"
    if ci_high < -sesoi:
        return "negative_effect_outside_sesoi"
    return "inconclusive"


def _approx_mde_score(n_pairs: int, alpha: float, target_power: float) -> float:
    """Approximate minimum detectable score delta for paired-binary style tests."""
    if n_pairs <= 0:
        return float("nan")
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(target_power))
    return 0.5 * (z_alpha + z_beta) / np.sqrt(float(n_pairs))


def main() -> None:
    args = parse_args()
    ctx = start_run("21", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp15_dir = Path(args.exp15_run_dir) if args.exp15_run_dir else _latest_exp15_run(args.model)
        summary_path = exp15_dir / "tables" / "cross_dataset_transfer_condition_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing condition summary: {summary_path}")

        refs = {
            "exp15_run_dir": str(exp15_dir),
            "summary_csv": str(summary_path),
            "sesoi": args.sesoi,
            "alpha": args.alpha,
            "target_power": args.target_power,
        }
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **refs})
            return

        df = pd.read_csv(summary_path)
        if df.empty:
            out_rows: list[dict[str, Any]] = []
        else:
            out_rows = []
            for _, row in df.iterrows():
                condition = str(row.get("condition", ""))
                rank_source = str(row.get("rank_source", ""))
                target_source = str(row.get("target_source", ""))
                delta = _to_float(row.get("stereotype_score_delta", ""))
                ci_low = _to_float(row.get("stereotype_score_delta_ci_low", ""))
                ci_high = _to_float(row.get("stereotype_score_delta_ci_high", ""))
                cls = _classify(ci_low, ci_high, args.sesoi)
                cross_source = rank_source != target_source
                n_pairs = int(_to_float(row.get("n_pairs", 0))) if not np.isnan(_to_float(row.get("n_pairs", 0))) else 0
                mde = _approx_mde_score(n_pairs, args.alpha, args.target_power)
                if np.isnan(mde):
                    power_tag = "unknown"
                elif mde <= args.sesoi:
                    power_tag = "adequate_for_sesoi"
                else:
                    power_tag = "underpowered_for_sesoi"
                out_rows.append(
                    {
                        "condition": condition,
                        "rank_source": rank_source,
                        "target_source": target_source,
                        "cross_source": cross_source,
                        "n_pairs": n_pairs,
                        "delta_score": "" if np.isnan(delta) else round(float(delta), 8),
                        "delta_score_ci_low": "" if np.isnan(ci_low) else round(float(ci_low), 8),
                        "delta_score_ci_high": "" if np.isnan(ci_high) else round(float(ci_high), 8),
                        "sesoi": args.sesoi,
                        "alpha": args.alpha,
                        "target_power": args.target_power,
                        "mde_score_approx": "" if np.isnan(mde) else round(float(mde), 8),
                        "power_vs_sesoi": power_tag,
                        "equivalence_decision": cls,
                    }
                )

        out_path = ctx.tables_dir / "transfer_equivalence_summary.csv"
        write_csv(
            out_path,
            out_rows,
            fieldnames=[
                "condition",
                "rank_source",
                "target_source",
                "cross_source",
                "n_pairs",
                "delta_score",
                "delta_score_ci_low",
                "delta_score_ci_high",
                "sesoi",
                "alpha",
                "target_power",
                "mde_score_approx",
                "power_vs_sesoi",
                "equivalence_decision",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Exp21 equivalence classifications.")

        cross_rows = [r for r in out_rows if bool(r["cross_source"])]
        counts: dict[str, int] = {}
        for r in cross_rows:
            key = str(r["equivalence_decision"])
            counts[key] = counts.get(key, 0) + 1

        metrics = {
            "rows": len(out_rows),
            "cross_rows": len(cross_rows),
            "equivalent_cross_rows": counts.get("equivalent_within_sesoi", 0),
            "inconclusive_cross_rows": counts.get("inconclusive", 0),
            "sesoi": args.sesoi,
            "alpha": args.alpha,
            "target_power": args.target_power,
            "dry_run": False,
        }
        complete_run(ctx, metrics=metrics)
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
