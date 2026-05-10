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

from stereacl.analysis import write_csv, write_json
from stereacl.run_context import complete_run, fail_run, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 17: suppressor contamination audit using causal ground truth from Exp09."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ranking-source", choices=["union", "dla", "atp"], default="union")
    parser.add_argument("--exp2-run-dir", default="")
    parser.add_argument("--exp3-run-dir", default="")
    parser.add_argument("--exp9-run-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _latest_run_dir(
    experiment_slug: str,
    required_relpaths: list[str] | None = None,
    model_name: str | None = None,
) -> Path:
    root = PROJECT_ROOT / "results" / experiment_slug
    candidates = sorted(root.glob("*/*/manifest.json"))
    best: tuple[str, Path] | None = None
    for manifest_path in candidates:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        if model_name is not None and payload.get("parameters", {}).get("model") != model_name:
            continue
        ended = payload.get("ended_at_utc") or ""
        run_dir = Path(payload["run_dir"])
        if required_relpaths:
            if any(not (run_dir / rel).exists() for rel in required_relpaths):
                continue
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        raise FileNotFoundError(f"No completed run found for {experiment_slug}.")
    return best[1]


def _rounded(v: float | int | None) -> float | str:
    if v is None:
        return ""
    try:
        x = float(v)
    except Exception:
        return ""
    if np.isnan(x) or np.isinf(x):
        return ""
    return round(x, 8)


def _rank_for_source(row: pd.Series, ranking_source: str) -> float:
    def _f(name: str) -> float:
        val = row.get(name, "")
        if val == "" or pd.isna(val):
            return float("inf")
        try:
            return float(val)
        except Exception:
            return float("inf")

    if ranking_source == "dla":
        return _f("dla_rank")
    if ranking_source == "atp":
        return _f("atp_rank")
    return min(_f("dla_rank"), _f("atp_rank"))


def main() -> None:
    args = parse_args()
    ctx = start_run("17", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp2_dir = (
            Path(args.exp2_run_dir)
            if args.exp2_run_dir
            else _latest_run_dir(
                "02_component_dla",
                required_relpaths=["tables/component_dla_scores.csv"],
                model_name=args.model,
            )
        )
        exp3_dir = (
            Path(args.exp3_run_dir)
            if args.exp3_run_dir
            else _latest_run_dir(
                "03_attribution_patching",
                required_relpaths=["tables/attribution_patch_scores.csv"],
                model_name=args.model,
            )
        )
        exp9_dir = (
            Path(args.exp9_run_dir)
            if args.exp9_run_dir
            else _latest_run_dir(
                "09_dla_atp_adjudication",
                required_relpaths=["tables/adjudication_single_ablation.csv"],
                model_name=args.model,
            )
        )

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs_path,
            {
                "exp2_run_dir": str(exp2_dir),
                "exp3_run_dir": str(exp3_dir),
                "exp9_run_dir": str(exp9_dir),
                "top_k": args.top_k,
                "ranking_source": args.ranking_source,
            },
        )
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True})
            return

        df2 = pd.read_csv(exp2_dir / "tables" / "component_dla_scores.csv")
        df3 = pd.read_csv(exp3_dir / "tables" / "attribution_patch_scores.csv")
        df9 = pd.read_csv(exp9_dir / "tables" / "adjudication_single_ablation.csv")

        for df in (df2, df3, df9):
            if "head_index" not in df.columns:
                df["head_index"] = np.nan
            df["head_index"] = pd.to_numeric(df["head_index"], errors="coerce")
            df["layer"] = pd.to_numeric(df["layer"], errors="coerce").astype("Int64")
            df["component_type"] = df["component_type"].astype(str)
            df["component_id"] = df["component_id"].astype(str)
            df["axis"] = df["axis"].astype(str)

        keys = ["axis", "component_type", "layer", "component_id", "head_index"]
        d2 = df2[keys + [c for c in ["mean_dla_score", "mean_abs_dla_score"] if c in df2.columns]].copy()
        d3 = df3[keys + [c for c in ["mean_attr_score", "mean_abs_attr_score"] if c in df3.columns]].copy()
        d9 = df9.copy()

        merged = d9.merge(d2, on=keys, how="left").merge(d3, on=keys, how="left")

        merged["rank_source"] = merged.apply(lambda r: _rank_for_source(r, args.ranking_source), axis=1)
        merged = merged[np.isfinite(pd.to_numeric(merged["rank_source"], errors="coerce"))].copy()
        merged["rank_source"] = pd.to_numeric(merged["rank_source"], errors="coerce")

        # Causal labels from observed single-site deltas.
        # delta < 0 => promoter, delta > 0 => suppressor, delta == 0 => neutral.
        merged["causal_label"] = np.where(
            pd.to_numeric(merged["stereotype_score_delta"], errors="coerce") < 0,
            "promoter",
            np.where(
                pd.to_numeric(merged["stereotype_score_delta"], errors="coerce") > 0,
                "suppressor",
                "neutral",
            ),
        )
        merged["dla_sign_label"] = np.where(
            pd.to_numeric(merged["mean_dla_score"], errors="coerce") < 0,
            "suppressor",
            np.where(
                pd.to_numeric(merged["mean_dla_score"], errors="coerce") > 0,
                "promoter",
                "neutral",
            ),
        )

        axis_rows: list[dict[str, Any]] = []
        for axis, group in merged.groupby("axis"):
            ranked = group.sort_values("rank_source", ascending=True).head(args.top_k).copy()
            if ranked.empty:
                continue
            suppressors_causal = int((ranked["causal_label"] == "suppressor").sum())
            promoters_causal = int((ranked["causal_label"] == "promoter").sum())
            neutral_causal = int((ranked["causal_label"] == "neutral").sum())
            suppressors_dla = int((ranked["dla_sign_label"] == "suppressor").sum())

            score_delta_sum = float(pd.to_numeric(ranked["stereotype_score_delta"], errors="coerce").sum())
            margin_delta_sum = float(pd.to_numeric(ranked["mean_margin_delta"], errors="coerce").sum())
            score_delta_mean = float(pd.to_numeric(ranked["stereotype_score_delta"], errors="coerce").mean())
            margin_delta_mean = float(pd.to_numeric(ranked["mean_margin_delta"], errors="coerce").mean())

            n = int(len(ranked))
            axis_rows.append(
                {
                    "axis": str(axis),
                    "ranking_source": args.ranking_source,
                    "top_k": int(args.top_k),
                    "n_selected": n,
                    "causal_promoters": promoters_causal,
                    "causal_suppressors": suppressors_causal,
                    "causal_neutral": neutral_causal,
                    "causal_suppressor_fraction": _rounded(suppressors_causal / n if n else float("nan")),
                    "dla_sign_suppressors": suppressors_dla,
                    "dla_sign_suppressor_fraction": _rounded(suppressors_dla / n if n else float("nan")),
                    "mean_score_delta_over_selected": _rounded(score_delta_mean),
                    "sum_score_delta_over_selected": _rounded(score_delta_sum),
                    "mean_margin_delta_over_selected": _rounded(margin_delta_mean),
                    "sum_margin_delta_over_selected": _rounded(margin_delta_sum),
                }
            )

        axis_out = ctx.tables_dir / "suppressor_contamination_by_axis.csv"
        write_csv(
            axis_out,
            axis_rows,
            fieldnames=[
                "axis",
                "ranking_source",
                "top_k",
                "n_selected",
                "causal_promoters",
                "causal_suppressors",
                "causal_neutral",
                "causal_suppressor_fraction",
                "dla_sign_suppressors",
                "dla_sign_suppressor_fraction",
                "mean_score_delta_over_selected",
                "sum_score_delta_over_selected",
                "mean_margin_delta_over_selected",
                "sum_margin_delta_over_selected",
            ],
        )
        ctx.register_artifact(
            axis_out,
            artifact_type="table",
            description="Per-axis suppressor contamination in top-k using causal ground truth labels.",
        )

        overall = pd.DataFrame(axis_rows)
        if not overall.empty:
            n_total = int(overall["n_selected"].sum())
            overall_row = {
                "axis": "overall",
                "ranking_source": args.ranking_source,
                "top_k": int(args.top_k),
                "n_selected": n_total,
                "causal_promoters": int(overall["causal_promoters"].sum()),
                "causal_suppressors": int(overall["causal_suppressors"].sum()),
                "causal_neutral": int(overall["causal_neutral"].sum()),
                "causal_suppressor_fraction": _rounded(
                    float(overall["causal_suppressors"].sum()) / n_total if n_total else float("nan")
                ),
                "dla_sign_suppressors": int(overall["dla_sign_suppressors"].sum()),
                "dla_sign_suppressor_fraction": _rounded(
                    float(overall["dla_sign_suppressors"].sum()) / n_total if n_total else float("nan")
                ),
                "mean_score_delta_over_selected": _rounded(
                    float(pd.to_numeric(overall["mean_score_delta_over_selected"], errors="coerce").mean())
                ),
                "sum_score_delta_over_selected": _rounded(
                    float(pd.to_numeric(overall["sum_score_delta_over_selected"], errors="coerce").sum())
                ),
                "mean_margin_delta_over_selected": _rounded(
                    float(pd.to_numeric(overall["mean_margin_delta_over_selected"], errors="coerce").mean())
                ),
                "sum_margin_delta_over_selected": _rounded(
                    float(pd.to_numeric(overall["sum_margin_delta_over_selected"], errors="coerce").sum())
                ),
            }
        else:
            overall_row = {
                "axis": "overall",
                "ranking_source": args.ranking_source,
                "top_k": int(args.top_k),
                "n_selected": 0,
                "causal_promoters": 0,
                "causal_suppressors": 0,
                "causal_neutral": 0,
                "causal_suppressor_fraction": "",
                "dla_sign_suppressors": 0,
                "dla_sign_suppressor_fraction": "",
                "mean_score_delta_over_selected": "",
                "sum_score_delta_over_selected": "",
                "mean_margin_delta_over_selected": "",
                "sum_margin_delta_over_selected": "",
            }

        overall_out = ctx.tables_dir / "suppressor_contamination_overall.csv"
        write_csv(overall_out, [overall_row], fieldnames=list(overall_row.keys()))
        ctx.register_artifact(overall_out, artifact_type="table", description="Overall suppressor contamination summary.")

        complete_run(
            ctx,
            metrics={
                "rows_axis": len(axis_rows),
                "rows_overall": 1,
                "top_k": int(args.top_k),
                "ranking_source": args.ranking_source,
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()

