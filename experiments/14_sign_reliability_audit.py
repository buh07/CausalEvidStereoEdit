#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import write_csv, write_json
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, spearman_safe, wilson_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 14: sign reliability audit (DLA vs AtP vs causal direction)."
    )
    parser.add_argument("--model", default="gpt2")
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


def _rounded(value: float | int | None) -> float | str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    if np.isnan(v) or np.isinf(v):
        return ""
    return round(v, 8)


def _normalize_component_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "head_index" not in out.columns:
        out["head_index"] = np.nan

    def _head(v: Any) -> int | None:
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        try:
            return int(v)
        except Exception:
            return None

    out["head_index"] = out["head_index"].map(_head)
    out["layer"] = out["layer"].astype(int)
    out["component_type"] = out["component_type"].astype(str)
    out["component_id"] = out["component_id"].astype(str)
    out["key"] = (
        out["axis"].astype(str)
        + "||"
        + out["component_type"].astype(str)
        + "||"
        + out["layer"].astype(str)
        + "||"
        + out["component_id"].astype(str)
        + "||"
        + out["head_index"].map(lambda x: "" if pd.isna(x) else str(int(x)))
    )
    return out


def _sign(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=int)
    out[arr > 0] = 1
    out[arr < 0] = -1
    return out


def _sign_agreement(score_delta: np.ndarray, signed_metric: np.ndarray) -> tuple[float, float, float, float, int, int]:
    # Causal promoter direction: ablating a promoter should reduce score => delta < 0 => +1 promoter sign.
    causal = _sign(-score_delta)
    metric_sign = _sign(signed_metric)
    mask = (causal != 0) & (metric_sign != 0) & np.isfinite(score_delta) & np.isfinite(signed_metric)
    n = int(np.sum(mask))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0, 0
    agree = int(np.sum(causal[mask] == metric_sign[mask]))
    rate = agree / n
    ci_lo, ci_hi = wilson_interval(agree, n)
    p = float(binomtest(k=agree, n=n, p=0.5, alternative="two-sided").pvalue)
    return rate, ci_lo, ci_hi, p, n, agree


def _compute_row(df: pd.DataFrame, axis_label: str) -> dict[str, Any]:
    score_delta = pd.to_numeric(df["stereotype_score_delta"], errors="coerce").to_numpy(dtype=float)
    margin_delta = pd.to_numeric(df["mean_margin_delta"], errors="coerce").to_numpy(dtype=float)
    dla_score = pd.to_numeric(df["mean_dla_score"], errors="coerce").to_numpy(dtype=float)
    atp_score = pd.to_numeric(df["mean_attr_score"], errors="coerce").to_numpy(dtype=float)

    rho_dla_score, p_dla_score, n_dla_score = spearman_safe(dla_score, -score_delta)
    rho_atp_score, p_atp_score, n_atp_score = spearman_safe(atp_score, -score_delta)
    rho_dla_margin, p_dla_margin, n_dla_margin = spearman_safe(dla_score, -margin_delta)
    rho_atp_margin, p_atp_margin, n_atp_margin = spearman_safe(atp_score, -margin_delta)

    dla_agree, dla_agree_lo, dla_agree_hi, p_dla_agree, n_dla_agree, k_dla_agree = _sign_agreement(score_delta, dla_score)
    atp_agree, atp_agree_lo, atp_agree_hi, p_atp_agree, n_atp_agree, k_atp_agree = _sign_agreement(score_delta, atp_score)

    return {
        "axis": axis_label,
        "n_components": int(len(df)),
        "rho_dla_vs_neg_score_delta": _rounded(rho_dla_score),
        "p_rho_dla_vs_neg_score_delta": _rounded(p_dla_score),
        "q_rho_dla_vs_neg_score_delta": "",
        "n_rho_dla_vs_neg_score_delta": n_dla_score,
        "rho_atp_vs_neg_score_delta": _rounded(rho_atp_score),
        "p_rho_atp_vs_neg_score_delta": _rounded(p_atp_score),
        "q_rho_atp_vs_neg_score_delta": "",
        "n_rho_atp_vs_neg_score_delta": n_atp_score,
        "rho_dla_vs_neg_margin_delta": _rounded(rho_dla_margin),
        "p_rho_dla_vs_neg_margin_delta": _rounded(p_dla_margin),
        "q_rho_dla_vs_neg_margin_delta": "",
        "n_rho_dla_vs_neg_margin_delta": n_dla_margin,
        "rho_atp_vs_neg_margin_delta": _rounded(rho_atp_margin),
        "p_rho_atp_vs_neg_margin_delta": _rounded(p_atp_margin),
        "q_rho_atp_vs_neg_margin_delta": "",
        "n_rho_atp_vs_neg_margin_delta": n_atp_margin,
        "dla_sign_agreement_rate": _rounded(dla_agree),
        "dla_sign_agreement_ci_low": _rounded(dla_agree_lo),
        "dla_sign_agreement_ci_high": _rounded(dla_agree_hi),
        "p_dla_sign_agreement": _rounded(p_dla_agree),
        "q_dla_sign_agreement": "",
        "n_dla_sign_agreement": n_dla_agree,
        "k_dla_sign_agreement": k_dla_agree,
        "atp_sign_agreement_rate": _rounded(atp_agree),
        "atp_sign_agreement_ci_low": _rounded(atp_agree_lo),
        "atp_sign_agreement_ci_high": _rounded(atp_agree_hi),
        "p_atp_sign_agreement": _rounded(p_atp_agree),
        "q_atp_sign_agreement": "",
        "n_atp_sign_agreement": n_atp_agree,
        "k_atp_sign_agreement": k_atp_agree,
    }


def _to_float_or_nan(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _apply_fdr(rows: list[dict[str, Any]], p_col: str, q_col: str) -> None:
    p_vals = [_to_float_or_nan(row.get(p_col, "")) for row in rows]
    q_vals = benjamini_hochberg(p_vals)
    for i, q in enumerate(q_vals):
        rows[i][q_col] = _rounded(q)


def main() -> None:
    args = parse_args()
    ctx = start_run("14", parameters=vars(args), project_root=PROJECT_ROOT)
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
            },
        )
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True})
            return

        df2 = _normalize_component_key(pd.read_csv(exp2_dir / "tables" / "component_dla_scores.csv"))
        df3 = _normalize_component_key(pd.read_csv(exp3_dir / "tables" / "attribution_patch_scores.csv"))
        df9 = _normalize_component_key(pd.read_csv(exp9_dir / "tables" / "adjudication_single_ablation.csv"))

        keep2 = ["key", "mean_dla_score", "mean_abs_dla_score"]
        keep3 = ["key", "mean_attr_score", "mean_abs_attr_score"]
        keep9 = ["key", "axis", "component_type", "layer", "component_id", "head_index", "stereotype_score_delta", "mean_margin_delta"]

        merged = df9[keep9].merge(df2[keep2], on="key", how="left").merge(df3[keep3], on="key", how="left")

        by_axis_rows: list[dict[str, Any]] = []
        for axis, group in merged.groupby("axis"):
            by_axis_rows.append(_compute_row(group, axis_label=str(axis)))

        # BH-FDR correction across per-axis tests.
        for p_col, q_col in [
            ("p_rho_dla_vs_neg_score_delta", "q_rho_dla_vs_neg_score_delta"),
            ("p_rho_atp_vs_neg_score_delta", "q_rho_atp_vs_neg_score_delta"),
            ("p_rho_dla_vs_neg_margin_delta", "q_rho_dla_vs_neg_margin_delta"),
            ("p_rho_atp_vs_neg_margin_delta", "q_rho_atp_vs_neg_margin_delta"),
            ("p_dla_sign_agreement", "q_dla_sign_agreement"),
            ("p_atp_sign_agreement", "q_atp_sign_agreement"),
        ]:
            _apply_fdr(by_axis_rows, p_col, q_col)

        overall_row = _compute_row(merged, axis_label="overall")
        # Mirror table-level inference policy for the model-level row so downstream
        # summaries can report corrected q-values consistently.
        for p_col, q_col in [
            ("p_rho_dla_vs_neg_score_delta", "q_rho_dla_vs_neg_score_delta"),
            ("p_rho_atp_vs_neg_score_delta", "q_rho_atp_vs_neg_score_delta"),
            ("p_rho_dla_vs_neg_margin_delta", "q_rho_dla_vs_neg_margin_delta"),
            ("p_rho_atp_vs_neg_margin_delta", "q_rho_atp_vs_neg_margin_delta"),
            ("p_dla_sign_agreement", "q_dla_sign_agreement"),
            ("p_atp_sign_agreement", "q_atp_sign_agreement"),
        ]:
            _apply_fdr([overall_row], p_col, q_col)

        by_axis_path = ctx.tables_dir / "sign_reliability_by_axis.csv"
        overall_path = ctx.tables_dir / "sign_reliability_overall.csv"

        fieldnames = list(by_axis_rows[0].keys()) if by_axis_rows else list(overall_row.keys())
        write_csv(by_axis_path, by_axis_rows, fieldnames=fieldnames)
        write_csv(overall_path, [overall_row], fieldnames=list(overall_row.keys()))

        ctx.register_artifact(by_axis_path, artifact_type="table", description="Per-axis sign reliability audit.")
        ctx.register_artifact(overall_path, artifact_type="table", description="Overall sign reliability audit.")

        summary_md = ctx.artifacts_dir / "sign_reliability_summary.md"
        summary_lines = [
            "# Exp14 Sign Reliability Audit",
            "",
            f"- Model: `{args.model}`",
            f"- Components evaluated: `{len(merged)}`",
            f"- Axes evaluated: `{len(by_axis_rows)}`",
            "",
            "## Overall",
            f"- rho(DLA, -score_delta): `{overall_row['rho_dla_vs_neg_score_delta']}`",
            f"- rho(AtP, -score_delta): `{overall_row['rho_atp_vs_neg_score_delta']}`",
            f"- DLA sign agreement: `{overall_row['dla_sign_agreement_rate']}`",
            f"- AtP sign agreement: `{overall_row['atp_sign_agreement_rate']}`",
            "",
        ]
        summary_md.write_text("\n".join(summary_lines), encoding="utf-8")
        ctx.register_artifact(summary_md, artifact_type="artifact", description="Short sign-audit narrative summary.")

        complete_run(
            ctx,
            metrics={
                "rows_axis": len(by_axis_rows),
                "rows_overall": 1,
                "components_merged": int(len(merged)),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
