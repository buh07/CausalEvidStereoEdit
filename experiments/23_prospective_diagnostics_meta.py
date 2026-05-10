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
from stereacl.stats import benjamini_hochberg


DEFAULT_MODELS = [
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "/jumbo/lisp/f004ndc/models/mistral-7b-v0.1",
    "/jumbo/lisp/f004ndc/models/olmo-2-7b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 23: prospective model-level meta summary for Exp14/17/18 diagnostics."
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _latest_run_dir(slug: str, model: str, required_csv: str) -> Path:
    root = PROJECT_ROOT / "results" / slug
    candidates = sorted(root.glob("*/*/manifest.json"))
    best: tuple[str, Path] | None = None
    for mp in candidates:
        payload = json.loads(mp.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        if payload.get("parameters", {}).get("model") != model:
            continue
        rd = Path(payload["run_dir"])
        if not (rd / "tables" / required_csv).exists():
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, rd)
    if best is None:
        raise FileNotFoundError(f"No completed run for slug={slug} model={model}")
    return best[1]


def _to_float(x: Any) -> float:
    try:
        if x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _safe_q_from_p_family(rows: list[dict[str, Any]], p_key: str, q_key: str) -> None:
    vals = [_to_float(r.get(p_key, "")) for r in rows]
    q_vals = benjamini_hochberg(vals)
    for i, q in enumerate(q_vals):
        rows[i][q_key] = "" if np.isnan(q) or np.isinf(q) else round(float(q), 8)


def main() -> None:
    args = parse_args()
    ctx = start_run("23", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

        refs = {"models": models}
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Exp23 model list.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, "models": len(models)})
            return

        out_rows: list[dict[str, Any]] = []
        for model in models:
            r14 = _latest_run_dir("14_sign_reliability_audit", model, "sign_reliability_overall.csv")
            r17 = _latest_run_dir("17_suppressor_contamination_audit", model, "suppressor_contamination_overall.csv")
            r18 = _latest_run_dir("18_injection_controls", model, "injection_control_contrasts.csv")

            t14 = pd.read_csv(r14 / "tables" / "sign_reliability_overall.csv")
            t17 = pd.read_csv(r17 / "tables" / "suppressor_contamination_overall.csv")
            t18 = pd.read_csv(r18 / "tables" / "injection_control_contrasts.csv")

            row14 = t14.iloc[0].to_dict() if not t14.empty else {}
            row17 = t17.iloc[0].to_dict() if not t17.empty else {}

            tr = t18[t18["contrast"] == "true_minus_random"]
            ts = t18[t18["contrast"] == "true_minus_shuffled"]
            row_tr = tr.iloc[0].to_dict() if not tr.empty else {}
            row_ts = ts.iloc[0].to_dict() if not ts.empty else {}

            out_rows.append(
                {
                    "model": model,
                    "exp14_run_dir": str(r14),
                    "exp17_run_dir": str(r17),
                    "exp18_run_dir": str(r18),
                    "dla_sign_agreement_rate": round(_to_float(row14.get("dla_sign_agreement_rate", "")), 8),
                    "p_dla_sign_agreement": round(_to_float(row14.get("p_dla_sign_agreement", "")), 8),
                    "atp_sign_agreement_rate": round(_to_float(row14.get("atp_sign_agreement_rate", "")), 8),
                    "p_atp_sign_agreement": round(_to_float(row14.get("p_atp_sign_agreement", "")), 8),
                    "suppressor_fraction_topk": round(_to_float(row17.get("causal_suppressor_fraction", "")), 8),
                    "dla_sign_suppressor_fraction": round(_to_float(row17.get("dla_sign_suppressor_fraction", "")), 8),
                    "true_minus_random_score_delta": round(_to_float(row_tr.get("mean_score_contrast", "")), 8),
                    "p_true_minus_random_score": round(_to_float(row_tr.get("paired_p_score_sign", "")), 8),
                    "true_minus_shuffled_score_delta": round(_to_float(row_ts.get("mean_score_contrast", "")), 8),
                    "p_true_minus_shuffled_score": round(_to_float(row_ts.get("paired_p_score_sign", "")), 8),
                    "q_dla_sign_agreement_frozen": "",
                    "q_atp_sign_agreement_frozen": "",
                    "q_true_minus_random_score_frozen": "",
                    "q_true_minus_shuffled_score_frozen": "",
                }
            )

        _safe_q_from_p_family(out_rows, "p_dla_sign_agreement", "q_dla_sign_agreement_frozen")
        _safe_q_from_p_family(out_rows, "p_atp_sign_agreement", "q_atp_sign_agreement_frozen")
        _safe_q_from_p_family(out_rows, "p_true_minus_random_score", "q_true_minus_random_score_frozen")
        _safe_q_from_p_family(out_rows, "p_true_minus_shuffled_score", "q_true_minus_shuffled_score_frozen")

        out_path = ctx.tables_dir / "prospective_diagnostics_meta.csv"
        write_csv(
            out_path,
            out_rows,
            fieldnames=[
                "model",
                "exp14_run_dir",
                "exp17_run_dir",
                "exp18_run_dir",
                "dla_sign_agreement_rate",
                "p_dla_sign_agreement",
                "q_dla_sign_agreement_frozen",
                "atp_sign_agreement_rate",
                "p_atp_sign_agreement",
                "q_atp_sign_agreement_frozen",
                "suppressor_fraction_topk",
                "dla_sign_suppressor_fraction",
                "true_minus_random_score_delta",
                "p_true_minus_random_score",
                "q_true_minus_random_score_frozen",
                "true_minus_shuffled_score_delta",
                "p_true_minus_shuffled_score",
                "q_true_minus_shuffled_score_frozen",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Exp23 frozen-family diagnostics meta summary.")

        complete_run(
            ctx,
            metrics={
                "models": len(out_rows),
                "dla_significant_models": int(sum(1 for r in out_rows if _to_float(r.get("q_dla_sign_agreement_frozen", "")) < 0.05)),
                "true_vs_random_significant_models": int(sum(1 for r in out_rows if _to_float(r.get("q_true_minus_random_score_frozen", "")) < 0.05)),
                "true_vs_shuffled_significant_models": int(sum(1 for r in out_rows if _to_float(r.get("q_true_minus_shuffled_score_frozen", "")) < 0.05)),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
