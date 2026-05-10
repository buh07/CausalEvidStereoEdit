#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    p = argparse.ArgumentParser(description="Summarize Phase 1 prospective replication outputs.")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--seeds", default="11,29,47")
    return p.parse_args()


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _latest_manifest(slug: str, model: str, seed: int | None, start_dt: datetime) -> Path | None:
    root = PROJECT_ROOT / "results" / slug
    best: tuple[datetime, Path] | None = None
    for mp in root.glob("*/*/manifest.json"):
        try:
            payload = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") != "completed":
            continue
        params = payload.get("parameters", {})
        if params.get("model") != model:
            continue
        if seed is not None and int(params.get("seed", -999999)) != seed:
            continue
        ended = _parse_ts(payload.get("ended_at_utc", ""))
        if ended is None or ended < start_dt:
            continue
        if best is None or ended > best[0]:
            best = (ended, Path(payload["run_dir"]))
    return best[1] if best else None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_float(x: Any) -> float:
    try:
        if x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    state_dir = PROJECT_ROOT / "results" / args.run_tag / "state"
    start_path = state_dir / "orch_start_utc.txt"
    if not start_path.exists():
        raise FileNotFoundError(f"Missing {start_path}")
    start_dt = _parse_ts(start_path.read_text(encoding="utf-8").strip())
    if start_dt is None:
        raise ValueError("Could not parse orch_start_utc.txt")

    rows16: list[dict[str, Any]] = []
    rows18: list[dict[str, Any]] = []
    rows14: list[dict[str, Any]] = []
    rows17: list[dict[str, Any]] = []

    for model in models:
        # Exp16/18 per seed
        for seed in seeds:
            r16 = _latest_manifest("16_asymmetry_matrix", model, seed, start_dt)
            if r16:
                t = _read_csv(r16 / "tables" / "asymmetry_contrast.csv")
                tr = t[t["contrast"] == "primary_inject_anti_minus_remove_stereo"]
                if not tr.empty:
                    rr = tr.iloc[0].to_dict()
                    rows16.append(
                        {
                            "model": model,
                            "seed": seed,
                            "run_dir": str(r16),
                            "effect": _to_float(rr.get("mean_score_contrast", "")),
                            "ci_low": _to_float(rr.get("mean_score_contrast_ci_low", "")),
                            "ci_high": _to_float(rr.get("mean_score_contrast_ci_high", "")),
                            "q_score": _to_float(rr.get("q_score_sign", "")),
                        }
                    )

            r18 = _latest_manifest("18_injection_controls", model, seed, start_dt)
            if r18:
                t = _read_csv(r18 / "tables" / "injection_control_contrasts.csv")
                rr = t[t["contrast"] == "true_minus_random"]
                rs = t[t["contrast"] == "true_minus_shuffled"]
                row_r = rr.iloc[0].to_dict() if not rr.empty else {}
                row_s = rs.iloc[0].to_dict() if not rs.empty else {}
                rows18.append(
                    {
                        "model": model,
                        "seed": seed,
                        "run_dir": str(r18),
                        "true_minus_random_effect": _to_float(row_r.get("mean_score_contrast", "")),
                        "true_minus_random_q": _to_float(row_r.get("q_score_sign", "")),
                        "true_minus_shuffled_effect": _to_float(row_s.get("mean_score_contrast", "")),
                        "true_minus_shuffled_q": _to_float(row_s.get("q_score_sign", "")),
                    }
                )

        # Exp14/17 single seed latest in this orchestration window
        r14 = _latest_manifest("14_sign_reliability_audit", model, None, start_dt)
        if r14:
            t14 = _read_csv(r14 / "tables" / "sign_reliability_overall.csv")
            if not t14.empty:
                row = t14.iloc[0].to_dict()
                rows14.append(
                    {
                        "model": model,
                        "run_dir": str(r14),
                        "dla_sign_agreement_rate": _to_float(row.get("dla_sign_agreement_rate", "")),
                        "dla_sign_q": _to_float(row.get("q_dla_sign_agreement", "")),
                        "atp_sign_agreement_rate": _to_float(row.get("atp_sign_agreement_rate", "")),
                        "atp_sign_q": _to_float(row.get("q_atp_sign_agreement", "")),
                    }
                )

        r17 = _latest_manifest("17_suppressor_contamination_audit", model, None, start_dt)
        if r17:
            t17 = _read_csv(r17 / "tables" / "suppressor_contamination_overall.csv")
            if not t17.empty:
                row = t17.iloc[0].to_dict()
                rows17.append(
                    {
                        "model": model,
                        "run_dir": str(r17),
                        "suppressor_fraction": _to_float(row.get("causal_suppressor_fraction", "")),
                        "dla_sign_suppressor_fraction": _to_float(row.get("dla_sign_suppressor_fraction", "")),
                    }
                )

    out_dir = PROJECT_ROOT / "results" / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df16 = pd.DataFrame(rows16)
    df18 = pd.DataFrame(rows18)
    df14 = pd.DataFrame(rows14)
    df17 = pd.DataFrame(rows17)

    df16.to_csv(out_dir / "phase1_exp16_seed_results.csv", index=False)
    df18.to_csv(out_dir / "phase1_exp18_seed_results.csv", index=False)
    df14.to_csv(out_dir / "phase1_exp14_model_results.csv", index=False)
    df17.to_csv(out_dir / "phase1_exp17_model_results.csv", index=False)

    summary_rows = []
    for model in models:
        m16 = df16[df16["model"] == model] if not df16.empty else pd.DataFrame()
        m18 = df18[df18["model"] == model] if not df18.empty else pd.DataFrame()
        m14 = df14[df14["model"] == model] if not df14.empty else pd.DataFrame()
        m17 = df17[df17["model"] == model] if not df17.empty else pd.DataFrame()
        summary_rows.append(
            {
                "model": model,
                "exp16_seed_count": int(len(m16)),
                "exp16_mean_effect": float(m16["effect"].mean()) if not m16.empty else np.nan,
                "exp16_sig_seed_count": int((m16["q_score"] < 0.05).sum()) if not m16.empty else 0,
                "exp18_seed_count": int(len(m18)),
                "exp18_true_random_sig_seed_count": int((m18["true_minus_random_q"] < 0.05).sum()) if not m18.empty else 0,
                "exp18_true_shuffled_sig_seed_count": int((m18["true_minus_shuffled_q"] < 0.05).sum()) if not m18.empty else 0,
                "exp14_dla_q": float(m14["dla_sign_q"].iloc[0]) if not m14.empty else np.nan,
                "exp17_suppressor_fraction": float(m17["suppressor_fraction"].iloc[0]) if not m17.empty else np.nan,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "phase1_prospective_summary.csv", index=False)

    # Majority metric for headline tracking
    if not df_summary.empty:
        majority_5of7_metric = int((df_summary["exp16_sig_seed_count"] >= 2).sum())
    else:
        majority_5of7_metric = 0

    md_lines = []
    md_lines.append(f"# Phase 1 Prospective Summary ({args.run_tag})")
    md_lines.append("")
    md_lines.append(f"- Models covered: {len(models)}")
    md_lines.append(f"- Exp16 majority metric (>=2/3 significant seeds): {majority_5of7_metric}/{len(models)}")
    md_lines.append("")
    md_lines.append("## Model Summary")
    md_lines.append("")
    if df_summary.empty:
        md_lines.append("No completed model summaries found yet.")
    else:
        try:
            md_lines.append(df_summary.to_markdown(index=False))
        except Exception:
            md_lines.append("`tabulate` not installed; emitting CSV-style rows instead.")
            md_lines.append("")
            md_lines.append(",".join(df_summary.columns.tolist()))
            for _, r in df_summary.iterrows():
                md_lines.append(",".join(str(r[c]) for c in df_summary.columns))

    (out_dir / "phase1_prospective_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote summaries under: {out_dir}")


if __name__ == "__main__":
    main()
