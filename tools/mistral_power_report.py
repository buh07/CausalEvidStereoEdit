#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


MISTRAL_NAMES = {
    "Mistral-7B",
    "mistral7b",
    "/jumbo/lisp/f004ndc/models/mistral-7b-v0.1",
}


def _to_float(v: Any) -> float:
    try:
        if v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _approx_mde(n_pairs: int, alpha: float, target_power: float) -> float:
    if n_pairs <= 0:
        return float("nan")
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(target_power))
    return 0.5 * (z_alpha + z_beta) / np.sqrt(float(n_pairs))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Mistral extension power/MDE appendix report from frozen Exp16 artifacts."
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--contrast", default="primary_inject_anti_minus_remove_stereo")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--target-power", type=float, default=0.80)
    p.add_argument("--sesoi", type=float, default=0.10)
    return p.parse_args()


def _find_mistral(models: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for model_name, payload_any in models.items():
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = str(payload.get("label", ""))
        key = str(payload.get("key", ""))
        if model_name in MISTRAL_NAMES or label in MISTRAL_NAMES or key in MISTRAL_NAMES:
            return model_name, payload
        if "mistral" in model_name.lower() or "mistral" in label.lower() or "mistral" in key.lower():
            return model_name, payload
    return None


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    match = _find_mistral(models)
    if match is None:
        raise ValueError("Could not find a Mistral model in run map.")

    model_name, payload = match
    label = str(payload.get("label") or model_name)
    exp16_dir_raw = payload.get("exp16_canonical_run_dir", "")
    if not exp16_dir_raw:
        raise ValueError("Mistral row missing exp16_canonical_run_dir.")
    exp16_dir = Path(str(exp16_dir_raw))

    table_path = exp16_dir / "tables" / "asymmetry_contrast.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    df = pd.read_csv(table_path)
    row = df[df["contrast"] == args.contrast]
    if row.empty:
        raise ValueError(f"Contrast {args.contrast} not found in {table_path}")
    r = row.iloc[0]

    n_pairs = int(_to_float(r.get("n_pairs", 0))) if np.isfinite(_to_float(r.get("n_pairs", 0))) else 0
    observed = _to_float(r.get("mean_score_contrast", ""))
    p_score = _to_float(r.get("paired_p_score_sign", ""))
    q_score = _to_float(r.get("q_score_sign", ""))
    ci_low = _to_float(r.get("mean_score_contrast_ci_low", ""))
    ci_high = _to_float(r.get("mean_score_contrast_ci_high", ""))

    mde = _approx_mde(n_pairs=n_pairs, alpha=args.alpha, target_power=args.target_power)
    if np.isfinite(mde) and np.isfinite(observed):
        required_multiplier = mde / max(abs(observed), 1e-9)
    else:
        required_multiplier = float("nan")

    power_tag = "underpowered_for_sesoi"
    if np.isfinite(mde):
        power_tag = "adequate_for_sesoi" if mde <= args.sesoi else "underpowered_for_sesoi"

    verdict = "inconclusive_under_current_power"
    if np.isfinite(q_score) and q_score < args.alpha:
        verdict = "significant"

    rows = [
        {
            "model": model_name,
            "model_label": label,
            "exp16_run_dir": str(exp16_dir),
            "contrast": args.contrast,
            "n_pairs": n_pairs,
            "observed_score_contrast": observed,
            "score_ci_low": ci_low,
            "score_ci_high": ci_high,
            "p_score_sign": p_score,
            "q_score_sign": q_score,
            "alpha": args.alpha,
            "target_power": args.target_power,
            "sesoi": args.sesoi,
            "mde_score_approx": mde,
            "power_vs_sesoi": power_tag,
            "mde_over_observed_abs": required_multiplier,
            "verdict": verdict,
            "interpretation": (
                "Directional estimate can be positive while remaining non-significant when n is low; "
                "treat as inconclusive unless powered near SESOI."
            ),
        }
    ]

    out_csv = out_dir / "mistral_power_report.csv"
    meta_path = out_dir / "mistral_power_report_meta.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    meta = {
        "run_map": str(run_map_path),
        "output": str(out_csv),
        "mistral_model": model_name,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(out_csv)
    print(meta_path)


if __name__ == "__main__":
    main()
