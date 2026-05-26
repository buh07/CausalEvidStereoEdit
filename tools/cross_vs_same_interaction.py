#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def _to_float(v: Any) -> float:
    try:
        if v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _bootstrap_mean_ci(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    n = finite.size
    for i in range(n_resamples):
        sample = rng.choice(finite, size=n, replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute cross-position vs same-position interaction on matched per-pair "
            "inject-minus-remove score contrasts."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--bootstrap-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def _label(model: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model, model)


def _read_primary_mean(run_dir: Path, table_name: str) -> float:
    df = pd.read_csv(run_dir / "tables" / table_name)
    row = df[df["contrast"] == "primary_inject_anti_minus_remove_stereo"]
    if row.empty:
        return float("nan")
    return _to_float(row.iloc[0].get("mean_score_contrast", ""))


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map is missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
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
        exp16_dir = Path(str(exp16_dir_raw))
        exp26_dir = Path(str(exp26_dir_raw))

        p16 = exp16_dir / "tables" / "asymmetry_pair_deltas.csv"
        p26 = exp26_dir / "tables" / "ar_same_position_pair_deltas.csv"
        if not p16.exists() or not p26.exists():
            missing.append(f"{model_name}:missing_pair_tables")
            continue

        d16 = pd.read_csv(p16)[["pair_id", "primary_score_contrast"]].copy()
        d26 = pd.read_csv(p26)[["pair_id", "primary_score_contrast"]].copy()
        d16["cross_primary"] = pd.to_numeric(d16["primary_score_contrast"], errors="coerce")
        d26["same_primary"] = pd.to_numeric(d26["primary_score_contrast"], errors="coerce")
        merged = d16[["pair_id", "cross_primary"]].merge(d26[["pair_id", "same_primary"]], on="pair_id", how="inner")
        merged = merged.dropna(subset=["cross_primary", "same_primary"]).copy()
        if merged.empty:
            missing.append(f"{model_name}:no_overlap_pairs")
            continue

        interaction = merged["same_primary"].to_numpy(dtype=float) - merged["cross_primary"].to_numpy(dtype=float)
        mean_diff = float(np.mean(interaction))
        ci_lo, ci_hi = _bootstrap_mean_ci(interaction, n_resamples=args.bootstrap_n, seed=args.seed)

        nonzero = interaction[np.abs(interaction) > 1e-12]
        if nonzero.size:
            n_pos = int(np.sum(nonzero > 0))
            n_neg = int(np.sum(nonzero < 0))
            n_eff = n_pos + n_neg
            p_sign = float(binomtest(k=n_pos, n=n_eff, p=0.5, alternative="two-sided").pvalue)
            try:
                p_wilcoxon = float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", mode="auto").pvalue)
            except Exception:
                p_wilcoxon = float("nan")
        else:
            n_pos = 0
            n_neg = 0
            p_sign = float("nan")
            p_wilcoxon = float("nan")

        cross_mean = _read_primary_mean(exp16_dir, "asymmetry_contrast.csv")
        same_mean = _read_primary_mean(exp26_dir, "ar_same_position_contrast.csv")

        rows.append(
            {
                "model": model_name,
                "model_label": label,
                "exp16_run_dir": str(exp16_dir),
                "exp26_run_dir": str(exp26_dir),
                "n_overlap_pairs": int(len(merged)),
                "cross_mean_score_contrast": cross_mean,
                "same_mean_score_contrast": same_mean,
                "mean_interaction_same_minus_cross": mean_diff,
                "interaction_ci_low": ci_lo,
                "interaction_ci_high": ci_hi,
                "n_positive_interaction": n_pos,
                "n_negative_interaction": n_neg,
                "p_sign": p_sign,
                "p_wilcoxon": p_wilcoxon,
            }
        )

    out_csv = out_dir / "cross_vs_same_interaction.csv"
    columns = [
        "model",
        "model_label",
        "exp16_run_dir",
        "exp26_run_dir",
        "n_overlap_pairs",
        "cross_mean_score_contrast",
        "same_mean_score_contrast",
        "mean_interaction_same_minus_cross",
        "interaction_ci_low",
        "interaction_ci_high",
        "n_positive_interaction",
        "n_negative_interaction",
        "p_sign",
        "p_wilcoxon",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)

    meta = {
        "run_map": str(run_map_path),
        "output": str(out_csv),
        "bootstrap_n": args.bootstrap_n,
        "seed": args.seed,
        "missing": missing,
    }
    meta_path = out_dir / "cross_vs_same_interaction_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(out_csv)
    print(meta_path)


if __name__ == "__main__":
    main()
