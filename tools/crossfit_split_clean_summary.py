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


def _label(model_name: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model_name, model_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize cross-fit split-clean diagnostics across fold seeds for ranking-dependent "
            "families (Exp14 sign reliability, Exp17 suppressor contamination)."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def _read_exp14_overall(run_dir: Path) -> dict[str, float]:
    fp = run_dir / "tables" / "sign_reliability_overall.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp)
    if df.empty:
        raise ValueError(f"Empty file: {fp}")
    r = df.iloc[0]
    return {
        "dla_rate": _to_float(r.get("dla_sign_agreement_rate", "")),
        "dla_q": _to_float(r.get("q_dla_sign_agreement", "")),
        "dla_n": _to_float(r.get("n_dla_sign_agreement", "")),
        "dla_k": _to_float(r.get("k_dla_sign_agreement", "")),
        "atp_rate": _to_float(r.get("atp_sign_agreement_rate", "")),
        "atp_q": _to_float(r.get("q_atp_sign_agreement", "")),
        "atp_n": _to_float(r.get("n_atp_sign_agreement", "")),
        "atp_k": _to_float(r.get("k_atp_sign_agreement", "")),
    }


def _read_exp17_overall(run_dir: Path) -> dict[str, float]:
    fp = run_dir / "tables" / "suppressor_contamination_overall.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp)
    if df.empty:
        raise ValueError(f"Empty file: {fp}")
    r = df.iloc[0]
    return {
        "suppressor_fraction": _to_float(r.get("causal_suppressor_fraction", "")),
        "n_selected": _to_float(r.get("n_selected", "")),
        "causal_suppressors": _to_float(r.get("causal_suppressors", "")),
        "mean_score_delta_over_selected": _to_float(r.get("mean_score_delta_over_selected", "")),
    }


def _summ(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)

        exp14_seed_runs = payload.get("exp14_crossfit_seed_runs", {})
        exp17_seed_runs = payload.get("exp17_crossfit_seed_runs", {})
        if not isinstance(exp14_seed_runs, dict):
            exp14_seed_runs = {}
        if not isinstance(exp17_seed_runs, dict):
            exp17_seed_runs = {}

        seeds = sorted({*exp14_seed_runs.keys(), *exp17_seed_runs.keys()}, key=lambda s: int(str(s)))
        if not seeds:
            missing.append(f"{model_name}:missing_crossfit_seeds")
            continue

        dla_rates: list[float] = []
        atp_rates: list[float] = []
        sup_fracs: list[float] = []

        for seed in seeds:
            row: dict[str, Any] = {
                "model": model_name,
                "model_label": label,
                "seed": int(seed),
                "exp14_run_dir": "",
                "exp17_run_dir": "",
                "dla_sign_rate": float("nan"),
                "dla_sign_q": float("nan"),
                "dla_sign_n": float("nan"),
                "atp_sign_rate": float("nan"),
                "atp_sign_q": float("nan"),
                "atp_sign_n": float("nan"),
                "suppressor_fraction": float("nan"),
                "n_selected": float("nan"),
                "mean_score_delta_over_selected": float("nan"),
            }

            e14_raw = exp14_seed_runs.get(seed, "")
            if e14_raw:
                e14 = Path(str(e14_raw))
                row["exp14_run_dir"] = str(e14)
                try:
                    v14 = _read_exp14_overall(e14)
                    row.update(
                        {
                            "dla_sign_rate": v14["dla_rate"],
                            "dla_sign_q": v14["dla_q"],
                            "dla_sign_n": v14["dla_n"],
                            "atp_sign_rate": v14["atp_rate"],
                            "atp_sign_q": v14["atp_q"],
                            "atp_sign_n": v14["atp_n"],
                        }
                    )
                    if np.isfinite(v14["dla_rate"]):
                        dla_rates.append(v14["dla_rate"])
                    if np.isfinite(v14["atp_rate"]):
                        atp_rates.append(v14["atp_rate"])
                except Exception as exc:
                    missing.append(f"{model_name}:seed={seed}:exp14:{exc}")
            else:
                missing.append(f"{model_name}:seed={seed}:missing_exp14")

            e17_raw = exp17_seed_runs.get(seed, "")
            if e17_raw:
                e17 = Path(str(e17_raw))
                row["exp17_run_dir"] = str(e17)
                try:
                    v17 = _read_exp17_overall(e17)
                    row.update(
                        {
                            "suppressor_fraction": v17["suppressor_fraction"],
                            "n_selected": v17["n_selected"],
                            "mean_score_delta_over_selected": v17["mean_score_delta_over_selected"],
                        }
                    )
                    if np.isfinite(v17["suppressor_fraction"]):
                        sup_fracs.append(v17["suppressor_fraction"])
                except Exception as exc:
                    missing.append(f"{model_name}:seed={seed}:exp17:{exc}")
            else:
                missing.append(f"{model_name}:seed={seed}:missing_exp17")

            fold_rows.append(row)

        dla_mean, dla_sd = _summ(dla_rates)
        atp_mean, atp_sd = _summ(atp_rates)
        sup_mean, sup_sd = _summ(sup_fracs)

        agg_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "n_folds": len(seeds),
                "fold_seeds": ",".join(str(int(s)) for s in seeds),
                "dla_sign_rate_mean": dla_mean,
                "dla_sign_rate_sd": dla_sd,
                "atp_sign_rate_mean": atp_mean,
                "atp_sign_rate_sd": atp_sd,
                "suppressor_fraction_mean": sup_mean,
                "suppressor_fraction_sd": sup_sd,
                "dla_status_crossfit": (
                    "below_chance"
                    if np.isfinite(dla_mean) and dla_mean < 0.5
                    else "at_or_above_chance"
                ),
            }
        )

    fold_path = out_dir / "crossfit_split_clean_fold_rows.csv"
    agg_path = out_dir / "crossfit_split_clean_summary.csv"
    meta_path = out_dir / "crossfit_split_clean_summary_meta.json"

    pd.DataFrame(
        fold_rows,
        columns=[
            "model",
            "model_label",
            "seed",
            "exp14_run_dir",
            "exp17_run_dir",
            "dla_sign_rate",
            "dla_sign_q",
            "dla_sign_n",
            "atp_sign_rate",
            "atp_sign_q",
            "atp_sign_n",
            "suppressor_fraction",
            "n_selected",
            "mean_score_delta_over_selected",
        ],
    ).to_csv(fold_path, index=False)

    pd.DataFrame(
        agg_rows,
        columns=[
            "model",
            "model_label",
            "n_folds",
            "fold_seeds",
            "dla_sign_rate_mean",
            "dla_sign_rate_sd",
            "atp_sign_rate_mean",
            "atp_sign_rate_sd",
            "suppressor_fraction_mean",
            "suppressor_fraction_sd",
            "dla_status_crossfit",
        ],
    ).to_csv(agg_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "fold_rows": str(fold_path),
        "summary": str(agg_path),
        "missing": missing,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(fold_path)
    print(agg_path)
    print(meta_path)


if __name__ == "__main__":
    main()
