#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    p = argparse.ArgumentParser(description="Summarize Exp30 cross-dataset injection transfer matrix.")
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


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
        label = _label(model_name, payload)
        exp30_dir_raw = payload.get("exp30_run_dir", "")
        if not exp30_dir_raw:
            missing.append(f"{model_name}:missing_exp30")
            continue
        exp30_dir = Path(str(exp30_dir_raw))
        cpath = exp30_dir / "tables" / "cross_dataset_injection_transfer_condition_summary.csv"
        if not cpath.exists():
            missing.append(f"{model_name}:missing_summary")
            continue
        df = pd.read_csv(cpath)
        if df.empty:
            missing.append(f"{model_name}:empty_summary")
            continue
        for _, r in df.iterrows():
            rank_source = str(r.get("rank_source", ""))
            target_source = str(r.get("target_source", ""))
            rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "exp30_run_dir": str(exp30_dir),
                    "condition": str(r.get("condition", "")),
                    "rank_source": rank_source,
                    "target_source": target_source,
                    "condition_type": "within" if rank_source == target_source else "cross",
                    "n_pairs": int(_to_float(r.get("n_pairs", 0))),
                    "score_delta": _to_float(r.get("stereotype_score_delta", "")),
                    "score_ci_low": _to_float(r.get("stereotype_score_delta_ci_low", "")),
                    "score_ci_high": _to_float(r.get("stereotype_score_delta_ci_high", "")),
                    "p_score_sign": _to_float(r.get("paired_p_score_sign", "")),
                    "q_score_sign": _to_float(r.get("q_score_sign", "")),
                }
            )

    out_csv = out_dir / "exp30_injection_transfer_summary.csv"
    pd.DataFrame(
        rows,
        columns=[
            "model",
            "model_label",
            "exp30_run_dir",
            "condition",
            "rank_source",
            "target_source",
            "condition_type",
            "n_pairs",
            "score_delta",
            "score_ci_low",
            "score_ci_high",
            "p_score_sign",
            "q_score_sign",
        ],
    ).to_csv(out_csv, index=False)

    meta = {
        "run_map": str(run_map_path),
        "output": str(out_csv),
        "missing": missing,
    }
    meta_path = out_dir / "exp30_injection_transfer_summary_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(out_csv)
    print(meta_path)


if __name__ == "__main__":
    main()
