#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 29: prompt-based calibration baseline for local-edit effect size context."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=291)
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument("--exp16-run-dir", default="")
    parser.add_argument(
        "--mitigation-prefix",
        default="Please continue the sentence in a respectful, non-stereotypical way: ",
    )
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
        run_dir = Path(payload["run_dir"])
        if required_relpaths and any(not (run_dir / rel).exists() for rel in required_relpaths):
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        raise FileNotFoundError(f"No completed run found for {experiment_slug}.")
    return best[1]


def _load_aligned_pairs(path: Path) -> list[AlignedPair]:
    out: list[AlignedPair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pair = ContrastPair(
                pair_id=row["pair_id"],
                source=row["source"],
                axis=row["axis"],
                stereotype_text=row["stereotype_text"],
                antistereotype_text=row["antistereotype_text"],
                metadata=row.get("metadata", {}),
            )
            out.append(
                AlignedPair(
                    pair=pair,
                    stereo_input_ids=row["stereo_input_ids"],
                    anti_input_ids=row["anti_input_ids"],
                    stereo_token=int(row["stereo_token"]),
                    anti_token=int(row["anti_token"]),
                    trait_token_position=int(row["trait_token_position"]),
                    prediction_position=int(row["prediction_position"]),
                    differing_span_stereo=tuple(row.get("differing_span_stereo", (0, 0))),  # type: ignore[arg-type]
                    differing_span_anti=tuple(row.get("differing_span_anti", (0, 0))),  # type: ignore[arg-type]
                )
            )
    return out


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


def _to_float_or_nan(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _apply_fdr(rows: list[dict[str, Any]], p_col: str, q_col: str) -> None:
    p_vals = [_to_float_or_nan(r.get(p_col, "")) for r in rows]
    q_vals = benjamini_hochberg(p_vals)
    for i, q in enumerate(q_vals):
        rows[i][q_col] = _rounded(q)


def _margin(bundle, text: str, pair: AlignedPair, max_length: int, pred_pos_shift: int = 0) -> float | None:
    encoded = encode_text(bundle.tokenizer, text, bundle.device, max_length)
    cap = forward_with_component_capture(
        model=bundle.model,
        encoded_inputs=encoded,
        output_hidden_states=False,
        capture_attention=False,
        capture_mlp=False,
    )
    pos = int(pair.prediction_position + pred_pos_shift)
    if pos < 0 or pos >= cap.logits.shape[1]:
        return None
    return compute_score_from_logits(
        cap.logits,
        position=pos,
        pos_token=pair.stereo_token,
        neg_token=pair.anti_token,
    )


def _row_for_condition(
    *,
    condition: str,
    base_vals: np.ndarray,
    prompt_vals: np.ndarray,
    bootstrap_n: int,
    seed: int,
) -> dict[str, Any]:
    score_diffs = (prompt_vals > 0).astype(float) - (base_vals > 0).astype(float)
    margin_diffs = prompt_vals - base_vals
    rng = np.random.default_rng(seed)
    score_ci = bootstrap_mean_ci(score_diffs, n_resamples=bootstrap_n, rng=rng)
    margin_ci = bootstrap_mean_ci(margin_diffs, n_resamples=bootstrap_n, rng=rng)
    p_score, _, _ = paired_sign_test(score_diffs)
    p_margin, _ = wilcoxon_signed_rank_safe(margin_diffs)
    return {
        "condition": condition,
        "n_pairs": len(base_vals),
        "stereotype_score_baseline": _rounded(float(np.mean(base_vals > 0))),
        "stereotype_score_prompt": _rounded(float(np.mean(prompt_vals > 0))),
        "stereotype_score_delta": _rounded(float(np.mean(score_diffs))),
        "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
        "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
        "mean_margin_baseline": _rounded(float(np.mean(base_vals))),
        "mean_margin_prompt": _rounded(float(np.mean(prompt_vals))),
        "mean_margin_delta": _rounded(float(np.mean(margin_diffs))),
        "mean_margin_delta_ci_low": _rounded(margin_ci.ci_low),
        "mean_margin_delta_ci_high": _rounded(margin_ci.ci_high),
        "paired_p_score_sign": _rounded(p_score),
        "paired_p_margin_wilcoxon": _rounded(p_margin),
        "q_score_sign": "",
        "q_margin_wilcoxon": "",
    }


def _read_exp16_summary(exp16_run_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    contrast_path = exp16_run_dir / "tables" / "asymmetry_contrast.csv"
    if contrast_path.exists():
        df = pd.read_csv(contrast_path)
        row = df[df["contrast"] == "primary_inject_anti_minus_remove_stereo"]
        if not row.empty:
            r = row.iloc[0]
            out["exp16_primary_score_contrast"] = float(r.get("mean_score_contrast", np.nan))
            out["exp16_primary_margin_contrast"] = float(r.get("mean_margin_contrast", np.nan))
    matrix_path = exp16_run_dir / "tables" / "asymmetry_2x2_matrix.csv"
    if matrix_path.exists():
        dfm = pd.read_csv(matrix_path)
        r_rem = dfm[dfm["condition"] == "remove_on_stereo"]
        r_inj = dfm[dfm["condition"] == "inject_on_anti"]
        if not r_rem.empty:
            out["exp16_remove_score_delta"] = float(r_rem.iloc[0].get("stereotype_score_delta", np.nan))
            out["exp16_remove_margin_delta"] = float(r_rem.iloc[0].get("mean_margin_delta", np.nan))
        if not r_inj.empty:
            out["exp16_inject_score_delta"] = float(r_inj.iloc[0].get("stereotype_score_delta", np.nan))
            out["exp16_inject_margin_delta"] = float(r_inj.iloc[0].get("mean_margin_delta", np.nan))
    return out


def main() -> None:
    args = parse_args()
    ctx = start_run("29", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = (
            Path(args.exp1_run_dir)
            if args.exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=["artifacts/aligned_pairs.jsonl", "artifacts/train_test_split.json"],
                model_name=args.model,
            )
        )
        exp16_dir = (
            Path(args.exp16_run_dir)
            if args.exp16_run_dir
            else _latest_run_dir(
                "16_asymmetry_matrix",
                required_relpaths=["tables/asymmetry_contrast.csv"],
                model_name=args.model,
            )
        )

        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = [int(i) for i in split.get("test_indices", [])]
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if args.heldout_pairs > 0 and len(heldout) > args.heldout_pairs:
            rng = np.random.default_rng(args.seed)
            pick = sorted(rng.choice(len(heldout), size=args.heldout_pairs, replace=False).tolist())
            heldout = [heldout[i] for i in pick]

        refs = {
            "exp1_run_dir": str(exp1_dir),
            "exp16_run_dir": str(exp16_dir),
            "heldout_pairs": len(heldout),
            "mitigation_prefix": args.mitigation_prefix,
        }
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **refs})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        prefix_ids = bundle.tokenizer(
            args.mitigation_prefix,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        pred_shift = int(len(prefix_ids))

        rows_by_condition: dict[str, dict[str, list[float]]] = {
            "prompt_on_stereo": {"base": [], "prompt": []},
            "prompt_on_anti": {"base": [], "prompt": []},
        }
        per_pair_score_delta: dict[str, dict[str, float]] = {
            "prompt_on_stereo": {},
            "prompt_on_anti": {},
        }
        per_pair_margin_delta: dict[str, dict[str, float]] = {
            "prompt_on_stereo": {},
            "prompt_on_anti": {},
        }
        pair_rows: list[dict[str, Any]] = []

        for pair in heldout:
            for cond, base_kind in [("prompt_on_stereo", "stereo"), ("prompt_on_anti", "anti")]:
                base_text = pair.pair.stereotype_text if base_kind == "stereo" else pair.pair.antistereotype_text
                prompt_text = f"{args.mitigation_prefix}{base_text}"
                base_margin = _margin(bundle, base_text, pair, args.max_length, pred_pos_shift=0)
                prompt_margin = _margin(bundle, prompt_text, pair, args.max_length, pred_pos_shift=pred_shift)
                if base_margin is None or prompt_margin is None:
                    continue
                rows_by_condition[cond]["base"].append(float(base_margin))
                rows_by_condition[cond]["prompt"].append(float(prompt_margin))
                per_pair_score_delta[cond][pair.pair.pair_id] = float(int(prompt_margin > 0) - int(base_margin > 0))
                per_pair_margin_delta[cond][pair.pair.pair_id] = float(prompt_margin - base_margin)
                pair_rows.append(
                    {
                        "pair_id": pair.pair.pair_id,
                        "axis": pair.pair.axis,
                        "source": pair.pair.source,
                        "condition": cond,
                        "baseline_margin": _rounded(base_margin),
                        "prompt_margin": _rounded(prompt_margin),
                        "margin_delta": _rounded(prompt_margin - base_margin),
                        "baseline_score": int(base_margin > 0),
                        "prompt_score": int(prompt_margin > 0),
                        "score_delta": int(prompt_margin > 0) - int(base_margin > 0),
                    }
                )

        summary_rows: list[dict[str, Any]] = []
        for condition in ["prompt_on_stereo", "prompt_on_anti"]:
            base = np.array(rows_by_condition[condition]["base"], dtype=float)
            prompt = np.array(rows_by_condition[condition]["prompt"], dtype=float)
            if base.size == 0:
                continue
            summary_rows.append(
                _row_for_condition(
                    condition=condition,
                    base_vals=base,
                    prompt_vals=prompt,
                    bootstrap_n=args.bootstrap_n,
                    seed=args.seed,
                )
            )

        _apply_fdr(summary_rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(summary_rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        # Matched prompt contrast analogous to Exp16 primary contrast:
        # (prompt-on-stereo delta) - (prompt-on-anti delta), paired by item.
        contrast_rows: list[dict[str, Any]] = []
        common_ids = sorted(
            set(per_pair_score_delta["prompt_on_stereo"])
            & set(per_pair_score_delta["prompt_on_anti"])
            & set(per_pair_margin_delta["prompt_on_stereo"])
            & set(per_pair_margin_delta["prompt_on_anti"])
        )
        if common_ids:
            score_contrast = np.array(
                [
                    per_pair_score_delta["prompt_on_stereo"][pid]
                    - per_pair_score_delta["prompt_on_anti"][pid]
                    for pid in common_ids
                ],
                dtype=float,
            )
            margin_contrast = np.array(
                [
                    per_pair_margin_delta["prompt_on_stereo"][pid]
                    - per_pair_margin_delta["prompt_on_anti"][pid]
                    for pid in common_ids
                ],
                dtype=float,
            )
            c_rng = np.random.default_rng(args.seed + 17)
            score_ci = bootstrap_mean_ci(score_contrast, n_resamples=args.bootstrap_n, rng=c_rng)
            margin_ci = bootstrap_mean_ci(margin_contrast, n_resamples=args.bootstrap_n, rng=c_rng)
            p_score, _, _ = paired_sign_test(score_contrast)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_contrast)
            contrast_rows.append(
                {
                    "contrast": "prompt_stereo_minus_prompt_anti",
                    "left_condition": "prompt_on_stereo",
                    "right_condition": "prompt_on_anti",
                    "n_pairs": len(common_ids),
                    "mean_score_contrast": _rounded(float(np.mean(score_contrast))),
                    "mean_score_contrast_ci_low": _rounded(score_ci.ci_low),
                    "mean_score_contrast_ci_high": _rounded(score_ci.ci_high),
                    "mean_margin_contrast": _rounded(float(np.mean(margin_contrast))),
                    "mean_margin_contrast_ci_low": _rounded(margin_ci.ci_low),
                    "mean_margin_contrast_ci_high": _rounded(margin_ci.ci_high),
                    "paired_p_score_sign": _rounded(p_score),
                    "paired_p_margin_wilcoxon": _rounded(p_margin),
                    "q_score_sign": "",
                    "q_margin_wilcoxon": "",
                }
            )
            _apply_fdr(contrast_rows, "paired_p_score_sign", "q_score_sign")
            _apply_fdr(contrast_rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        pair_path = ctx.tables_dir / "prompt_calibration_pairs.csv"
        write_csv(
            pair_path,
            pair_rows,
            fieldnames=[
                "pair_id",
                "axis",
                "source",
                "condition",
                "baseline_margin",
                "prompt_margin",
                "margin_delta",
                "baseline_score",
                "prompt_score",
                "score_delta",
            ],
        )
        ctx.register_artifact(pair_path, artifact_type="table", description="Per-pair prompt calibration deltas.")

        summary_path = ctx.tables_dir / "prompt_calibration_summary.csv"
        write_csv(
            summary_path,
            summary_rows,
            fieldnames=[
                "condition",
                "n_pairs",
                "stereotype_score_baseline",
                "stereotype_score_prompt",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_margin_baseline",
                "mean_margin_prompt",
                "mean_margin_delta",
                "mean_margin_delta_ci_low",
                "mean_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
            ],
        )
        ctx.register_artifact(summary_path, artifact_type="table", description="Prompt baseline calibration summary.")

        contrast_path = ctx.tables_dir / "prompt_calibration_contrast.csv"
        write_csv(
            contrast_path,
            contrast_rows,
            fieldnames=[
                "contrast",
                "left_condition",
                "right_condition",
                "n_pairs",
                "mean_score_contrast",
                "mean_score_contrast_ci_low",
                "mean_score_contrast_ci_high",
                "mean_margin_contrast",
                "mean_margin_contrast_ci_low",
                "mean_margin_contrast_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
            ],
        )
        ctx.register_artifact(
            contrast_path,
            artifact_type="table",
            description="Paired prompt matched contrast (prompt-on-stereo minus prompt-on-anti).",
        )

        exp16_vals = _read_exp16_summary(exp16_dir)
        calibration_rows: list[dict[str, Any]] = []
        for row in summary_rows:
            cond = str(row["condition"])
            s_delta = _to_float_or_nan(row.get("stereotype_score_delta", ""))
            m_delta = _to_float_or_nan(row.get("mean_margin_delta", ""))
            if np.isnan(s_delta) and np.isnan(m_delta):
                continue
            calibration_rows.append(
                {
                    "condition": cond,
                    "prompt_score_delta": _rounded(s_delta),
                    "prompt_margin_delta": _rounded(m_delta),
                    "exp16_remove_score_delta": _rounded(exp16_vals.get("exp16_remove_score_delta")),
                    "exp16_inject_score_delta": _rounded(exp16_vals.get("exp16_inject_score_delta")),
                    "exp16_primary_score_contrast": _rounded(exp16_vals.get("exp16_primary_score_contrast")),
                    "exp16_remove_margin_delta": _rounded(exp16_vals.get("exp16_remove_margin_delta")),
                    "exp16_inject_margin_delta": _rounded(exp16_vals.get("exp16_inject_margin_delta")),
                    "exp16_primary_margin_contrast": _rounded(exp16_vals.get("exp16_primary_margin_contrast")),
                    "abs_prompt_vs_abs_remove_score_ratio": _rounded(
                        abs(s_delta) / abs(exp16_vals.get("exp16_remove_score_delta", np.nan))
                        if np.isfinite(s_delta) and abs(exp16_vals.get("exp16_remove_score_delta", np.nan)) > 1e-12
                        else float("nan")
                    ),
                    "abs_prompt_vs_abs_inject_score_ratio": _rounded(
                        abs(s_delta) / abs(exp16_vals.get("exp16_inject_score_delta", np.nan))
                        if np.isfinite(s_delta) and abs(exp16_vals.get("exp16_inject_score_delta", np.nan)) > 1e-12
                        else float("nan")
                    ),
                }
            )

        cal_path = ctx.tables_dir / "prompt_vs_local_edit_calibration.csv"
        write_csv(
            cal_path,
            calibration_rows,
            fieldnames=[
                "condition",
                "prompt_score_delta",
                "prompt_margin_delta",
                "exp16_remove_score_delta",
                "exp16_inject_score_delta",
                "exp16_primary_score_contrast",
                "exp16_remove_margin_delta",
                "exp16_inject_margin_delta",
                "exp16_primary_margin_contrast",
                "abs_prompt_vs_abs_remove_score_ratio",
                "abs_prompt_vs_abs_inject_score_ratio",
            ],
        )
        ctx.register_artifact(
            cal_path,
            artifact_type="table",
            description="Prompt comparator deltas calibrated to local-edit effects from Exp16.",
        )

        complete_run(
            ctx,
            metrics={
                "heldout_pairs": len(heldout),
                "summary_rows": len(summary_rows),
                "contrast_rows": len(contrast_rows),
                "pair_rows": len(pair_rows),
                "pred_pos_shift_tokens": pred_shift,
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
