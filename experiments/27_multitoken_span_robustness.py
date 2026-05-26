#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.data import ContrastPair, build_contrast_pairs, deterministic_split_indices
from stereacl.interventions import make_direction_projection_at_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe


@dataclass(frozen=True)
class MultiTokenAlignedPair:
    pair: ContrastPair
    stereo_input_ids: list[int]
    anti_input_ids: list[int]
    stereo_span_tokens: list[int]
    anti_span_tokens: list[int]
    span_start: int
    span_end_stereo: int
    span_end_anti: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 27: multi-token span-level robustness extension for direction ablation."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=120)
    parser.add_argument("--per-source-limit", type=int, default=2500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-span-len", type=int, default=3)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--exp1-run-dir", default="")
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


def _find_diff_span(tokens_a: list[int], tokens_b: list[int]) -> tuple[int, int, int, int] | None:
    min_len = min(len(tokens_a), len(tokens_b))
    start = 0
    while start < min_len and tokens_a[start] == tokens_b[start]:
        start += 1
    if start == len(tokens_a) and start == len(tokens_b):
        return None
    end_a = len(tokens_a) - 1
    end_b = len(tokens_b) - 1
    while end_a >= start and end_b >= start and tokens_a[end_a] == tokens_b[end_b]:
        end_a -= 1
        end_b -= 1
    return start, end_a, start, end_b


def _align_multitoken_pairs(
    pairs: list[ContrastPair],
    tokenizer,
    max_span_len: int,
) -> tuple[list[MultiTokenAlignedPair], dict[str, int]]:
    kept: list[MultiTokenAlignedPair] = []
    stats = {
        "input_pairs": len(pairs),
        "dropped_non_diff": 0,
        "dropped_single_token": 0,
        "dropped_unequal_span": 0,
        "dropped_span_too_long": 0,
        "dropped_bad_pos": 0,
        "kept_pairs": 0,
    }
    for pair in pairs:
        s_ids = tokenizer(pair.stereotype_text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        a_ids = tokenizer(pair.antistereotype_text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        diff = _find_diff_span(s_ids, a_ids)
        if diff is None:
            stats["dropped_non_diff"] += 1
            continue
        s0, s1, a0, a1 = diff
        s_span = s_ids[s0 : s1 + 1]
        a_span = a_ids[a0 : a1 + 1]
        if len(s_span) <= 1 or len(a_span) <= 1:
            stats["dropped_single_token"] += 1
            continue
        if len(s_span) != len(a_span) or s0 != a0:
            stats["dropped_unequal_span"] += 1
            continue
        if len(s_span) > max_span_len:
            stats["dropped_span_too_long"] += 1
            continue
        if s0 <= 0:
            stats["dropped_bad_pos"] += 1
            continue
        kept.append(
            MultiTokenAlignedPair(
                pair=pair,
                stereo_input_ids=s_ids,
                anti_input_ids=a_ids,
                stereo_span_tokens=s_span,
                anti_span_tokens=a_span,
                span_start=s0,
                span_end_stereo=s1,
                span_end_anti=a1,
            )
        )
    stats["kept_pairs"] = len(kept)
    return kept, stats


def _compose(
    first: Callable[[torch.Tensor], torch.Tensor] | None,
    second: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if first is None:
        return second

    def _c(x: torch.Tensor) -> torch.Tensor:
        return second(first(x))

    return _c


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
    p_vals = [_to_float_or_nan(row.get(p_col, "")) for row in rows]
    q_vals = benjamini_hochberg(p_vals)
    for i, q in enumerate(q_vals):
        rows[i][q_col] = _rounded(q)


def _span_margin(
    *,
    bundle,
    pair: MultiTokenAlignedPair,
    max_length: int,
    patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
) -> float | None:
    encoded = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, max_length)
    cap = forward_with_component_capture(
        model=bundle.model,
        encoded_inputs=encoded,
        output_hidden_states=False,
        capture_attention=False,
        capture_mlp=False,
        residual_patch_map=patch_map,
    )
    vals: list[float] = []
    for k, (tok_s, tok_a) in enumerate(zip(pair.stereo_span_tokens, pair.anti_span_tokens, strict=False)):
        pred_pos = pair.span_start + k - 1
        if pred_pos < 0 or pred_pos >= cap.logits.shape[1]:
            return None
        vals.append(
            compute_score_from_logits(
                cap.logits,
                position=pred_pos,
                pos_token=int(tok_s),
                neg_token=int(tok_a),
            )
        )
    if not vals:
        return None
    return float(np.mean(vals))


def _build_span_patch(
    *,
    pair: MultiTokenAlignedPair,
    directions: dict[tuple[str, int], np.ndarray],
    device: torch.device,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    axis = pair.pair.axis
    pred_positions = [pair.span_start + k - 1 for k in range(len(pair.stereo_span_tokens))]
    for (d_axis, layer), d_np in directions.items():
        if d_axis != axis:
            continue
        idx = layer - 1
        d = torch.tensor(d_np, device=device, dtype=torch.float32)
        for pos in pred_positions:
            h = make_direction_projection_at_position_hook(pos, d)
            patch[idx] = _compose(patch.get(idx), h)
    return patch


def main() -> None:
    args = parse_args()
    ctx = start_run("27", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = (
            Path(args.exp1_run_dir)
            if args.exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=["artifacts/directions_layerwise.npz"],
                model_name=args.model,
            )
        )
        directions = load_directions_npz(exp1_dir / "artifacts" / "directions_layerwise.npz")

        pairs = build_contrast_pairs(
            include_stereoset=True,
            include_crows=True,
            include_seegull=True,
            per_source_limit=args.per_source_limit,
        )

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        mt_pairs, mt_stats = _align_multitoken_pairs(pairs, bundle.tokenizer, max_span_len=args.max_span_len)

        train_idx, test_idx = deterministic_split_indices(len(mt_pairs), test_fraction=0.2, seed=args.seed)
        heldout = [mt_pairs[int(i)] for i in test_idx if 0 <= int(i) < len(mt_pairs)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        refs = {
            "exp1_run_dir": str(exp1_dir),
            "directions": len(directions),
            "max_span_len": args.max_span_len,
            "heldout_pairs": len(heldout),
            "alignment_stats": mt_stats,
        }
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **refs})
            return

        base_vals: list[float] = []
        ablated_vals: list[float] = []
        pair_rows: list[dict[str, Any]] = []

        for pair in heldout:
            base = _span_margin(bundle=bundle, pair=pair, max_length=args.max_length, patch_map=None)
            if base is None:
                continue
            patch = _build_span_patch(pair=pair, directions=directions, device=bundle.device)
            if not patch:
                continue
            abl = _span_margin(bundle=bundle, pair=pair, max_length=args.max_length, patch_map=patch)
            if abl is None:
                continue
            base_vals.append(base)
            ablated_vals.append(abl)
            pair_rows.append(
                {
                    "pair_id": pair.pair.pair_id,
                    "source": pair.pair.source,
                    "axis": pair.pair.axis,
                    "span_len": len(pair.stereo_span_tokens),
                    "baseline_span_margin": _rounded(base),
                    "ablated_span_margin": _rounded(abl),
                    "span_margin_delta": _rounded(abl - base),
                    "baseline_span_score": int(base > 0),
                    "ablated_span_score": int(abl > 0),
                    "span_score_delta": int(abl > 0) - int(base > 0),
                }
            )

        rows: list[dict[str, Any]] = []
        if ablated_vals:
            arr_base = np.array(base_vals, dtype=float)
            arr_abl = np.array(ablated_vals, dtype=float)
            score_diff = (arr_abl > 0).astype(float) - (arr_base > 0).astype(float)
            margin_diff = arr_abl - arr_base
            rng = np.random.default_rng(args.seed)
            score_ci = bootstrap_mean_ci(score_diff, n_resamples=args.bootstrap_n, rng=rng)
            margin_ci = bootstrap_mean_ci(margin_diff, n_resamples=args.bootstrap_n, rng=rng)
            p_score, _, _ = paired_sign_test(score_diff)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_diff)

            rows.append(
                {
                    "condition": "direction_ablation_span_level",
                    "n_pairs": len(arr_abl),
                    "stereotype_score_baseline": round(float(np.mean(arr_base > 0)), 8),
                    "stereotype_score_ablated": round(float(np.mean(arr_abl > 0)), 8),
                    "stereotype_score_delta": round(float(np.mean(score_diff)), 8),
                    "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
                    "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
                    "mean_span_margin_baseline": round(float(np.mean(arr_base)), 8),
                    "mean_span_margin_ablated": round(float(np.mean(arr_abl)), 8),
                    "mean_span_margin_delta": round(float(np.mean(margin_diff)), 8),
                    "mean_span_margin_delta_ci_low": _rounded(margin_ci.ci_low),
                    "mean_span_margin_delta_ci_high": _rounded(margin_ci.ci_high),
                    "paired_p_score_sign": _rounded(p_score),
                    "paired_p_margin_wilcoxon": _rounded(p_margin),
                    "q_score_sign": "",
                    "q_margin_wilcoxon": "",
                }
            )
        _apply_fdr(rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        pair_path = ctx.tables_dir / "multitoken_span_pairs.csv"
        write_csv(
            pair_path,
            pair_rows,
            fieldnames=[
                "pair_id",
                "source",
                "axis",
                "span_len",
                "baseline_span_margin",
                "ablated_span_margin",
                "span_margin_delta",
                "baseline_span_score",
                "ablated_span_score",
                "span_score_delta",
            ],
        )
        ctx.register_artifact(pair_path, artifact_type="table", description="Per-pair multi-token span deltas.")

        out_path = ctx.tables_dir / "multitoken_span_summary.csv"
        write_csv(
            out_path,
            rows,
            fieldnames=[
                "condition",
                "n_pairs",
                "stereotype_score_baseline",
                "stereotype_score_ablated",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_span_margin_baseline",
                "mean_span_margin_ablated",
                "mean_span_margin_delta",
                "mean_span_margin_delta_ci_low",
                "mean_span_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Exp27 span-level summary.")

        align_path = ctx.tables_dir / "multitoken_alignment_stats.csv"
        write_csv(
            align_path,
            [
                {"metric": k, "value": v}
                for k, v in mt_stats.items()
            ],
            fieldnames=["metric", "value"],
        )
        ctx.register_artifact(
            align_path,
            artifact_type="table",
            description="Multi-token alignment retention statistics.",
        )

        complete_run(
            ctx,
            metrics={
                "heldout_pairs": len(heldout),
                "evaluated_pairs": len(pair_rows),
                "rows": len(rows),
                "kept_multitoken_pairs": mt_stats.get("kept_pairs", 0),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
