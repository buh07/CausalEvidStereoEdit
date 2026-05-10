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
from stereacl.attention_heads import build_attention_projection_specs, make_attention_head_zero_hook
from stereacl.data import ContrastPair
from stereacl.interventions import make_zero_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


STEREOSET_SOURCE = "stereoset_intrasentence"
CROWS_SOURCE = "crows_pairs"


CONDITIONS = [
    ("stereoset_to_stereoset", STEREOSET_SOURCE, STEREOSET_SOURCE),
    ("stereoset_to_crows", STEREOSET_SOURCE, CROWS_SOURCE),
    ("crows_to_crows", CROWS_SOURCE, CROWS_SOURCE),
    ("crows_to_stereoset", CROWS_SOURCE, STEREOSET_SOURCE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 15: cross-dataset component-transfer 2x2 matrix."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--heldout-pairs",
        type=int,
        default=120,
        help="Per-target-source cap for heldout evaluation pairs.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--promoters-only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument("--exp2-stereoset-run-dir", default="")
    parser.add_argument("--exp2-crows-run-dir", default="")
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


def _load_aligned_pairs(path: Path) -> list[AlignedPair]:
    pairs: list[AlignedPair] = []
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
            pairs.append(
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
    return pairs


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def _load_top_components_by_axis(
    exp2_run_dir: Path,
    top_k: int,
    promoters_only: bool,
) -> dict[str, list[tuple[str, int, str, int | None]]]:
    path = exp2_run_dir / "tables" / "component_dla_scores.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    out: dict[str, list[tuple[str, int, str, int | None]]] = {}
    for axis, group in df.groupby("axis"):
        g = group
        if promoters_only and "mean_dla_score" in g.columns:
            g = g[g["mean_dla_score"] > 0]
        g = g.sort_values("mean_abs_dla_score", ascending=False).head(top_k)
        comps: list[tuple[str, int, str, int | None]] = []
        for _, row in g.iterrows():
            layer = int(row["layer"])
            comps.append(
                (
                    str(row["component_type"]),
                    layer,
                    str(row["component_id"]) if "component_id" in row.index and not pd.isna(row["component_id"]) else f"L{layer}",
                    _optional_int(row["head_index"]) if "head_index" in row.index else None,
                )
            )
        out[str(axis)] = comps
    return out


def _is_attention_component_type(component_type: str) -> bool:
    return component_type.startswith("attention")


def _compose(
    first: Callable[[torch.Tensor], torch.Tensor] | None,
    second: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if first is None:
        return second

    def _c(x: torch.Tensor) -> torch.Tensor:
        return second(first(x))

    return _c


def _forward_logits_with_patches(
    model,
    encoded_inputs: dict[str, torch.Tensor],
    attention_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
    mlp_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
    preproj_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
    head_specs: dict[int, Any],
) -> torch.Tensor:
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    try:
        if preproj_patch_map:
            for layer_idx, patch_hook in preproj_patch_map.items():
                spec = head_specs.get(layer_idx)
                if spec is None:
                    continue
                module = spec.projection_module

                def _make_hook(hook_fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
                    def _hook(_module, inputs: tuple[torch.Tensor, ...]):
                        if not inputs:
                            return None
                        patched = hook_fn(inputs[0])
                        if len(inputs) == 1:
                            return (patched,)
                        return (patched, *inputs[1:])

                    return _hook

                hooks.append(module.register_forward_pre_hook(_make_hook(patch_hook)))

        cap = forward_with_component_capture(
            model,
            encoded_inputs,
            output_hidden_states=False,
            capture_attention=bool(attention_patch_map),
            capture_mlp=bool(mlp_patch_map),
            attention_patch_map=attention_patch_map if attention_patch_map else None,
            mlp_patch_map=mlp_patch_map if mlp_patch_map else None,
        )
        return cap.logits
    finally:
        for handle in hooks:
            handle.remove()


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


def _cap_pairs_by_source(pairs: list[AlignedPair], per_source_cap: int) -> list[AlignedPair]:
    if per_source_cap <= 0:
        return pairs
    out: list[AlignedPair] = []
    counts: dict[str, int] = {}
    for pair in pairs:
        src = pair.pair.source
        count = counts.get(src, 0)
        if count >= per_source_cap:
            continue
        counts[src] = count + 1
        out.append(pair)
    return out


def main() -> None:
    args = parse_args()
    ctx = start_run("15", parameters=vars(args), project_root=PROJECT_ROOT)
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
        exp2_ss_dir = (
            Path(args.exp2_stereoset_run_dir)
            if args.exp2_stereoset_run_dir
            else _latest_run_dir(
                "02_component_dla",
                required_relpaths=["tables/component_dla_scores.csv"],
                model_name=args.model,
            )
        )
        exp2_cr_dir = (
            Path(args.exp2_crows_run_dir)
            if args.exp2_crows_run_dir
            else _latest_run_dir(
                "02_component_dla",
                required_relpaths=["tables/component_dla_scores.csv"],
                model_name=args.model,
            )
        )

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs_path,
            {
                "exp1_run_dir": str(exp1_dir),
                "exp2_stereoset_run_dir": str(exp2_ss_dir),
                "exp2_crows_run_dir": str(exp2_cr_dir),
                "top_k": args.top_k,
                "promoters_only": args.promoters_only,
            },
        )
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text())
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        heldout = [p for p in heldout if p.pair.source in {STEREOSET_SOURCE, CROWS_SOURCE}]
        heldout = _cap_pairs_by_source(heldout, args.heldout_pairs)

        top_ss = _load_top_components_by_axis(exp2_ss_dir, top_k=args.top_k, promoters_only=args.promoters_only)
        top_cr = _load_top_components_by_axis(exp2_cr_dir, top_k=args.top_k, promoters_only=args.promoters_only)

        if args.dry_run:
            complete_run(
                ctx,
                metrics={
                    "heldout_pairs": len(heldout),
                    "axes_stereoset_rank": len(top_ss),
                    "axes_crows_rank": len(top_cr),
                    "dry_run": True,
                },
            )
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        head_specs = build_attention_projection_specs(bundle.model)

        baseline_by_pair: dict[str, float] = {}
        for pair in heldout:
            pos = pair.prediction_position
            encoded = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)
            with torch.no_grad():
                cap = forward_with_component_capture(
                    bundle.model,
                    encoded,
                    output_hidden_states=False,
                    capture_attention=False,
                    capture_mlp=False,
                )
            if pos < cap.logits.shape[1]:
                baseline_by_pair[pair.pair.pair_id] = compute_score_from_logits(
                    cap.logits,
                    position=pos,
                    pos_token=pair.stereo_token,
                    neg_token=pair.anti_token,
                )

        rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(args.seed)
        condition_pool: dict[str, dict[str, list[float] | set[str]]] = {}

        ranking_map = {
            STEREOSET_SOURCE: top_ss,
            CROWS_SOURCE: top_cr,
        }

        for condition_name, rank_source, target_source in CONDITIONS:
            condition_pool.setdefault(
                condition_name,
                {
                    "axes": set(),
                    "score_base": [],
                    "score_abl": [],
                    "score_diff": [],
                    "margin_base": [],
                    "margin_abl": [],
                    "margin_diff": [],
                },
            )
            top_by_axis = ranking_map[rank_source]
            target_pairs = [p for p in heldout if p.pair.source == target_source]
            axes = sorted({p.pair.axis for p in target_pairs} & set(top_by_axis.keys()))
            for axis in axes:
                sites = top_by_axis.get(axis, [])
                if not sites:
                    continue

                axis_pairs = [p for p in target_pairs if p.pair.axis == axis]
                paired_base: list[float] = []
                paired_abl: list[float] = []
                for pair in axis_pairs:
                    base_val = baseline_by_pair.get(pair.pair.pair_id)
                    if base_val is None:
                        continue
                    pos = pair.prediction_position
                    encoded = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)

                    attn_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
                    mlp_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
                    preproj_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}

                    for (ct, layer, _cid, head_index) in sites:
                        idx = layer - 1
                        h = make_zero_position_hook(pos)
                        if ct == "attention_head" and head_index is not None and idx in head_specs:
                            try:
                                head_hook = make_attention_head_zero_hook(
                                    spec=head_specs[idx],
                                    position=pos,
                                    head_index=head_index,
                                )
                            except ValueError:
                                continue
                            preproj_patch[idx] = _compose(preproj_patch.get(idx), head_hook)
                        elif _is_attention_component_type(ct):
                            attn_patch[idx] = _compose(attn_patch.get(idx), h)
                        else:
                            mlp_patch[idx] = _compose(mlp_patch.get(idx), h)

                    logits = _forward_logits_with_patches(
                        model=bundle.model,
                        encoded_inputs=encoded,
                        attention_patch_map=attn_patch,
                        mlp_patch_map=mlp_patch,
                        preproj_patch_map=preproj_patch,
                        head_specs=head_specs,
                    )
                    if pos >= logits.shape[1]:
                        continue
                    paired_base.append(base_val)
                    paired_abl.append(
                        compute_score_from_logits(
                            logits,
                            position=pos,
                            pos_token=pair.stereo_token,
                            neg_token=pair.anti_token,
                        )
                    )

                if not paired_abl:
                    continue

                arr_base = np.array(paired_base, dtype=float)
                arr_abl = np.array(paired_abl, dtype=float)

                score_base = float(np.mean(arr_base > 0))
                score_abl = float(np.mean(arr_abl > 0))
                score_delta = score_abl - score_base

                margin_base = float(np.mean(arr_base))
                margin_abl = float(np.mean(arr_abl))
                margin_delta = margin_abl - margin_base

                score_pair_diffs = (arr_abl > 0).astype(float) - (arr_base > 0).astype(float)
                margin_pair_diffs = arr_abl - arr_base

                pooled = condition_pool[condition_name]
                cast_axes = pooled["axes"]
                if isinstance(cast_axes, set):
                    cast_axes.add(axis)
                for v in (arr_base > 0).astype(float).tolist():
                    pooled["score_base"].append(float(v))  # type: ignore[index]
                for v in (arr_abl > 0).astype(float).tolist():
                    pooled["score_abl"].append(float(v))  # type: ignore[index]
                for v in score_pair_diffs.tolist():
                    pooled["score_diff"].append(float(v))  # type: ignore[index]
                for v in arr_base.tolist():
                    pooled["margin_base"].append(float(v))  # type: ignore[index]
                for v in arr_abl.tolist():
                    pooled["margin_abl"].append(float(v))  # type: ignore[index]
                for v in margin_pair_diffs.tolist():
                    pooled["margin_diff"].append(float(v))  # type: ignore[index]

                score_ci = bootstrap_mean_ci(score_pair_diffs, n_resamples=args.bootstrap_n, rng=rng)
                margin_ci = bootstrap_mean_ci(margin_pair_diffs, n_resamples=args.bootstrap_n, rng=rng)
                p_score_sign, _, _ = paired_sign_test(score_pair_diffs)
                p_margin_w, _ = wilcoxon_signed_rank_safe(margin_pair_diffs)

                rows.append(
                    {
                        "condition": condition_name,
                        "rank_source": rank_source,
                        "target_source": target_source,
                        "axis": axis,
                        "n_sites": len(sites),
                        "n_pairs": len(arr_abl),
                        "stereotype_score_baseline": round(score_base, 8),
                        "stereotype_score_ablated": round(score_abl, 8),
                        "stereotype_score_delta": round(score_delta, 8),
                        "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
                        "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
                        "mean_margin_baseline": round(margin_base, 8),
                        "mean_margin_ablated": round(margin_abl, 8),
                        "mean_margin_delta": round(margin_delta, 8),
                        "mean_margin_delta_ci_low": _rounded(margin_ci.ci_low),
                        "mean_margin_delta_ci_high": _rounded(margin_ci.ci_high),
                        "paired_p_score_sign": _rounded(p_score_sign),
                        "paired_p_margin_wilcoxon": _rounded(p_margin_w),
                        "q_score_sign": "",
                        "q_margin_wilcoxon": "",
                        "transfer_efficiency_score": "",
                        "transfer_efficiency_margin": "",
                    }
                )

        _apply_fdr(rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        # Transfer efficiency relative to within-source baseline condition.
        within_lookup: dict[tuple[str, str], tuple[float, float]] = {}
        for row in rows:
            if row["rank_source"] == row["target_source"]:
                within_lookup[(row["rank_source"], row["axis"])] = (
                    float(row["stereotype_score_delta"]),
                    float(row["mean_margin_delta"]),
                )

        for row in rows:
            key = (row["rank_source"], row["axis"])
            if key not in within_lookup:
                continue
            base_score_delta, base_margin_delta = within_lookup[key]
            cur_score_delta = float(row["stereotype_score_delta"])
            cur_margin_delta = float(row["mean_margin_delta"])
            if abs(base_score_delta) > 1e-12:
                row["transfer_efficiency_score"] = _rounded(cur_score_delta / base_score_delta)
            if abs(base_margin_delta) > 1e-12:
                row["transfer_efficiency_margin"] = _rounded(cur_margin_delta / base_margin_delta)

        out_path = ctx.tables_dir / "cross_dataset_transfer_matrix.csv"
        write_csv(
            out_path,
            rows,
            fieldnames=[
                "condition",
                "rank_source",
                "target_source",
                "axis",
                "n_sites",
                "n_pairs",
                "stereotype_score_baseline",
                "stereotype_score_ablated",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_margin_baseline",
                "mean_margin_ablated",
                "mean_margin_delta",
                "mean_margin_delta_ci_low",
                "mean_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
                "transfer_efficiency_score",
                "transfer_efficiency_margin",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Cross-dataset component transfer matrix.")

        # Condition-level aggregates.
        summary_rows: list[dict[str, Any]] = []
        if rows:
            df = pd.DataFrame(rows)
            for condition, group in df.groupby("condition"):
                pooled = condition_pool.get(str(condition), {})
                score_base = np.array(pooled.get("score_base", []), dtype=float)
                score_abl = np.array(pooled.get("score_abl", []), dtype=float)
                score_diff = np.array(pooled.get("score_diff", []), dtype=float)
                margin_base = np.array(pooled.get("margin_base", []), dtype=float)
                margin_abl = np.array(pooled.get("margin_abl", []), dtype=float)
                margin_diff = np.array(pooled.get("margin_diff", []), dtype=float)

                if score_diff.size > 0:
                    score_ci = bootstrap_mean_ci(score_diff, n_resamples=args.bootstrap_n, rng=rng)
                    p_score, _, _ = paired_sign_test(score_diff)
                    score_base_mean = float(np.mean(score_base))
                    score_abl_mean = float(np.mean(score_abl))
                    score_delta_mean = float(np.mean(score_diff))
                else:
                    score_ci = bootstrap_mean_ci(np.array([float("nan")]), n_resamples=1, rng=rng)
                    p_score = float("nan")
                    score_base_mean = float("nan")
                    score_abl_mean = float("nan")
                    score_delta_mean = float("nan")

                if margin_diff.size > 0:
                    margin_ci = bootstrap_mean_ci(margin_diff, n_resamples=args.bootstrap_n, rng=rng)
                    p_margin, _ = wilcoxon_signed_rank_safe(margin_diff)
                    margin_base_mean = float(np.mean(margin_base))
                    margin_abl_mean = float(np.mean(margin_abl))
                    margin_delta_mean = float(np.mean(margin_diff))
                else:
                    margin_ci = bootstrap_mean_ci(np.array([float("nan")]), n_resamples=1, rng=rng)
                    p_margin = float("nan")
                    margin_base_mean = float("nan")
                    margin_abl_mean = float("nan")
                    margin_delta_mean = float("nan")

                axes_set = pooled.get("axes", set())
                axes_count = len(axes_set) if isinstance(axes_set, set) else int(group["axis"].nunique())
                summary_rows.append(
                    {
                        "condition": str(condition),
                        "rank_source": str(group["rank_source"].iloc[0]),
                        "target_source": str(group["target_source"].iloc[0]),
                        "axes_count": axes_count,
                        "n_pairs": int(score_diff.size),
                        "stereotype_score_baseline": _rounded(score_base_mean),
                        "stereotype_score_ablated": _rounded(score_abl_mean),
                        "stereotype_score_delta": _rounded(score_delta_mean),
                        "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
                        "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
                        "mean_margin_baseline": _rounded(margin_base_mean),
                        "mean_margin_ablated": _rounded(margin_abl_mean),
                        "mean_margin_delta": _rounded(margin_delta_mean),
                        "mean_margin_delta_ci_low": _rounded(margin_ci.ci_low),
                        "mean_margin_delta_ci_high": _rounded(margin_ci.ci_high),
                        "paired_p_score_sign": _rounded(p_score),
                        "paired_p_margin_wilcoxon": _rounded(p_margin),
                        "q_score_sign": "",
                        "q_margin_wilcoxon": "",
                        "mean_transfer_eff_score": _rounded(float(pd.to_numeric(group["transfer_efficiency_score"], errors="coerce").mean())),
                        "mean_transfer_eff_margin": _rounded(float(pd.to_numeric(group["transfer_efficiency_margin"], errors="coerce").mean())),
                    }
                )

        _apply_fdr(summary_rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(summary_rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")
        summary_path = ctx.tables_dir / "cross_dataset_transfer_condition_summary.csv"
        write_csv(
            summary_path,
            summary_rows,
            fieldnames=[
                "condition",
                "rank_source",
                "target_source",
                "axes_count",
                "n_pairs",
                "stereotype_score_baseline",
                "stereotype_score_ablated",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_margin_baseline",
                "mean_margin_ablated",
                "mean_margin_delta",
                "mean_margin_delta_ci_low",
                "mean_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
                "mean_transfer_eff_score",
                "mean_transfer_eff_margin",
            ],
        )
        ctx.register_artifact(summary_path, artifact_type="table", description="Condition-level transfer summary.")

        complete_run(
            ctx,
            metrics={
                "rows": len(rows),
                "conditions": len({r["condition"] for r in rows}),
                "axes_total": len({(r["condition"], r["axis"]) for r in rows}),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
