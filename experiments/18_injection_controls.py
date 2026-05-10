#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.interventions import make_direction_injection_at_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 18: injection specificity controls (true vs random vs shuffled-axis).")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--heldout-pairs", type=int, default=120)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=81)
    p.add_argument("--exp1-run-dir", default="")
    p.add_argument("--eval-sources", default="", help="Optional comma-separated source filter.")
    p.add_argument("--axes", default="", help="Optional comma-separated axis filter.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _parse_csv_set(raw: str) -> set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


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
        if required_relpaths and any(not (run_dir / rel).exists() for rel in required_relpaths):
            continue
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
    p_vals = [_to_float_or_nan(r.get(p_col, "")) for r in rows]
    q_vals = benjamini_hochberg(p_vals)
    for i, q in enumerate(q_vals):
        rows[i][q_col] = _rounded(q)


def _margin_for_text(
    *,
    bundle,
    text: str,
    pair: AlignedPair,
    max_length: int,
    residual_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
) -> float | None:
    encoded = encode_text(bundle.tokenizer, text, bundle.device, max_length)
    cap = forward_with_component_capture(
        model=bundle.model,
        encoded_inputs=encoded,
        output_hidden_states=False,
        capture_attention=False,
        capture_mlp=False,
        residual_patch_map=residual_patch_map,
    )
    pos = pair.prediction_position
    if pos >= cap.logits.shape[1]:
        return None
    return compute_score_from_logits(
        cap.logits,
        position=pos,
        pos_token=pair.stereo_token,
        neg_token=pair.anti_token,
    )


def _axis_shuffle_map(directions: dict[tuple[str, int], np.ndarray]) -> dict[tuple[str, int], tuple[str, int]]:
    by_layer: dict[int, list[str]] = {}
    for axis, layer in directions:
        by_layer.setdefault(layer, [])
        if axis not in by_layer[layer]:
            by_layer[layer].append(axis)
    out: dict[tuple[str, int], tuple[str, int]] = {}
    for layer, axes in by_layer.items():
        axes_sorted = sorted(axes)
        if len(axes_sorted) < 2:
            continue
        for i, axis in enumerate(axes_sorted):
            mapped = axes_sorted[(i + 1) % len(axes_sorted)]
            out[(axis, layer)] = (mapped, layer)
    return out


def _build_patch_map(
    *,
    pair: AlignedPair,
    directions: dict[tuple[str, int], np.ndarray],
    device: torch.device,
    rng: np.random.Generator,
    mode: str,  # true | random | shuffled
    axis_shuffle: dict[tuple[str, int], tuple[str, int]],
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    axis = pair.pair.axis
    pos = pair.prediction_position
    for (dir_axis, layer), direction_np in directions.items():
        if dir_axis != axis:
            continue
        idx = layer - 1
        if mode == "true":
            use = direction_np
        elif mode == "random":
            use = rng.standard_normal(direction_np.shape[0]).astype(np.float32)
        else:
            mapped_key = axis_shuffle.get((axis, layer))
            if mapped_key is None or mapped_key not in directions:
                continue
            use = directions[mapped_key]
        d = torch.tensor(use, device=device, dtype=torch.float32)
        hook = make_direction_injection_at_position_hook(pos, d, alpha=1.0)
        patch_map[idx] = _compose(patch_map.get(idx), hook)
    return patch_map


def main() -> None:
    args = parse_args()
    ctx = start_run("18", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = (
            Path(args.exp1_run_dir)
            if args.exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=[
                    "artifacts/aligned_pairs.jsonl",
                    "artifacts/train_test_split.json",
                    "artifacts/directions_layerwise.npz",
                ],
                model_name=args.model,
            )
        )
        eval_sources = _parse_csv_set(args.eval_sources)
        axis_filter = _parse_csv_set(args.axes)

        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if eval_sources:
            heldout = [p for p in heldout if p.pair.source in eval_sources]
        if axis_filter:
            heldout = [p for p in heldout if p.pair.axis in axis_filter]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        directions = load_directions_npz(exp1_dir / "artifacts" / "directions_layerwise.npz")
        axis_shuffle = _axis_shuffle_map(directions)

        refs = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs,
            {
                "exp1_run_dir": str(exp1_dir),
                "heldout_pairs": len(heldout),
                "directions_loaded": len(directions),
                "eval_sources": sorted(eval_sources),
                "axes": sorted(axis_filter),
            },
        )
        ctx.register_artifact(refs, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"heldout_pairs": len(heldout), "dry_run": True})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        rng = np.random.default_rng(args.seed)

        baseline_anti: dict[str, float] = {}
        for pair in heldout:
            m = _margin_for_text(
                bundle=bundle,
                text=pair.pair.antistereotype_text,
                pair=pair,
                max_length=args.max_length,
                residual_patch_map=None,
            )
            if m is not None:
                baseline_anti[pair.pair.pair_id] = float(m)

        condition_specs = [
            ("inject_true_on_anti", "true"),
            ("inject_random_norm_on_anti", "random"),
            ("inject_shuffled_axis_on_anti", "shuffled"),
        ]

        rows: list[dict[str, Any]] = []
        pair_diffs_score: dict[str, dict[str, float]] = {}
        pair_diffs_margin: dict[str, dict[str, float]] = {}

        for cond_name, mode in condition_specs:
            pair_ids: list[str] = []
            base_vals: list[float] = []
            edit_vals: list[float] = []
            for pair in heldout:
                pair_id = pair.pair.pair_id
                base_val = baseline_anti.get(pair_id)
                if base_val is None:
                    continue
                patch = _build_patch_map(
                    pair=pair,
                    directions=directions,
                    device=bundle.device,
                    rng=rng,
                    mode=mode,
                    axis_shuffle=axis_shuffle,
                )
                if not patch:
                    continue
                edited = _margin_for_text(
                    bundle=bundle,
                    text=pair.pair.antistereotype_text,
                    pair=pair,
                    max_length=args.max_length,
                    residual_patch_map=patch,
                )
                if edited is None:
                    continue
                pair_ids.append(pair_id)
                base_vals.append(base_val)
                edit_vals.append(float(edited))

            if not edit_vals:
                continue

            arr_base = np.array(base_vals, dtype=float)
            arr_edit = np.array(edit_vals, dtype=float)
            score_base = float(np.mean(arr_base > 0))
            score_edit = float(np.mean(arr_edit > 0))
            score_diffs = (arr_edit > 0).astype(float) - (arr_base > 0).astype(float)
            margin_diffs = arr_edit - arr_base

            score_ci = bootstrap_mean_ci(score_diffs, n_resamples=args.bootstrap_n, rng=rng)
            margin_ci = bootstrap_mean_ci(margin_diffs, n_resamples=args.bootstrap_n, rng=rng)
            p_score, _, _ = paired_sign_test(score_diffs)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_diffs)

            pair_diffs_score[cond_name] = {pid: float(d) for pid, d in zip(pair_ids, score_diffs.tolist())}
            pair_diffs_margin[cond_name] = {pid: float(d) for pid, d in zip(pair_ids, margin_diffs.tolist())}

            rows.append(
                {
                    "condition": cond_name,
                    "n_pairs": len(arr_edit),
                    "stereotype_score_baseline": round(score_base, 8),
                    "stereotype_score_intervened": round(score_edit, 8),
                    "stereotype_score_delta": round(score_edit - score_base, 8),
                    "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
                    "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
                    "mean_margin_baseline": round(float(np.mean(arr_base)), 8),
                    "mean_margin_intervened": round(float(np.mean(arr_edit)), 8),
                    "mean_margin_delta": round(float(np.mean(margin_diffs)), 8),
                    "mean_margin_delta_ci_low": _rounded(margin_ci.ci_low),
                    "mean_margin_delta_ci_high": _rounded(margin_ci.ci_high),
                    "paired_p_score_sign": _rounded(p_score),
                    "paired_p_margin_wilcoxon": _rounded(p_margin),
                    "q_score_sign": "",
                    "q_margin_wilcoxon": "",
                }
            )

        _apply_fdr(rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        out = ctx.tables_dir / "injection_controls.csv"
        write_csv(
            out,
            rows,
            fieldnames=[
                "condition",
                "n_pairs",
                "stereotype_score_baseline",
                "stereotype_score_intervened",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_margin_baseline",
                "mean_margin_intervened",
                "mean_margin_delta",
                "mean_margin_delta_ci_low",
                "mean_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
            ],
        )
        ctx.register_artifact(out, artifact_type="table", description="Injection controls (true/random/shuffled).")

        contrast_rows: list[dict[str, Any]] = []
        contrast_specs = [
            ("true_minus_random", "inject_true_on_anti", "inject_random_norm_on_anti"),
            ("true_minus_shuffled", "inject_true_on_anti", "inject_shuffled_axis_on_anti"),
        ]
        for name, a, b in contrast_specs:
            a_score = pair_diffs_score.get(a, {})
            b_score = pair_diffs_score.get(b, {})
            a_margin = pair_diffs_margin.get(a, {})
            b_margin = pair_diffs_margin.get(b, {})
            common = sorted(set(a_score) & set(b_score) & set(a_margin) & set(b_margin))
            if not common:
                continue
            score_contrast = np.array([a_score[k] - b_score[k] for k in common], dtype=float)
            margin_contrast = np.array([a_margin[k] - b_margin[k] for k in common], dtype=float)

            score_ci = bootstrap_mean_ci(score_contrast, n_resamples=args.bootstrap_n, rng=rng)
            margin_ci = bootstrap_mean_ci(margin_contrast, n_resamples=args.bootstrap_n, rng=rng)
            p_score, _, _ = paired_sign_test(score_contrast)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_contrast)
            contrast_rows.append(
                {
                    "contrast": name,
                    "a_condition": a,
                    "b_condition": b,
                    "n_pairs": len(common),
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

        contrast_out = ctx.tables_dir / "injection_control_contrasts.csv"
        write_csv(
            contrast_out,
            contrast_rows,
            fieldnames=[
                "contrast",
                "a_condition",
                "b_condition",
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
        ctx.register_artifact(contrast_out, artifact_type="table", description="True-vs-control injection contrasts.")

        complete_run(
            ctx,
            metrics={
                "rows": len(rows),
                "contrast_rows": len(contrast_rows),
                "heldout_pairs": len(heldout),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
