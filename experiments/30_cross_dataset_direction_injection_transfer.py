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

STEREOSET_SOURCE = "stereoset_intrasentence"
CROWS_SOURCE = "crows_pairs"
CONDITIONS = [
    ("stereoset_to_stereoset", STEREOSET_SOURCE, STEREOSET_SOURCE),
    ("stereoset_to_crows", STEREOSET_SOURCE, CROWS_SOURCE),
    ("crows_to_crows", CROWS_SOURCE, CROWS_SOURCE),
    ("crows_to_stereoset", CROWS_SOURCE, STEREOSET_SOURCE),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 30: cross-dataset 2x2 transfer matrix for direction injection "
            "(positive-control intervention with strong within-source signal)."
        )
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--heldout-pairs", type=int, default=120, help="Per-target-source cap.")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=307)
    p.add_argument("--exp1-mixed-run-dir", default="")
    p.add_argument("--exp1-stereoset-run-dir", default="")
    p.add_argument("--exp1-crows-run-dir", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _latest_run_dir(experiment_slug: str, model_name: str, required_relpaths: list[str]) -> Path:
    root = PROJECT_ROOT / "results" / experiment_slug
    best: tuple[str, Path] | None = None
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        if payload.get("parameters", {}).get("model") != model_name:
            continue
        run_dir = Path(payload["run_dir"])
        if any(not (run_dir / rel).exists() for rel in required_relpaths):
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        raise FileNotFoundError(f"No completed {experiment_slug} run for model {model_name}.")
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
    p_vals = [_to_float_or_nan(r.get(p_col, "")) for r in rows]
    q_vals = benjamini_hochberg(p_vals)
    for i, q in enumerate(q_vals):
        rows[i][q_col] = _rounded(q)


def _compose(
    first: Callable[[torch.Tensor], torch.Tensor] | None,
    second: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if first is None:
        return second

    def _c(x: torch.Tensor) -> torch.Tensor:
        return second(first(x))

    return _c


def _cap_pairs_by_source(pairs: list[AlignedPair], per_source_cap: int) -> list[AlignedPair]:
    if per_source_cap <= 0:
        return pairs
    out: list[AlignedPair] = []
    counts: dict[str, int] = {}
    for pair in pairs:
        src = pair.pair.source
        c = counts.get(src, 0)
        if c >= per_source_cap:
            continue
        counts[src] = c + 1
        out.append(pair)
    return out


def _build_injection_patch(
    pair: AlignedPair,
    directions: dict[tuple[str, int], np.ndarray],
    device: torch.device,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    axis = pair.pair.axis
    pos = pair.prediction_position
    for (d_axis, layer), d_np in directions.items():
        if d_axis != axis:
            continue
        idx = layer - 1
        d = torch.tensor(d_np, device=device, dtype=torch.float32)
        h = make_direction_injection_at_position_hook(pos, d, alpha=1.0)
        patch[idx] = _compose(patch.get(idx), h)
    return patch


def _margin_for_text(
    *,
    bundle,
    text: str,
    pair: AlignedPair,
    max_length: int,
    patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
) -> float | None:
    encoded = encode_text(bundle.tokenizer, text, bundle.device, max_length)
    cap = forward_with_component_capture(
        model=bundle.model,
        encoded_inputs=encoded,
        output_hidden_states=False,
        capture_attention=False,
        capture_mlp=False,
        residual_patch_map=patch_map,
    )
    pos = pair.prediction_position
    if pos < 0 or pos >= cap.logits.shape[1]:
        return None
    return compute_score_from_logits(
        cap.logits,
        position=pos,
        pos_token=pair.stereo_token,
        neg_token=pair.anti_token,
    )


def main() -> None:
    args = parse_args()
    ctx = start_run("30", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_mixed = (
            Path(args.exp1_mixed_run_dir)
            if args.exp1_mixed_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                model_name=args.model,
                required_relpaths=["artifacts/aligned_pairs.jsonl", "artifacts/train_test_split.json"],
            )
        )
        exp1_ss = (
            Path(args.exp1_stereoset_run_dir)
            if args.exp1_stereoset_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                model_name=args.model,
                required_relpaths=["artifacts/directions_layerwise.npz"],
            )
        )
        exp1_cr = (
            Path(args.exp1_crows_run_dir)
            if args.exp1_crows_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                model_name=args.model,
                required_relpaths=["artifacts/directions_layerwise.npz"],
            )
        )

        aligned_pairs = _load_aligned_pairs(exp1_mixed / "artifacts" / "aligned_pairs.jsonl")
        split = json.loads((exp1_mixed / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = [int(i) for i in split.get("test_indices", [])]
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        heldout = [p for p in heldout if p.pair.source in {STEREOSET_SOURCE, CROWS_SOURCE}]
        heldout = _cap_pairs_by_source(heldout, args.heldout_pairs)

        directions_ss = load_directions_npz(exp1_ss / "artifacts" / "directions_layerwise.npz")
        directions_cr = load_directions_npz(exp1_cr / "artifacts" / "directions_layerwise.npz")
        directions_by_source = {STEREOSET_SOURCE: directions_ss, CROWS_SOURCE: directions_cr}

        dep = {
            "exp1_mixed_run_dir": str(exp1_mixed),
            "exp1_stereoset_run_dir": str(exp1_ss),
            "exp1_crows_run_dir": str(exp1_cr),
            "heldout_pairs": len(heldout),
            "condition_mode": "inject_on_anti",
        }
        dep_path = ctx.artifacts_dir / "dependencies.json"
        write_json(dep_path, dep)
        ctx.register_artifact(dep_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **dep})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        rng = np.random.default_rng(args.seed)

        # Cache baseline anti margins once.
        baseline_anti: dict[str, float] = {}
        for pair in heldout:
            m = _margin_for_text(
                bundle=bundle,
                text=pair.pair.antistereotype_text,
                pair=pair,
                max_length=args.max_length,
                patch_map=None,
            )
            if m is not None:
                baseline_anti[pair.pair.pair_id] = float(m)

        rows: list[dict[str, Any]] = []
        axis_buckets: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
        pair_rows: list[dict[str, Any]] = []
        for condition, rank_source, target_source in CONDITIONS:
            directions = directions_by_source[rank_source]
            pairs = [p for p in heldout if p.pair.source == target_source and p.pair.axis in {a for (a, _l) in directions}]
            if not pairs:
                continue

            base_vals: list[float] = []
            edit_vals: list[float] = []
            pids: list[str] = []
            axes_used: set[str] = set()
            for pair in pairs:
                pid = pair.pair.pair_id
                b = baseline_anti.get(pid)
                if b is None:
                    continue
                patch = _build_injection_patch(pair, directions, bundle.device)
                if not patch:
                    continue
                e = _margin_for_text(
                    bundle=bundle,
                    text=pair.pair.antistereotype_text,
                    pair=pair,
                    max_length=args.max_length,
                    patch_map=patch,
                )
                if e is None:
                    continue
                pids.append(pid)
                axes_used.add(pair.pair.axis)
                base_vals.append(float(b))
                edit_vals.append(float(e))
                pair_rows.append(
                    {
                        "condition": condition,
                        "rank_source": rank_source,
                        "target_source": target_source,
                        "pair_id": pid,
                        "axis": pair.pair.axis,
                        "baseline_margin_anti": _rounded(b),
                        "injected_margin_anti": _rounded(e),
                        "margin_delta": _rounded(e - b),
                        "score_delta": int(e > 0) - int(b > 0),
                    }
                )
                axis_key = (condition, rank_source, target_source, pair.pair.axis)
                bucket = axis_buckets.setdefault(axis_key, {"score_delta": [], "margin_delta": []})
                bucket["score_delta"].append(float(int(e > 0) - int(b > 0)))
                bucket["margin_delta"].append(float(e - b))

            if not edit_vals:
                continue
            arr_base = np.array(base_vals, dtype=float)
            arr_edit = np.array(edit_vals, dtype=float)
            score_diffs = (arr_edit > 0).astype(float) - (arr_base > 0).astype(float)
            margin_diffs = arr_edit - arr_base
            score_ci = bootstrap_mean_ci(score_diffs, n_resamples=args.bootstrap_n, rng=rng)
            margin_ci = bootstrap_mean_ci(margin_diffs, n_resamples=args.bootstrap_n, rng=rng)
            p_score, _, _ = paired_sign_test(score_diffs)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_diffs)
            rows.append(
                {
                    "condition": condition,
                    "rank_source": rank_source,
                    "target_source": target_source,
                    "axes_count": len(axes_used),
                    "n_pairs": len(arr_edit),
                    "stereotype_score_baseline": _rounded(float(np.mean(arr_base > 0))),
                    "stereotype_score_injected": _rounded(float(np.mean(arr_edit > 0))),
                    "stereotype_score_delta": _rounded(float(np.mean(score_diffs))),
                    "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
                    "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
                    "mean_margin_baseline": _rounded(float(np.mean(arr_base))),
                    "mean_margin_injected": _rounded(float(np.mean(arr_edit))),
                    "mean_margin_delta": _rounded(float(np.mean(margin_diffs))),
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

        axis_rows: list[dict[str, Any]] = []
        for (condition, rank_source, target_source, axis), bucket in sorted(axis_buckets.items()):
            score_diffs = np.array(bucket["score_delta"], dtype=float)
            margin_diffs = np.array(bucket["margin_delta"], dtype=float)
            if score_diffs.size == 0:
                continue
            score_ci = bootstrap_mean_ci(score_diffs, n_resamples=args.bootstrap_n, rng=rng)
            margin_ci = bootstrap_mean_ci(margin_diffs, n_resamples=args.bootstrap_n, rng=rng)
            p_score, _, _ = paired_sign_test(score_diffs)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_diffs)
            axis_rows.append(
                {
                    "condition": condition,
                    "rank_source": rank_source,
                    "target_source": target_source,
                    "axis": axis,
                    "n_pairs": int(score_diffs.size),
                    "stereotype_score_delta": _rounded(float(np.mean(score_diffs))),
                    "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
                    "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
                    "mean_margin_delta": _rounded(float(np.mean(margin_diffs))),
                    "mean_margin_delta_ci_low": _rounded(margin_ci.ci_low),
                    "mean_margin_delta_ci_high": _rounded(margin_ci.ci_high),
                    "paired_p_score_sign": _rounded(p_score),
                    "paired_p_margin_wilcoxon": _rounded(p_margin),
                    "q_score_sign": "",
                    "q_margin_wilcoxon": "",
                }
            )
        _apply_fdr(axis_rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(axis_rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        summary_path = ctx.tables_dir / "cross_dataset_injection_transfer_condition_summary.csv"
        write_csv(
            summary_path,
            rows,
            fieldnames=[
                "condition",
                "rank_source",
                "target_source",
                "axes_count",
                "n_pairs",
                "stereotype_score_baseline",
                "stereotype_score_injected",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_margin_baseline",
                "mean_margin_injected",
                "mean_margin_delta",
                "mean_margin_delta_ci_low",
                "mean_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
            ],
        )
        ctx.register_artifact(
            summary_path,
            artifact_type="table",
            description="Condition-level 2x2 cross-dataset transfer matrix for inject-on-anti.",
        )

        pair_path = ctx.tables_dir / "cross_dataset_injection_transfer_pairs.csv"
        write_csv(
            pair_path,
            pair_rows,
            fieldnames=[
                "condition",
                "rank_source",
                "target_source",
                "pair_id",
                "axis",
                "baseline_margin_anti",
                "injected_margin_anti",
                "margin_delta",
                "score_delta",
            ],
        )
        ctx.register_artifact(pair_path, artifact_type="table", description="Per-pair injection transfer deltas.")

        axis_path = ctx.tables_dir / "cross_dataset_injection_transfer_axis_summary.csv"
        write_csv(
            axis_path,
            axis_rows,
            fieldnames=[
                "condition",
                "rank_source",
                "target_source",
                "axis",
                "n_pairs",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_margin_delta",
                "mean_margin_delta_ci_low",
                "mean_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
            ],
        )
        ctx.register_artifact(axis_path, artifact_type="table", description="Axis-level Exp30 injection transfer deltas.")

        complete_run(
            ctx,
            metrics={
                "rows": len(rows),
                "pair_rows": len(pair_rows),
                "axis_rows": len(axis_rows),
                "heldout_pairs": len(heldout),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
