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
from stereacl.interventions import make_direction_projection_at_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair

SEE_SOURCE = "seegull_global_v2"
TARGET_AXIS = "nationality"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 19: leave-one-source-out (exclude SEEGeL) nationality sensitivity.")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=97)
    p.add_argument("--full-exp1-run-dir", default="")
    p.add_argument("--no-seegull-exp1-run-dir", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _latest_run_dir(
    experiment_slug: str,
    required_relpaths: list[str] | None = None,
    model_name: str | None = None,
    require_no_seegull: bool | None = None,
) -> Path:
    root = PROJECT_ROOT / "results" / experiment_slug
    candidates = sorted(root.glob("*/*/manifest.json"))
    best: tuple[str, Path] | None = None
    for manifest_path in candidates:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        params = payload.get("parameters", {})
        if model_name is not None and params.get("model") != model_name:
            continue
        if require_no_seegull is not None:
            if bool(params.get("no_seegull", False)) != require_no_seegull:
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


def _build_projection_patch(
    *,
    pair: AlignedPair,
    directions: dict[tuple[str, int], np.ndarray],
    device: torch.device,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    axis = pair.pair.axis
    pos = pair.prediction_position
    for (dir_axis, layer), direction_np in directions.items():
        if dir_axis != axis:
            continue
        idx = layer - 1
        d = torch.tensor(direction_np, device=device, dtype=torch.float32)
        hook = make_direction_projection_at_position_hook(pos, d)
        patch_map[idx] = _compose(patch_map.get(idx), hook)
    return patch_map


def _eval_direction_set(
    *,
    bundle,
    pairs: list[AlignedPair],
    directions: dict[tuple[str, int], np.ndarray],
    max_length: int,
    bootstrap_n: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    pair_ids: list[str] = []
    base_vals: list[float] = []
    edit_vals: list[float] = []
    for pair in pairs:
        base = _margin_for_text(
            bundle=bundle,
            text=pair.pair.stereotype_text,
            pair=pair,
            max_length=max_length,
            residual_patch_map=None,
        )
        if base is None:
            continue
        patch = _build_projection_patch(pair=pair, directions=directions, device=bundle.device)
        if not patch:
            continue
        edited = _margin_for_text(
            bundle=bundle,
            text=pair.pair.stereotype_text,
            pair=pair,
            max_length=max_length,
            residual_patch_map=patch,
        )
        if edited is None:
            continue
        pair_ids.append(pair.pair.pair_id)
        base_vals.append(float(base))
        edit_vals.append(float(edited))

    arr_base = np.array(base_vals, dtype=float)
    arr_edit = np.array(edit_vals, dtype=float)
    score_base = float(np.mean(arr_base > 0)) if arr_base.size else float("nan")
    score_edit = float(np.mean(arr_edit > 0)) if arr_edit.size else float("nan")
    score_diffs = (arr_edit > 0).astype(float) - (arr_base > 0).astype(float)
    margin_diffs = arr_edit - arr_base

    score_ci = bootstrap_mean_ci(score_diffs, n_resamples=bootstrap_n, rng=rng)
    margin_ci = bootstrap_mean_ci(margin_diffs, n_resamples=bootstrap_n, rng=rng)
    p_score, _, _ = paired_sign_test(score_diffs)
    p_margin, _ = wilcoxon_signed_rank_safe(margin_diffs)

    row = {
        "n_pairs": int(arr_edit.size),
        "stereotype_score_baseline": _rounded(score_base),
        "stereotype_score_ablated": _rounded(score_edit),
        "stereotype_score_delta": _rounded(score_edit - score_base),
        "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
        "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
        "mean_margin_baseline": _rounded(float(np.mean(arr_base))),
        "mean_margin_ablated": _rounded(float(np.mean(arr_edit))),
        "mean_margin_delta": _rounded(float(np.mean(margin_diffs))),
        "mean_margin_delta_ci_low": _rounded(margin_ci.ci_low),
        "mean_margin_delta_ci_high": _rounded(margin_ci.ci_high),
        "paired_p_score_sign": _rounded(p_score),
        "paired_p_margin_wilcoxon": _rounded(p_margin),
        "q_score_sign": "",
        "q_margin_wilcoxon": "",
    }
    return row, {pid: float(d) for pid, d in zip(pair_ids, score_diffs.tolist())}, {pid: float(d) for pid, d in zip(pair_ids, margin_diffs.tolist())}


def main() -> None:
    args = parse_args()
    ctx = start_run("19", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        full_exp1 = (
            Path(args.full_exp1_run_dir)
            if args.full_exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=["artifacts/aligned_pairs.jsonl", "artifacts/train_test_split.json", "artifacts/directions_layerwise.npz"],
                model_name=args.model,
                require_no_seegull=False,
            )
        )
        no_seegull_exp1 = (
            Path(args.no_seegull_exp1_run_dir)
            if args.no_seegull_exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=["artifacts/directions_layerwise.npz"],
                model_name=args.model,
                require_no_seegull=True,
            )
        )

        aligned_pairs = _load_aligned_pairs(full_exp1 / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((full_exp1 / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        heldout = [p for p in heldout if p.pair.source == SEE_SOURCE and p.pair.axis == TARGET_AXIS]

        full_dirs = load_directions_npz(full_exp1 / "artifacts" / "directions_layerwise.npz")
        no_seegull_dirs = load_directions_npz(no_seegull_exp1 / "artifacts" / "directions_layerwise.npz")

        refs = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs,
            {
                "full_exp1_run_dir": str(full_exp1),
                "no_seegull_exp1_run_dir": str(no_seegull_exp1),
                "heldout_pairs_seegull_nationality": len(heldout),
                "full_layers_nationality": sum(1 for (a, _l) in full_dirs if a == TARGET_AXIS),
                "no_seegull_layers_nationality": sum(1 for (a, _l) in no_seegull_dirs if a == TARGET_AXIS),
            },
        )
        ctx.register_artifact(refs, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"heldout_pairs": len(heldout), "dry_run": True})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        rng = np.random.default_rng(args.seed)

        rows: list[dict[str, Any]] = []
        full_row, full_score_map, full_margin_map = _eval_direction_set(
            bundle=bundle,
            pairs=heldout,
            directions=full_dirs,
            max_length=args.max_length,
            bootstrap_n=args.bootstrap_n,
            rng=rng,
        )
        full_row["condition"] = "full_training_sources"
        rows.append(full_row)

        loso_row, loso_score_map, loso_margin_map = _eval_direction_set(
            bundle=bundle,
            pairs=heldout,
            directions=no_seegull_dirs,
            max_length=args.max_length,
            bootstrap_n=args.bootstrap_n,
            rng=rng,
        )
        loso_row["condition"] = "exclude_seegull_training"
        rows.append(loso_row)

        _apply_fdr(rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        out = ctx.tables_dir / "seegull_loso_sensitivity.csv"
        write_csv(
            out,
            rows,
            fieldnames=[
                "condition",
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
            ],
        )
        ctx.register_artifact(out, artifact_type="table", description="SEEGeL nationality LO-SO sensitivity results.")

        common = sorted(set(full_score_map) & set(loso_score_map) & set(full_margin_map) & set(loso_margin_map))
        contrast_rows: list[dict[str, Any]] = []
        if common:
            score_contrast = np.array([full_score_map[k] - loso_score_map[k] for k in common], dtype=float)
            margin_contrast = np.array([full_margin_map[k] - loso_margin_map[k] for k in common], dtype=float)
            score_ci = bootstrap_mean_ci(score_contrast, n_resamples=args.bootstrap_n, rng=rng)
            margin_ci = bootstrap_mean_ci(margin_contrast, n_resamples=args.bootstrap_n, rng=rng)
            p_score, _, _ = paired_sign_test(score_contrast)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_contrast)
            contrast_rows.append(
                {
                    "contrast": "full_minus_exclude_seegull",
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

        contrast_out = ctx.tables_dir / "seegull_loso_contrast.csv"
        write_csv(
            contrast_out,
            contrast_rows,
            fieldnames=[
                "contrast",
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
        ctx.register_artifact(contrast_out, artifact_type="table", description="SEEGeL LO-SO contrast.")

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
