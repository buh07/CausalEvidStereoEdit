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
from stereacl.interventions import (
    make_direction_injection_at_position_hook,
    make_direction_injection_hook,
    make_direction_projection_at_position_hook,
    make_direction_projection_hook,
)
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 16: full 2x2 inject/remove matrix for asymmetry diagnosis."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="If set, edits are applied at prediction_position only (causal-local).",
    )
    parser.add_argument(
        "--exp1-run-dir",
        default="",
        help=(
            "Legacy shortcut: explicit Experiment 01 run directory used for both evaluation pairs "
            "and direction vectors. Prefer --eval-exp1-run-dir/--directions-exp1-run-dir."
        ),
    )
    parser.add_argument(
        "--eval-exp1-run-dir",
        default="",
        help=(
            "Experiment 01 run directory supplying aligned pairs and train/test split "
            "for heldout evaluation."
        ),
    )
    parser.add_argument(
        "--directions-exp1-run-dir",
        default="",
        help="Experiment 01 run directory supplying directions_layerwise.npz.",
    )
    parser.add_argument(
        "--emit-pair-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit per-pair condition deltas and primary contrasts for downstream diagnostics.",
    )
    parser.add_argument(
        "--emit-occupancy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emit prediction-position direction-occupancy summaries (h·d) with matched "
            "random-direction baselines for each heldout pair."
        ),
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


def _compose(
    first: Callable[[torch.Tensor], torch.Tensor] | None,
    second: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if first is None:
        return second

    def _c(x: torch.Tensor) -> torch.Tensor:
        return second(first(x))

    return _c


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


def _build_residual_patch(
    *,
    pair: AlignedPair,
    directions: dict[tuple[str, int], np.ndarray],
    device: torch.device,
    mode: str,  # "remove" or "inject"
    position_only: bool,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    axis = pair.pair.axis
    pos = pair.prediction_position
    for (dir_axis, layer), direction_np in directions.items():
        if dir_axis != axis:
            continue
        idx = layer - 1
        d = torch.tensor(direction_np, device=device, dtype=torch.float32)
        if mode == "remove":
            hook = (
                make_direction_projection_at_position_hook(pos, d)
                if position_only
                else make_direction_projection_hook(d)
            )
        else:
            hook = (
                make_direction_injection_at_position_hook(pos, d, alpha=1.0)
                if position_only
                else make_direction_injection_hook(d, alpha=1.0)
            )
        patch_map[idx] = _compose(patch_map.get(idx), hook)
    return patch_map


def _margin_for_text(
    *,
    bundle,
    text: str,
    pair: AlignedPair,
    max_length: int,
    residual_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
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


def _prediction_hidden_states(
    *,
    bundle,
    text: str,
    pair: AlignedPair,
    max_length: int,
) -> dict[int, torch.Tensor]:
    encoded = encode_text(bundle.tokenizer, text, bundle.device, max_length)
    cap = forward_with_component_capture(
        model=bundle.model,
        encoded_inputs=encoded,
        output_hidden_states=True,
        capture_attention=False,
        capture_mlp=False,
    )
    pos = pair.prediction_position
    if not cap.hidden_states:
        return {}
    out: dict[int, torch.Tensor] = {}
    for layer_idx, hs in enumerate(cap.hidden_states[1:], start=1):
        if pos < hs.shape[1]:
            out[layer_idx] = hs[0, pos, :].detach().float()
    return out


def main() -> None:
    args = parse_args()
    ctx = start_run("16", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        # Resolution order:
        # 1) explicit eval/direction dirs, 2) legacy --exp1-run-dir, 3) latest model-matched Exp01.
        if args.eval_exp1_run_dir:
            eval_exp1_dir = Path(args.eval_exp1_run_dir)
        elif args.exp1_run_dir:
            eval_exp1_dir = Path(args.exp1_run_dir)
        else:
            eval_exp1_dir = _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=[
                    "artifacts/aligned_pairs.jsonl",
                    "artifacts/train_test_split.json",
                ],
                model_name=args.model,
            )
        if args.directions_exp1_run_dir:
            directions_exp1_dir = Path(args.directions_exp1_run_dir)
        elif args.exp1_run_dir:
            directions_exp1_dir = Path(args.exp1_run_dir)
        else:
            directions_exp1_dir = eval_exp1_dir

        aligned_pairs = _load_aligned_pairs(eval_exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((eval_exp1_dir / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]
        directions = load_directions_npz(directions_exp1_dir / "artifacts" / "directions_layerwise.npz")

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs_path,
            {
                "eval_exp1_run_dir": str(eval_exp1_dir),
                "directions_exp1_run_dir": str(directions_exp1_dir),
                "heldout_pairs": len(heldout),
                "directions_loaded": len(directions),
                "position_only": args.position_only,
                "emit_occupancy": bool(args.emit_occupancy),
            },
        )
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(
                ctx,
                metrics={
                    "heldout_pairs": len(heldout),
                    "directions_loaded": len(directions),
                    "dry_run": True,
                },
            )
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        rng = np.random.default_rng(args.seed)

        baseline_stereo: dict[str, float] = {}
        baseline_anti: dict[str, float] = {}
        for pair in heldout:
            m_stereo = _margin_for_text(
                bundle=bundle,
                text=pair.pair.stereotype_text,
                pair=pair,
                max_length=args.max_length,
                residual_patch_map=None,
            )
            m_anti = _margin_for_text(
                bundle=bundle,
                text=pair.pair.antistereotype_text,
                pair=pair,
                max_length=args.max_length,
                residual_patch_map=None,
            )
            if m_stereo is not None:
                baseline_stereo[pair.pair.pair_id] = float(m_stereo)
            if m_anti is not None:
                baseline_anti[pair.pair.pair_id] = float(m_anti)

        conditions = [
            ("remove_on_stereo", "stereo", "remove"),
            ("remove_on_anti", "anti", "remove"),
            ("inject_on_stereo", "stereo", "inject"),
            ("inject_on_anti", "anti", "inject"),
        ]
        rows: list[dict[str, Any]] = []
        condition_pair_diffs_score: dict[str, dict[str, float]] = {}
        condition_pair_diffs_margin: dict[str, dict[str, float]] = {}

        for condition_name, base_kind, mode in conditions:
            pair_ids: list[str] = []
            base_vals: list[float] = []
            edit_vals: list[float] = []
            for pair in heldout:
                pair_id = pair.pair.pair_id
                base_map = baseline_stereo if base_kind == "stereo" else baseline_anti
                base_val = base_map.get(pair_id)
                if base_val is None:
                    continue
                text = pair.pair.stereotype_text if base_kind == "stereo" else pair.pair.antistereotype_text
                residual_patch = _build_residual_patch(
                    pair=pair,
                    directions=directions,
                    device=bundle.device,
                    mode=mode,
                    position_only=args.position_only,
                )
                if not residual_patch:
                    continue
                edited = _margin_for_text(
                    bundle=bundle,
                    text=text,
                    pair=pair,
                    max_length=args.max_length,
                    residual_patch_map=residual_patch,
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

            # Room-to-move diagnostics for asymmetric base distributions.
            room_down = score_base
            room_up = 1.0 - score_base
            norm_delta_down = (score_edit - score_base) / room_down if room_down > 1e-12 else float("nan")
            norm_delta_up = (score_edit - score_base) / room_up if room_up > 1e-12 else float("nan")

            score_pair_diff_map = {
                pair_id: float(s)
                for pair_id, s in zip(pair_ids, score_diffs.tolist())
            }
            margin_pair_diff_map = {
                pair_id: float(m)
                for pair_id, m in zip(pair_ids, margin_diffs.tolist())
            }
            condition_pair_diffs_score[condition_name] = score_pair_diff_map
            condition_pair_diffs_margin[condition_name] = margin_pair_diff_map

            rows.append(
                {
                    "condition": condition_name,
                    "base_distribution": base_kind,
                    "position_only": bool(args.position_only),
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
                    "room_to_move_down": round(room_down, 8),
                    "room_to_move_up": round(room_up, 8),
                    "normalized_delta_by_down_room": _rounded(norm_delta_down),
                    "normalized_delta_by_up_room": _rounded(norm_delta_up),
                }
            )

        _apply_fdr(rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        out_path = ctx.tables_dir / "asymmetry_2x2_matrix.csv"
        write_csv(
            out_path,
            rows,
            fieldnames=[
                "condition",
                "base_distribution",
                "position_only",
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
                "room_to_move_down",
                "room_to_move_up",
                "normalized_delta_by_down_room",
                "normalized_delta_by_up_room",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="2x2 inject/remove asymmetry matrix.")

        contrast_specs = [
            ("primary_inject_anti_minus_remove_stereo", "inject_on_anti", "remove_on_stereo"),
            ("same_base_stereo_inject_minus_remove", "inject_on_stereo", "remove_on_stereo"),
            ("same_base_anti_inject_minus_remove", "inject_on_anti", "remove_on_anti"),
        ]
        contrast_rows: list[dict[str, Any]] = []
        for contrast_name, inject_key, remove_key in contrast_specs:
            inj_score_map = condition_pair_diffs_score.get(inject_key, {})
            rem_score_map = condition_pair_diffs_score.get(remove_key, {})
            inj_margin_map = condition_pair_diffs_margin.get(inject_key, {})
            rem_margin_map = condition_pair_diffs_margin.get(remove_key, {})
            common_pair_ids = sorted(set(inj_score_map) & set(rem_score_map) & set(inj_margin_map) & set(rem_margin_map))
            if not common_pair_ids:
                continue
            score_contrast = np.array(
                [inj_score_map[pair_id] - rem_score_map[pair_id] for pair_id in common_pair_ids],
                dtype=float,
            )
            margin_contrast = np.array(
                [inj_margin_map[pair_id] - rem_margin_map[pair_id] for pair_id in common_pair_ids],
                dtype=float,
            )
            score_ci = bootstrap_mean_ci(score_contrast, n_resamples=args.bootstrap_n, rng=rng)
            margin_ci = bootstrap_mean_ci(margin_contrast, n_resamples=args.bootstrap_n, rng=rng)
            p_score, _, _ = paired_sign_test(score_contrast)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_contrast)
            contrast_rows.append(
                {
                    "contrast": contrast_name,
                    "inject_condition": inject_key,
                    "remove_condition": remove_key,
                    "n_pairs": len(common_pair_ids),
                    "mean_score_contrast": round(float(np.mean(score_contrast)), 8),
                    "mean_score_contrast_ci_low": _rounded(score_ci.ci_low),
                    "mean_score_contrast_ci_high": _rounded(score_ci.ci_high),
                    "mean_margin_contrast": round(float(np.mean(margin_contrast)), 8),
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
        contrast_path = ctx.tables_dir / "asymmetry_contrast.csv"
        write_csv(
            contrast_path,
            contrast_rows,
            fieldnames=[
                "contrast",
                "inject_condition",
                "remove_condition",
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
            description="Paired asymmetry contrasts (inject minus remove).",
        )

        if args.emit_pair_level:
            pair_rows: list[dict[str, Any]] = []
            cond_score_names = sorted(condition_pair_diffs_score.keys())
            cond_margin_names = sorted(condition_pair_diffs_margin.keys())
            for pair in heldout:
                pair_id = pair.pair.pair_id
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "source": pair.pair.source,
                    "axis": pair.pair.axis,
                    "baseline_margin_stereo": _rounded(baseline_stereo.get(pair_id)),
                    "baseline_margin_anti": _rounded(baseline_anti.get(pair_id)),
                }
                for cond in cond_score_names:
                    row[f"{cond}_score_delta"] = _rounded(condition_pair_diffs_score.get(cond, {}).get(pair_id))
                for cond in cond_margin_names:
                    row[f"{cond}_margin_delta"] = _rounded(condition_pair_diffs_margin.get(cond, {}).get(pair_id))

                inj_s = condition_pair_diffs_score.get("inject_on_anti", {}).get(pair_id)
                rem_s = condition_pair_diffs_score.get("remove_on_stereo", {}).get(pair_id)
                inj_m = condition_pair_diffs_margin.get("inject_on_anti", {}).get(pair_id)
                rem_m = condition_pair_diffs_margin.get("remove_on_stereo", {}).get(pair_id)
                row["primary_score_contrast"] = _rounded(None if inj_s is None or rem_s is None else float(inj_s - rem_s))
                row["primary_margin_contrast"] = _rounded(None if inj_m is None or rem_m is None else float(inj_m - rem_m))
                pair_rows.append(row)

            pair_fields = [
                "pair_id",
                "source",
                "axis",
                "baseline_margin_stereo",
                "baseline_margin_anti",
            ]
            pair_fields += [f"{c}_score_delta" for c in cond_score_names]
            pair_fields += [f"{c}_margin_delta" for c in cond_margin_names]
            pair_fields += ["primary_score_contrast", "primary_margin_contrast"]

            pair_path = ctx.tables_dir / "asymmetry_pair_deltas.csv"
            write_csv(pair_path, pair_rows, fieldnames=pair_fields)
            ctx.register_artifact(
                pair_path,
                artifact_type="table",
                description="Per-pair condition deltas and primary inject-minus-remove contrasts.",
            )

        occupancy_pair_rows = 0
        occupancy_layer_rows = 0
        if args.emit_occupancy:
            rng_occ = np.random.default_rng(args.seed + 9107)
            dir_unit: dict[tuple[str, int], torch.Tensor] = {}
            rand_unit: dict[tuple[str, int], torch.Tensor] = {}
            for key, d_np in directions.items():
                d_vec = torch.tensor(d_np, device=bundle.device, dtype=torch.float32)
                d_norm = float(torch.linalg.norm(d_vec).item())
                if not np.isfinite(d_norm) or d_norm <= 0.0:
                    continue
                d_hat = d_vec / d_norm
                dir_unit[key] = d_hat
                r = torch.tensor(rng_occ.standard_normal(d_vec.shape[0]), device=bundle.device, dtype=torch.float32)
                r_norm = float(torch.linalg.norm(r).item())
                if not np.isfinite(r_norm) or r_norm <= 0.0:
                    continue
                rand_unit[key] = r / r_norm

            occ_pair_rows: list[dict[str, Any]] = []
            occ_layer_rows: list[dict[str, Any]] = []
            for pair in heldout:
                axis = pair.pair.axis
                pair_id = pair.pair.pair_id
                base_variants = [
                    ("stereo", pair.pair.stereotype_text),
                    ("anti", pair.pair.antistereotype_text),
                ]
                for base_kind, text in base_variants:
                    hs_by_layer = _prediction_hidden_states(
                        bundle=bundle,
                        text=text,
                        pair=pair,
                        max_length=args.max_length,
                    )
                    true_proj_vals: list[float] = []
                    rand_proj_vals: list[float] = []
                    for (dir_axis, layer), d_hat in dir_unit.items():
                        if dir_axis != axis:
                            continue
                        h = hs_by_layer.get(layer)
                        if h is None:
                            continue
                        r_hat = rand_unit.get((dir_axis, layer))
                        if r_hat is None:
                            continue
                        true_proj = float(torch.dot(h.to(bundle.device), d_hat).item())
                        rand_proj = float(torch.dot(h.to(bundle.device), r_hat).item())
                        true_proj_vals.append(true_proj)
                        rand_proj_vals.append(rand_proj)
                        occ_layer_rows.append(
                            {
                                "pair_id": pair_id,
                                "source": pair.pair.source,
                                "axis": axis,
                                "base_kind": base_kind,
                                "layer": int(layer),
                                "true_proj": _rounded(true_proj),
                                "abs_true_proj": _rounded(abs(true_proj)),
                                "random_proj": _rounded(rand_proj),
                                "abs_random_proj": _rounded(abs(rand_proj)),
                            }
                        )

                    if not true_proj_vals:
                        continue
                    true_arr = np.array(true_proj_vals, dtype=float)
                    rand_arr = np.array(rand_proj_vals, dtype=float)
                    remove_score = condition_pair_diffs_score.get("remove_on_stereo", {}).get(pair_id, float("nan"))
                    remove_margin = condition_pair_diffs_margin.get("remove_on_stereo", {}).get(pair_id, float("nan"))
                    primary_score = float("nan")
                    primary_margin = float("nan")
                    inj_s = condition_pair_diffs_score.get("inject_on_anti", {}).get(pair_id)
                    rem_s = condition_pair_diffs_score.get("remove_on_stereo", {}).get(pair_id)
                    inj_m = condition_pair_diffs_margin.get("inject_on_anti", {}).get(pair_id)
                    rem_m = condition_pair_diffs_margin.get("remove_on_stereo", {}).get(pair_id)
                    if inj_s is not None and rem_s is not None:
                        primary_score = float(inj_s - rem_s)
                    if inj_m is not None and rem_m is not None:
                        primary_margin = float(inj_m - rem_m)
                    occ_pair_rows.append(
                        {
                            "pair_id": pair_id,
                            "source": pair.pair.source,
                            "axis": axis,
                            "base_kind": base_kind,
                            "n_layers_used": int(true_arr.size),
                            "mean_true_proj": _rounded(float(np.mean(true_arr))),
                            "mean_abs_true_proj": _rounded(float(np.mean(np.abs(true_arr)))),
                            "mean_random_proj": _rounded(float(np.mean(rand_arr))),
                            "mean_abs_random_proj": _rounded(float(np.mean(np.abs(rand_arr)))),
                            "mean_abs_true_minus_random": _rounded(float(np.mean(np.abs(true_arr)) - np.mean(np.abs(rand_arr)))),
                            "remove_on_stereo_score_delta": _rounded(remove_score if np.isfinite(remove_score) else None),
                            "remove_on_stereo_margin_delta": _rounded(remove_margin if np.isfinite(remove_margin) else None),
                            "primary_score_contrast": _rounded(primary_score if np.isfinite(primary_score) else None),
                            "primary_margin_contrast": _rounded(primary_margin if np.isfinite(primary_margin) else None),
                        }
                    )

            occ_pair_path = ctx.tables_dir / "asymmetry_occupancy_pair_summary.csv"
            occ_layer_path = ctx.tables_dir / "asymmetry_occupancy_layerwise.csv"
            write_csv(
                occ_pair_path,
                occ_pair_rows,
                fieldnames=[
                    "pair_id",
                    "source",
                    "axis",
                    "base_kind",
                    "n_layers_used",
                    "mean_true_proj",
                    "mean_abs_true_proj",
                    "mean_random_proj",
                    "mean_abs_random_proj",
                    "mean_abs_true_minus_random",
                    "remove_on_stereo_score_delta",
                    "remove_on_stereo_margin_delta",
                    "primary_score_contrast",
                    "primary_margin_contrast",
                ],
            )
            write_csv(
                occ_layer_path,
                occ_layer_rows,
                fieldnames=[
                    "pair_id",
                    "source",
                    "axis",
                    "base_kind",
                    "layer",
                    "true_proj",
                    "abs_true_proj",
                    "random_proj",
                    "abs_random_proj",
                ],
            )
            ctx.register_artifact(
                occ_pair_path,
                artifact_type="table",
                description="Per-pair prediction-position occupancy summaries (true direction vs random baseline).",
            )
            ctx.register_artifact(
                occ_layer_path,
                artifact_type="table",
                description="Per-layer prediction-position occupancy values (true direction vs random baseline).",
            )
            occupancy_pair_rows = len(occ_pair_rows)
            occupancy_layer_rows = len(occ_layer_rows)

        complete_run(
            ctx,
            metrics={
                "rows": len(rows),
                "contrast_rows": len(contrast_rows),
                "heldout_pairs": len(heldout),
                "position_only": bool(args.position_only),
                "pair_rows": len(heldout) if args.emit_pair_level else 0,
                "occupancy_pair_rows": occupancy_pair_rows,
                "occupancy_layer_rows": occupancy_layer_rows,
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
