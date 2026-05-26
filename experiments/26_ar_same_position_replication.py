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

from stereacl.analysis import compute_score_from_logits, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.interventions import make_direction_injection_at_position_hook, make_direction_projection_at_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 26: AR same-position-valid prefix-difference replication with matched inject/remove."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=120)
    parser.add_argument("--train-pairs", type=int, default=360)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=261)
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument(
        "--prompt-variant",
        choices=["therefore_next_word", "so_next_word", "thus_next_word", "plain_suffix"],
        default="therefore_next_word",
        help="Deterministic prompt template variant for same-position AR scoring.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help=(
            "Optional custom suffix appended after the base sentence. If set, this overrides --prompt-variant."
        ),
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
            "Emit same-position prediction-token occupancy summaries (h·d) with matched random-direction baselines."
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


def _template_suffix(variant: str, custom_suffix: str) -> str:
    if custom_suffix.strip():
        return custom_suffix.strip()
    mapping = {
        "therefore_next_word": "Therefore the next word is",
        "so_next_word": "So the next word is",
        "thus_next_word": "Thus the next word is",
        "plain_suffix": "The next word is",
    }
    return mapping.get(variant, mapping["therefore_next_word"])


def _arm_prompt(pair: AlignedPair, base: str, suffix: str) -> str:
    # Prefix-difference AR condition: stereotype signal is present before the scored token.
    text = pair.pair.stereotype_text if base == "stereo" else pair.pair.antistereotype_text
    return f"{text} {suffix}".strip()


def _score_position(bundle, prompt: str, max_length: int) -> tuple[dict[str, torch.Tensor], int]:
    encoded = encode_text(bundle.tokenizer, prompt, bundle.device, max_length)
    pos = int(encoded["input_ids"].shape[1] - 1)
    return encoded, pos


def _build_patch(
    *,
    axis: str,
    pos: int,
    directions: dict[tuple[str, int], np.ndarray],
    device: torch.device,
    mode: str,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    for (dir_axis, layer), direction_np in directions.items():
        if dir_axis != axis:
            continue
        idx = layer - 1
        d = torch.tensor(direction_np, device=device, dtype=torch.float32)
        if mode == "remove":
            h = make_direction_projection_at_position_hook(pos, d)
        else:
            h = make_direction_injection_at_position_hook(pos, d, alpha=1.0)
        patch_map[idx] = _compose(patch_map.get(idx), h)
    return patch_map


def _margin_with_patch(
    *,
    bundle,
    encoded: dict[str, torch.Tensor],
    pos: int,
    stereo_token: int,
    anti_token: int,
    patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
) -> float | None:
    cap = forward_with_component_capture(
        model=bundle.model,
        encoded_inputs=encoded,
        output_hidden_states=False,
        capture_attention=False,
        capture_mlp=False,
        residual_patch_map=patch_map,
    )
    if pos >= cap.logits.shape[1]:
        return None
    return compute_score_from_logits(
        cap.logits,
        position=pos,
        pos_token=stereo_token,
        neg_token=anti_token,
    )


def main() -> None:
    args = parse_args()
    ctx = start_run("26", parameters=vars(args), project_root=PROJECT_ROOT)
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
        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))

        train_idx = [int(i) for i in split.get("train_indices", [])]
        test_idx = [int(i) for i in split.get("test_indices", [])]
        train_pairs = [aligned_pairs[i] for i in train_idx if 0 <= i < len(aligned_pairs)]
        test_pairs = [aligned_pairs[i] for i in test_idx if 0 <= i < len(aligned_pairs)]

        rng = np.random.default_rng(args.seed)
        if args.train_pairs > 0 and len(train_pairs) > args.train_pairs:
            keep = sorted(rng.choice(len(train_pairs), size=args.train_pairs, replace=False).tolist())
            train_pairs = [train_pairs[i] for i in keep]
        if args.heldout_pairs > 0 and len(test_pairs) > args.heldout_pairs:
            keep = sorted(rng.choice(len(test_pairs), size=args.heldout_pairs, replace=False).tolist())
            test_pairs = [test_pairs[i] for i in keep]

        prompt_suffix = _template_suffix(args.prompt_variant, args.prompt_suffix)
        refs = {
            "exp1_run_dir": str(exp1_dir),
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
            "prompt_variant": args.prompt_variant,
            "prompt_suffix": prompt_suffix,
            "emit_occupancy": bool(args.emit_occupancy),
        }
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **refs})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)

        # Build same-position directions from prefix-difference prompts.
        stereo_buckets: dict[tuple[str, int], list[np.ndarray]] = {}
        anti_buckets: dict[tuple[str, int], list[np.ndarray]] = {}
        for pair in train_pairs:
            ps = _arm_prompt(pair, "stereo", prompt_suffix)
            pa = _arm_prompt(pair, "anti", prompt_suffix)
            enc_s, pos_s = _score_position(bundle, ps, args.max_length)
            enc_a, pos_a = _score_position(bundle, pa, args.max_length)

            cap_s = forward_with_component_capture(
                model=bundle.model,
                encoded_inputs=enc_s,
                output_hidden_states=True,
                capture_attention=False,
                capture_mlp=False,
            )
            cap_a = forward_with_component_capture(
                model=bundle.model,
                encoded_inputs=enc_a,
                output_hidden_states=True,
                capture_attention=False,
                capture_mlp=False,
            )
            if not cap_s.hidden_states or not cap_a.hidden_states:
                continue
            for layer_idx, (hs_s, hs_a) in enumerate(zip(cap_s.hidden_states[1:], cap_a.hidden_states[1:]), start=1):
                if pos_s >= hs_s.shape[1] or pos_a >= hs_a.shape[1]:
                    continue
                key = (pair.pair.axis, layer_idx)
                stereo_buckets.setdefault(key, []).append(hs_s[0, pos_s, :].detach().float().cpu().numpy())
                anti_buckets.setdefault(key, []).append(hs_a[0, pos_a, :].detach().float().cpu().numpy())

        directions: dict[tuple[str, int], np.ndarray] = {}
        for key in sorted(set(stereo_buckets) | set(anti_buckets)):
            s = stereo_buckets.get(key, [])
            a = anti_buckets.get(key, [])
            if len(s) < 2 or len(a) < 2:
                continue
            directions[key] = (np.mean(np.stack(s), axis=0) - np.mean(np.stack(a), axis=0)).astype(np.float32)

        rows: list[dict[str, Any]] = []
        condition_pair_diffs_score: dict[str, dict[str, float]] = {}
        condition_pair_diffs_margin: dict[str, dict[str, float]] = {}
        condition_pair_base_margin: dict[str, dict[str, float]] = {}
        condition_pair_edit_margin: dict[str, dict[str, float]] = {}
        conditions = [
            ("remove_on_stereo", "stereo", "remove"),
            ("inject_on_anti", "anti", "inject"),
        ]

        for cond_name, base_kind, mode in conditions:
            pair_ids: list[str] = []
            base_vals: list[float] = []
            edit_vals: list[float] = []
            for pair in test_pairs:
                prompt = _arm_prompt(pair, "stereo" if base_kind == "stereo" else "anti", prompt_suffix)
                enc, pos = _score_position(bundle, prompt, args.max_length)
                base_val = _margin_with_patch(
                    bundle=bundle,
                    encoded=enc,
                    pos=pos,
                    stereo_token=pair.stereo_token,
                    anti_token=pair.anti_token,
                    patch_map=None,
                )
                if base_val is None:
                    continue
                patch = _build_patch(
                    axis=pair.pair.axis,
                    pos=pos,
                    directions=directions,
                    device=bundle.device,
                    mode=mode,
                )
                if not patch:
                    continue
                edit_val = _margin_with_patch(
                    bundle=bundle,
                    encoded=enc,
                    pos=pos,
                    stereo_token=pair.stereo_token,
                    anti_token=pair.anti_token,
                    patch_map=patch,
                )
                if edit_val is None:
                    continue
                pair_ids.append(pair.pair.pair_id)
                base_vals.append(float(base_val))
                edit_vals.append(float(edit_val))

            if not edit_vals:
                continue

            arr_base = np.array(base_vals, dtype=float)
            arr_edit = np.array(edit_vals, dtype=float)
            score_diffs = (arr_edit > 0).astype(float) - (arr_base > 0).astype(float)
            margin_diffs = arr_edit - arr_base
            rng_ci = np.random.default_rng(args.seed)
            score_ci = bootstrap_mean_ci(score_diffs, n_resamples=args.bootstrap_n, rng=rng_ci)
            margin_ci = bootstrap_mean_ci(margin_diffs, n_resamples=args.bootstrap_n, rng=rng_ci)
            p_score, _, _ = paired_sign_test(score_diffs)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_diffs)

            condition_pair_diffs_score[cond_name] = {
                pid: float(v) for pid, v in zip(pair_ids, score_diffs.tolist())
            }
            condition_pair_diffs_margin[cond_name] = {
                pid: float(v) for pid, v in zip(pair_ids, margin_diffs.tolist())
            }
            condition_pair_base_margin[cond_name] = {
                pid: float(v) for pid, v in zip(pair_ids, arr_base.tolist())
            }
            condition_pair_edit_margin[cond_name] = {
                pid: float(v) for pid, v in zip(pair_ids, arr_edit.tolist())
            }

            rows.append(
                {
                    "condition": cond_name,
                    "n_pairs": len(arr_edit),
                    "stereotype_score_baseline": round(float(np.mean(arr_base > 0)), 8),
                    "stereotype_score_intervened": round(float(np.mean(arr_edit > 0)), 8),
                    "stereotype_score_delta": round(float(np.mean(score_diffs)), 8),
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

        matrix_path = ctx.tables_dir / "ar_same_position_matrix.csv"
        write_csv(
            matrix_path,
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
        ctx.register_artifact(matrix_path, artifact_type="table", description="Exp26 same-position matrix.")

        contrast_rows: list[dict[str, Any]] = []
        inj = condition_pair_diffs_score.get("inject_on_anti", {})
        rem = condition_pair_diffs_score.get("remove_on_stereo", {})
        inj_m = condition_pair_diffs_margin.get("inject_on_anti", {})
        rem_m = condition_pair_diffs_margin.get("remove_on_stereo", {})
        common = sorted(set(inj) & set(rem) & set(inj_m) & set(rem_m))
        if common:
            score_con = np.array([inj[p] - rem[p] for p in common], dtype=float)
            margin_con = np.array([inj_m[p] - rem_m[p] for p in common], dtype=float)
            rng_ci = np.random.default_rng(args.seed)
            score_ci = bootstrap_mean_ci(score_con, n_resamples=args.bootstrap_n, rng=rng_ci)
            margin_ci = bootstrap_mean_ci(margin_con, n_resamples=args.bootstrap_n, rng=rng_ci)
            p_score, _, _ = paired_sign_test(score_con)
            p_margin, _ = wilcoxon_signed_rank_safe(margin_con)
            contrast_rows.append(
                {
                    "contrast": "primary_inject_anti_minus_remove_stereo",
                    "n_pairs": len(common),
                    "mean_score_contrast": round(float(np.mean(score_con)), 8),
                    "mean_score_contrast_ci_low": _rounded(score_ci.ci_low),
                    "mean_score_contrast_ci_high": _rounded(score_ci.ci_high),
                    "mean_margin_contrast": round(float(np.mean(margin_con)), 8),
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

        contrast_path = ctx.tables_dir / "ar_same_position_contrast.csv"
        write_csv(
            contrast_path,
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
        ctx.register_artifact(contrast_path, artifact_type="table", description="Exp26 matched contrast.")

        if args.emit_pair_level:
            pair_rows: list[dict[str, Any]] = []
            cond_score_names = sorted(condition_pair_diffs_score.keys())
            cond_margin_names = sorted(condition_pair_diffs_margin.keys())
            for pair in test_pairs:
                pair_id = pair.pair.pair_id
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "source": pair.pair.source,
                    "axis": pair.pair.axis,
                }
                for cond in cond_margin_names:
                    row[f"{cond}_baseline_margin"] = _rounded(condition_pair_base_margin.get(cond, {}).get(pair_id))
                    row[f"{cond}_edited_margin"] = _rounded(condition_pair_edit_margin.get(cond, {}).get(pair_id))
                    row[f"{cond}_margin_delta"] = _rounded(condition_pair_diffs_margin.get(cond, {}).get(pair_id))
                for cond in cond_score_names:
                    row[f"{cond}_score_delta"] = _rounded(condition_pair_diffs_score.get(cond, {}).get(pair_id))

                inj_s = condition_pair_diffs_score.get("inject_on_anti", {}).get(pair_id)
                rem_s = condition_pair_diffs_score.get("remove_on_stereo", {}).get(pair_id)
                inj_m = condition_pair_diffs_margin.get("inject_on_anti", {}).get(pair_id)
                rem_m = condition_pair_diffs_margin.get("remove_on_stereo", {}).get(pair_id)
                row["primary_score_contrast"] = _rounded(None if inj_s is None or rem_s is None else float(inj_s - rem_s))
                row["primary_margin_contrast"] = _rounded(None if inj_m is None or rem_m is None else float(inj_m - rem_m))
                pair_rows.append(row)

            pair_fields = ["pair_id", "source", "axis"]
            for cond in cond_margin_names:
                pair_fields.extend([f"{cond}_baseline_margin", f"{cond}_edited_margin", f"{cond}_margin_delta"])
            pair_fields.extend([f"{c}_score_delta" for c in cond_score_names])
            pair_fields.extend(["primary_score_contrast", "primary_margin_contrast"])
            pair_path = ctx.tables_dir / "ar_same_position_pair_deltas.csv"
            write_csv(pair_path, pair_rows, fieldnames=pair_fields)
            ctx.register_artifact(
                pair_path,
                artifact_type="table",
                description="Exp26 per-pair condition deltas and primary inject-minus-remove contrasts.",
            )

        occupancy_pair_rows = 0
        occupancy_layer_rows = 0
        if args.emit_occupancy:
            rng_occ = np.random.default_rng(args.seed + 13091)
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
            for pair in test_pairs:
                axis = pair.pair.axis
                pair_id = pair.pair.pair_id
                for base_kind in ["stereo", "anti"]:
                    prompt = _arm_prompt(pair, base_kind, prompt_suffix)
                    enc, pos = _score_position(bundle, prompt, args.max_length)
                    cap = forward_with_component_capture(
                        model=bundle.model,
                        encoded_inputs=enc,
                        output_hidden_states=True,
                        capture_attention=False,
                        capture_mlp=False,
                    )
                    if not cap.hidden_states:
                        continue
                    true_proj_vals: list[float] = []
                    rand_proj_vals: list[float] = []
                    for (dir_axis, layer), d_hat in dir_unit.items():
                        if dir_axis != axis:
                            continue
                        hs = cap.hidden_states[layer] if layer < len(cap.hidden_states) else None
                        if hs is None or pos >= hs.shape[1]:
                            continue
                        r_hat = rand_unit.get((dir_axis, layer))
                        if r_hat is None:
                            continue
                        h = hs[0, pos, :].detach().float().to(bundle.device)
                        true_proj = float(torch.dot(h, d_hat).item())
                        rand_proj = float(torch.dot(h, r_hat).item())
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
                            "prompt_variant": args.prompt_variant,
                            "prompt_suffix": prompt_suffix,
                        }
                    )

            occ_pair_path = ctx.tables_dir / "ar_same_position_occupancy_pair_summary.csv"
            occ_layer_path = ctx.tables_dir / "ar_same_position_occupancy_layerwise.csv"
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
                    "prompt_variant",
                    "prompt_suffix",
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
                description="Exp26 same-position occupancy summaries (true direction vs random baseline).",
            )
            ctx.register_artifact(
                occ_layer_path,
                artifact_type="table",
                description="Exp26 same-position per-layer occupancy rows.",
            )
            occupancy_pair_rows = len(occ_pair_rows)
            occupancy_layer_rows = len(occ_layer_rows)

        template_meta_path = ctx.tables_dir / "ar_same_position_template_meta.csv"
        write_csv(
            template_meta_path,
            [
                {
                    "prompt_variant": args.prompt_variant,
                    "prompt_suffix": prompt_suffix,
                    "n_train_pairs": len(train_pairs),
                    "n_test_pairs": len(test_pairs),
                }
            ],
            fieldnames=["prompt_variant", "prompt_suffix", "n_train_pairs", "n_test_pairs"],
        )
        ctx.register_artifact(
            template_meta_path,
            artifact_type="table",
            description="Exp26 prompt-template metadata.",
        )

        complete_run(
            ctx,
            metrics={
                "n_train_pairs": len(train_pairs),
                "n_test_pairs": len(test_pairs),
                "directions": len(directions),
                "rows": len(rows),
                "contrast_rows": len(contrast_rows),
                "pair_rows": len(test_pairs) if args.emit_pair_level else 0,
                "occupancy_pair_rows": occupancy_pair_rows,
                "occupancy_layer_rows": occupancy_layer_rows,
                "prompt_variant": args.prompt_variant,
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
