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
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.data import ContrastPair, build_contrast_pairs, deterministic_split_indices
from stereacl.interventions import make_direction_injection_at_position_hook, make_direction_projection_at_position_hook
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
        description="Experiment 28: multi-token matched inject/remove asymmetry with span-level scoring."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=120)
    parser.add_argument("--per-source-limit", type=int, default=2500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-span-len", type=int, default=3)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=281)
    parser.add_argument(
        "--balance-mode",
        choices=["none", "quota"],
        default="none",
        help="Optional heldout balancing mode after deterministic split selection.",
    )
    parser.add_argument(
        "--source-quotas",
        default="stereoset_intrasentence:10,crows_pairs:10",
        help=(
            "Comma-separated minimum quotas per source for --balance-mode quota, "
            "for example: stereoset_intrasentence:10,crows_pairs:10"
        ),
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=0,
        help="Optional RNG seed for quota balancing; defaults to --seed when 0.",
    )
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument(
        "--eval-scope",
        choices=["all", "stereoset_crows", "stereoset_only", "crows_only"],
        default="all",
        help="Dataset sources included in multi-token matched evaluation.",
    )
    parser.add_argument(
        "--span-strata",
        default="all,2,3",
        help="Comma-separated strata for matched-contrast summaries (for example: all,2,3).",
    )
    parser.add_argument("--power-alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--sesoi", type=float, default=0.10)
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


def _approx_mde_score(n_pairs: int, alpha: float, target_power: float) -> float:
    if n_pairs <= 0:
        return float("nan")
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(target_power))
    return 0.5 * (z_alpha + z_beta) / np.sqrt(float(n_pairs))


def _span_margin_for_text(
    *,
    bundle,
    text: str,
    pair: MultiTokenAlignedPair,
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
    mode: str,
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
            h = make_direction_projection_at_position_hook(pos, d) if mode == "remove" else make_direction_injection_at_position_hook(pos, d, alpha=1.0)
            patch[idx] = _compose(patch.get(idx), h)
    return patch


def _parse_span_strata(spec: str) -> list[str]:
    out: list[str] = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(t)
    if not out:
        out = ["all"]
    if "all" not in out:
        out = ["all", *out]
    return out


def _include_by_scope(scope: str) -> tuple[bool, bool, bool]:
    if scope == "all":
        return True, True, True
    if scope == "stereoset_crows":
        return True, True, False
    if scope == "stereoset_only":
        return True, False, False
    if scope == "crows_only":
        return False, True, False
    raise ValueError(f"Unknown eval-scope: {scope}")


def _parse_source_quotas(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in str(spec).split(","):
        token = chunk.strip()
        if not token or ":" not in token:
            continue
        src, raw_n = token.split(":", 1)
        src = src.strip()
        try:
            n = int(raw_n.strip())
        except Exception:
            continue
        if src and n > 0:
            out[src] = n
    return out


def _apply_quota_balance(
    heldout: list[MultiTokenAlignedPair],
    *,
    quotas: dict[str, int],
    heldout_pairs: int,
    seed: int,
) -> tuple[list[MultiTokenAlignedPair], dict[str, Any]]:
    if not heldout:
        return heldout, {"mode": "quota", "selected": 0, "quotas": quotas}
    by_source: dict[str, list[MultiTokenAlignedPair]] = {}
    for pair in heldout:
        by_source.setdefault(pair.pair.source, []).append(pair)
    rng = np.random.default_rng(seed)
    selected: dict[str, MultiTokenAlignedPair] = {}
    per_source_selected: dict[str, int] = {}
    per_source_available: dict[str, int] = {s: len(v) for s, v in by_source.items()}

    for source, q in quotas.items():
        pool = by_source.get(source, [])
        if not pool:
            per_source_selected[source] = 0
            continue
        take = min(q, len(pool))
        idxs = rng.choice(len(pool), size=take, replace=False).tolist()
        for i in idxs:
            p = pool[i]
            selected[p.pair.pair_id] = p
        per_source_selected[source] = take

    all_pool = list(heldout)
    rng.shuffle(all_pool)
    if heldout_pairs > 0:
        for pair in all_pool:
            if len(selected) >= heldout_pairs:
                break
            selected.setdefault(pair.pair.pair_id, pair)

    out = list(selected.values())
    if heldout_pairs > 0 and len(out) > heldout_pairs:
        rng.shuffle(out)
        out = out[:heldout_pairs]

    meta = {
        "mode": "quota",
        "seed": seed,
        "quotas": quotas,
        "available_by_source": per_source_available,
        "selected_by_source": per_source_selected,
        "selected": len(out),
    }
    return out, meta


def main() -> None:
    args = parse_args()
    ctx = start_run("28", parameters=vars(args), project_root=PROJECT_ROOT)
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

        use_st, use_cr, use_sg = _include_by_scope(args.eval_scope)
        pairs = build_contrast_pairs(
            include_stereoset=use_st,
            include_crows=use_cr,
            include_seegull=use_sg,
            per_source_limit=args.per_source_limit,
        )

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        mt_pairs, mt_stats = _align_multitoken_pairs(pairs, bundle.tokenizer, max_span_len=args.max_span_len)

        train_idx, test_idx = deterministic_split_indices(len(mt_pairs), test_fraction=0.2, seed=args.seed)
        heldout = [mt_pairs[int(i)] for i in test_idx if 0 <= int(i) < len(mt_pairs)]
        balance_meta: dict[str, Any] = {"mode": "none"}
        quotas = _parse_source_quotas(args.source_quotas)
        if args.balance_mode == "quota":
            bal_seed = int(args.seed if args.balance_seed == 0 else args.balance_seed)
            heldout, balance_meta = _apply_quota_balance(
                heldout,
                quotas=quotas,
                heldout_pairs=args.heldout_pairs,
                seed=bal_seed,
            )
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        refs = {
            "exp1_run_dir": str(exp1_dir),
            "directions": len(directions),
            "eval_scope": args.eval_scope,
            "max_span_len": args.max_span_len,
            "heldout_pairs": len(heldout),
            "alignment_stats": mt_stats,
            "span_strata": _parse_span_strata(args.span_strata),
            "power_alpha": args.power_alpha,
            "target_power": args.target_power,
            "sesoi": args.sesoi,
            "balance_mode": args.balance_mode,
            "source_quotas": quotas,
            "balance_meta": balance_meta,
        }
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **refs})
            return

        baseline_stereo: dict[str, float] = {}
        baseline_anti: dict[str, float] = {}
        span_len_map: dict[str, int] = {}
        for pair in heldout:
            pair_id = pair.pair.pair_id
            span_len_map[pair_id] = len(pair.stereo_span_tokens)
            m_stereo = _span_margin_for_text(
                bundle=bundle,
                text=pair.pair.stereotype_text,
                pair=pair,
                max_length=args.max_length,
                patch_map=None,
            )
            m_anti = _span_margin_for_text(
                bundle=bundle,
                text=pair.pair.antistereotype_text,
                pair=pair,
                max_length=args.max_length,
                patch_map=None,
            )
            if m_stereo is not None:
                baseline_stereo[pair_id] = float(m_stereo)
            if m_anti is not None:
                baseline_anti[pair_id] = float(m_anti)

        conditions = [
            ("remove_on_stereo", "stereo", "remove"),
            ("remove_on_anti", "anti", "remove"),
            ("inject_on_stereo", "stereo", "inject"),
            ("inject_on_anti", "anti", "inject"),
        ]
        rows: list[dict[str, Any]] = []
        condition_pair_diffs_score: dict[str, dict[str, float]] = {}
        condition_pair_diffs_margin: dict[str, dict[str, float]] = {}
        condition_pair_base_margin: dict[str, dict[str, float]] = {}
        condition_pair_edit_margin: dict[str, dict[str, float]] = {}

        rng = np.random.default_rng(args.seed)

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
                patch = _build_span_patch(pair=pair, directions=directions, device=bundle.device, mode=mode)
                if not patch:
                    continue
                edited = _span_margin_for_text(
                    bundle=bundle,
                    text=text,
                    pair=pair,
                    max_length=args.max_length,
                    patch_map=patch,
                )
                if edited is None:
                    continue
                pair_ids.append(pair_id)
                base_vals.append(float(base_val))
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

            condition_pair_diffs_score[condition_name] = {pid: float(s) for pid, s in zip(pair_ids, score_diffs.tolist())}
            condition_pair_diffs_margin[condition_name] = {pid: float(m) for pid, m in zip(pair_ids, margin_diffs.tolist())}
            condition_pair_base_margin[condition_name] = {pid: float(m) for pid, m in zip(pair_ids, arr_base.tolist())}
            condition_pair_edit_margin[condition_name] = {pid: float(m) for pid, m in zip(pair_ids, arr_edit.tolist())}

            rows.append(
                {
                    "condition": condition_name,
                    "base_distribution": base_kind,
                    "n_pairs": len(arr_edit),
                    "stereotype_score_baseline": round(score_base, 8),
                    "stereotype_score_intervened": round(score_edit, 8),
                    "stereotype_score_delta": round(score_edit - score_base, 8),
                    "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
                    "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
                    "mean_span_margin_baseline": round(float(np.mean(arr_base)), 8),
                    "mean_span_margin_intervened": round(float(np.mean(arr_edit)), 8),
                    "mean_span_margin_delta": round(float(np.mean(margin_diffs)), 8),
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

        matrix_path = ctx.tables_dir / "multitoken_asymmetry_2x2_matrix.csv"
        write_csv(
            matrix_path,
            rows,
            fieldnames=[
                "condition",
                "base_distribution",
                "n_pairs",
                "stereotype_score_baseline",
                "stereotype_score_intervened",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_span_margin_baseline",
                "mean_span_margin_intervened",
                "mean_span_margin_delta",
                "mean_span_margin_delta_ci_low",
                "mean_span_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
            ],
        )
        ctx.register_artifact(matrix_path, artifact_type="table", description="Exp28 multi-token 2x2 matrix.")

        contrast_specs = [
            ("primary_inject_anti_minus_remove_stereo", "inject_on_anti", "remove_on_stereo"),
            ("same_base_stereo_inject_minus_remove", "inject_on_stereo", "remove_on_stereo"),
            ("same_base_anti_inject_minus_remove", "inject_on_anti", "remove_on_anti"),
        ]
        contrast_rows: list[dict[str, Any]] = []
        contrast_score_map: dict[str, dict[str, float]] = {}
        contrast_margin_map: dict[str, dict[str, float]] = {}

        for contrast_name, inject_key, remove_key in contrast_specs:
            inj_score_map = condition_pair_diffs_score.get(inject_key, {})
            rem_score_map = condition_pair_diffs_score.get(remove_key, {})
            inj_margin_map = condition_pair_diffs_margin.get(inject_key, {})
            rem_margin_map = condition_pair_diffs_margin.get(remove_key, {})
            common_pair_ids = sorted(set(inj_score_map) & set(rem_score_map) & set(inj_margin_map) & set(rem_margin_map))
            if not common_pair_ids:
                continue
            score_contrast = np.array([inj_score_map[p] - rem_score_map[p] for p in common_pair_ids], dtype=float)
            margin_contrast = np.array([inj_margin_map[p] - rem_margin_map[p] for p in common_pair_ids], dtype=float)
            contrast_score_map[contrast_name] = {pid: float(v) for pid, v in zip(common_pair_ids, score_contrast.tolist())}
            contrast_margin_map[contrast_name] = {pid: float(v) for pid, v in zip(common_pair_ids, margin_contrast.tolist())}
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

        contrast_path = ctx.tables_dir / "multitoken_asymmetry_contrast.csv"
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
        ctx.register_artifact(contrast_path, artifact_type="table", description="Exp28 multi-token matched contrasts.")

        pair_rows: list[dict[str, Any]] = []
        cond_score_names = sorted(condition_pair_diffs_score.keys())
        cond_margin_names = sorted(condition_pair_diffs_margin.keys())
        for pair in heldout:
            pair_id = pair.pair.pair_id
            row: dict[str, Any] = {
                "pair_id": pair_id,
                "source": pair.pair.source,
                "axis": pair.pair.axis,
                "span_len": len(pair.stereo_span_tokens),
                "baseline_margin_stereo": _rounded(baseline_stereo.get(pair_id)),
                "baseline_margin_anti": _rounded(baseline_anti.get(pair_id)),
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

        pair_fields = ["pair_id", "source", "axis", "span_len", "baseline_margin_stereo", "baseline_margin_anti"]
        for cond in cond_margin_names:
            pair_fields.extend([f"{cond}_baseline_margin", f"{cond}_edited_margin", f"{cond}_margin_delta"])
        pair_fields.extend([f"{c}_score_delta" for c in cond_score_names])
        pair_fields.extend(["primary_score_contrast", "primary_margin_contrast"])
        pair_path = ctx.tables_dir / "multitoken_asymmetry_pair_deltas.csv"
        write_csv(pair_path, pair_rows, fieldnames=pair_fields)
        ctx.register_artifact(pair_path, artifact_type="table", description="Exp28 per-pair deltas and contrasts.")

        strata_tokens = _parse_span_strata(args.span_strata)
        primary_map_s = contrast_score_map.get("primary_inject_anti_minus_remove_stereo", {})
        primary_map_m = contrast_margin_map.get("primary_inject_anti_minus_remove_stereo", {})
        primary_common = sorted(set(primary_map_s) & set(primary_map_m))

        strata_rows: list[dict[str, Any]] = []
        for token in strata_tokens:
            if token == "all":
                ids = list(primary_common)
            else:
                try:
                    span_target = int(token)
                except Exception:
                    continue
                ids = [pid for pid in primary_common if span_len_map.get(pid) == span_target]
            if not ids:
                continue
            s_arr = np.array([primary_map_s[pid] for pid in ids], dtype=float)
            m_arr = np.array([primary_map_m[pid] for pid in ids], dtype=float)
            s_ci = bootstrap_mean_ci(s_arr, n_resamples=args.bootstrap_n, rng=rng)
            m_ci = bootstrap_mean_ci(m_arr, n_resamples=args.bootstrap_n, rng=rng)
            p_s, _, _ = paired_sign_test(s_arr)
            p_m, _ = wilcoxon_signed_rank_safe(m_arr)
            mde = _approx_mde_score(len(ids), args.power_alpha, args.target_power)
            power_tag = "adequate_for_sesoi" if np.isfinite(mde) and mde <= args.sesoi else "underpowered_for_sesoi"
            strata_rows.append(
                {
                    "stratum": token,
                    "n_pairs": len(ids),
                    "mean_score_contrast": _rounded(float(np.mean(s_arr))),
                    "mean_score_contrast_ci_low": _rounded(s_ci.ci_low),
                    "mean_score_contrast_ci_high": _rounded(s_ci.ci_high),
                    "paired_p_score_sign": _rounded(p_s),
                    "q_score_sign": "",
                    "mean_margin_contrast": _rounded(float(np.mean(m_arr))),
                    "mean_margin_contrast_ci_low": _rounded(m_ci.ci_low),
                    "mean_margin_contrast_ci_high": _rounded(m_ci.ci_high),
                    "paired_p_margin_wilcoxon": _rounded(p_m),
                    "q_margin_wilcoxon": "",
                    "sesoi": _rounded(args.sesoi),
                    "power_alpha": _rounded(args.power_alpha),
                    "target_power": _rounded(args.target_power),
                    "mde_score_approx": _rounded(mde),
                    "power_vs_sesoi": power_tag,
                }
            )
        _apply_fdr(strata_rows, "paired_p_score_sign", "q_score_sign")
        _apply_fdr(strata_rows, "paired_p_margin_wilcoxon", "q_margin_wilcoxon")

        strata_path = ctx.tables_dir / "multitoken_matched_contrast_by_span.csv"
        write_csv(
            strata_path,
            strata_rows,
            fieldnames=[
                "stratum",
                "n_pairs",
                "mean_score_contrast",
                "mean_score_contrast_ci_low",
                "mean_score_contrast_ci_high",
                "paired_p_score_sign",
                "q_score_sign",
                "mean_margin_contrast",
                "mean_margin_contrast_ci_low",
                "mean_margin_contrast_ci_high",
                "paired_p_margin_wilcoxon",
                "q_margin_wilcoxon",
                "sesoi",
                "power_alpha",
                "target_power",
                "mde_score_approx",
                "power_vs_sesoi",
            ],
        )
        ctx.register_artifact(
            strata_path,
            artifact_type="table",
            description="Exp28 primary matched contrast stratified by span length with MDE/power tags.",
        )

        align_path = ctx.tables_dir / "multitoken_alignment_stats.csv"
        write_csv(
            align_path,
            [{"metric": k, "value": v} for k, v in mt_stats.items()],
            fieldnames=["metric", "value"],
        )
        ctx.register_artifact(
            align_path,
            artifact_type="table",
            description="Exp28 multi-token alignment retention statistics.",
        )

        comp_rows: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        source_span_counts: dict[tuple[str, int], int] = {}
        for pair in heldout:
            src = pair.pair.source
            span_len = len(pair.stereo_span_tokens)
            source_counts[src] = source_counts.get(src, 0) + 1
            key = (src, span_len)
            source_span_counts[key] = source_span_counts.get(key, 0) + 1
        for source, n in sorted(source_counts.items()):
            comp_rows.append(
                {
                    "row_type": "source_total",
                    "source": source,
                    "span_len": "",
                    "n_pairs": int(n),
                    "balance_mode": args.balance_mode,
                }
            )
        for (source, span_len), n in sorted(source_span_counts.items()):
            comp_rows.append(
                {
                    "row_type": "source_span",
                    "source": source,
                    "span_len": int(span_len),
                    "n_pairs": int(n),
                    "balance_mode": args.balance_mode,
                }
            )
        comp_path = ctx.tables_dir / "multitoken_eval_composition.csv"
        write_csv(
            comp_path,
            comp_rows,
            fieldnames=["row_type", "source", "span_len", "n_pairs", "balance_mode"],
        )
        ctx.register_artifact(
            comp_path,
            artifact_type="table",
            description="Exp28 heldout composition by source and span length.",
        )

        complete_run(
            ctx,
            metrics={
                "heldout_pairs": len(heldout),
                "rows": len(rows),
                "contrast_rows": len(contrast_rows),
                "pair_rows": len(pair_rows),
                "strata_rows": len(strata_rows),
                "kept_multitoken_pairs": mt_stats.get("kept_pairs", 0),
                "balance_mode": args.balance_mode,
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
