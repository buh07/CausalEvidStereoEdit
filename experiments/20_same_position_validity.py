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
    make_direction_projection_at_position_hook,
)
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 20: same-position vs cross-position direction validity test with "
            "norm-random controls at prediction position."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--exp1-trait-run-dir", default="")
    parser.add_argument("--exp1-pred-run-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _latest_run_dir(
    experiment_slug: str,
    required_relpaths: list[str] | None = None,
    model_name: str | None = None,
    direction_position: str | None = None,
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
        if direction_position is not None and params.get("direction_position") != direction_position:
            continue
        run_dir = Path(payload["run_dir"])
        if required_relpaths and any(not (run_dir / rel).exists() for rel in required_relpaths):
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        raise FileNotFoundError(f"No completed run found for {experiment_slug} with requested filters.")
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


def _rounded(x: float | int | None) -> float | str:
    if x is None:
        return ""
    try:
        v = float(x)
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


def _margin_with_patch(
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
    if pos >= cap.logits.shape[1]:
        return None
    return compute_score_from_logits(
        cap.logits,
        position=pos,
        pos_token=pair.stereo_token,
        neg_token=pair.anti_token,
    )


def _build_layer_patch(
    *,
    pair: AlignedPair,
    directions: dict[tuple[str, int], np.ndarray],
    device: torch.device,
    mode: str,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    axis = pair.pair.axis
    pos = pair.prediction_position
    patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    for (dir_axis, layer), direction_np in directions.items():
        if dir_axis != axis:
            continue
        idx = layer - 1
        d = torch.tensor(direction_np, device=device, dtype=torch.float32)
        if mode == "remove":
            hook = make_direction_projection_at_position_hook(pos, d)
        elif mode == "inject":
            hook = make_direction_injection_at_position_hook(pos, d, alpha=1.0)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        patch_map[idx] = _compose(patch_map.get(idx), hook)
    return patch_map


def _make_norm_random_directions(
    template: dict[tuple[str, int], np.ndarray],
    seed: int,
) -> dict[tuple[str, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[tuple[str, int], np.ndarray] = {}
    for key, vec in template.items():
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            out[key] = np.zeros_like(vec)
            continue
        r = rng.standard_normal(vec.shape[0]).astype(np.float32)
        r /= (np.linalg.norm(r) + 1e-8)
        out[key] = r * norm
    return out


def main() -> None:
    args = parse_args()
    ctx = start_run("20", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        req = [
            "artifacts/aligned_pairs.jsonl",
            "artifacts/train_test_split.json",
            "artifacts/directions_layerwise.npz",
        ]
        exp1_trait_dir = (
            Path(args.exp1_trait_run_dir)
            if args.exp1_trait_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=req,
                model_name=args.model,
                direction_position="trait",
            )
        )
        exp1_pred_dir = (
            Path(args.exp1_pred_run_dir)
            if args.exp1_pred_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=req,
                model_name=args.model,
                direction_position="prediction",
            )
        )

        aligned_pairs = _load_aligned_pairs(exp1_trait_dir / "artifacts" / "aligned_pairs.jsonl")
        split = json.loads((exp1_trait_dir / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = split.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        trait_dirs = load_directions_npz(exp1_trait_dir / "artifacts" / "directions_layerwise.npz")
        pred_dirs = load_directions_npz(exp1_pred_dir / "artifacts" / "directions_layerwise.npz")
        rand_dirs = _make_norm_random_directions(trait_dirs, seed=args.seed)

        refs = {
            "exp1_trait_run_dir": str(exp1_trait_dir),
            "exp1_pred_run_dir": str(exp1_pred_dir),
            "heldout_pairs": len(heldout),
            "trait_direction_count": len(trait_dirs),
            "pred_direction_count": len(pred_dirs),
        }
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **refs})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        rng = np.random.default_rng(args.seed)

        baseline_stereo: dict[str, float] = {}
        baseline_anti: dict[str, float] = {}
        for pair in heldout:
            b_stereo = _margin_with_patch(
                bundle=bundle,
                text=pair.pair.stereotype_text,
                pair=pair,
                max_length=args.max_length,
                patch_map=None,
            )
            b_anti = _margin_with_patch(
                bundle=bundle,
                text=pair.pair.antistereotype_text,
                pair=pair,
                max_length=args.max_length,
                patch_map=None,
            )
            if b_stereo is not None:
                baseline_stereo[pair.pair.pair_id] = float(b_stereo)
            if b_anti is not None:
                baseline_anti[pair.pair.pair_id] = float(b_anti)

        arms = [
            ("same_position_pred", pred_dirs),
            ("cross_position_trait", trait_dirs),
            ("norm_random_control", rand_dirs),
        ]

        pair_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for arm_name, arm_dirs in arms:
            score_remove_diffs: list[float] = []
            margin_remove_diffs: list[float] = []
            score_inject_diffs: list[float] = []
            margin_inject_diffs: list[float] = []
            asym_score_diffs: list[float] = []
            asym_margin_diffs: list[float] = []

            for pair in heldout:
                pid = pair.pair.pair_id
                b_stereo = baseline_stereo.get(pid)
                b_anti = baseline_anti.get(pid)
                if b_stereo is None or b_anti is None:
                    continue

                remove_patch = _build_layer_patch(
                    pair=pair,
                    directions=arm_dirs,
                    device=bundle.device,
                    mode="remove",
                )
                inject_patch = _build_layer_patch(
                    pair=pair,
                    directions=arm_dirs,
                    device=bundle.device,
                    mode="inject",
                )
                if not remove_patch or not inject_patch:
                    continue

                m_remove = _margin_with_patch(
                    bundle=bundle,
                    text=pair.pair.stereotype_text,
                    pair=pair,
                    max_length=args.max_length,
                    patch_map=remove_patch,
                )
                m_inject = _margin_with_patch(
                    bundle=bundle,
                    text=pair.pair.antistereotype_text,
                    pair=pair,
                    max_length=args.max_length,
                    patch_map=inject_patch,
                )
                if m_remove is None or m_inject is None:
                    continue

                s_remove_base = float(b_stereo > 0)
                s_remove_edit = float(m_remove > 0)
                s_inj_base = float(b_anti > 0)
                s_inj_edit = float(m_inject > 0)

                d_remove_score = s_remove_edit - s_remove_base
                d_remove_margin = float(m_remove - b_stereo)
                d_inj_score = s_inj_edit - s_inj_base
                d_inj_margin = float(m_inject - b_anti)
                d_asym_score = d_inj_score - d_remove_score
                d_asym_margin = d_inj_margin - d_remove_margin

                score_remove_diffs.append(d_remove_score)
                margin_remove_diffs.append(d_remove_margin)
                score_inject_diffs.append(d_inj_score)
                margin_inject_diffs.append(d_inj_margin)
                asym_score_diffs.append(d_asym_score)
                asym_margin_diffs.append(d_asym_margin)

                pair_rows.append(
                    {
                        "arm": arm_name,
                        "pair_id": pid,
                        "axis": pair.pair.axis,
                        "source": pair.pair.source,
                        "remove_score_delta": _rounded(d_remove_score),
                        "remove_margin_delta": _rounded(d_remove_margin),
                        "inject_score_delta": _rounded(d_inj_score),
                        "inject_margin_delta": _rounded(d_inj_margin),
                        "asymmetry_score_delta": _rounded(d_asym_score),
                        "asymmetry_margin_delta": _rounded(d_asym_margin),
                    }
                )

            if not asym_score_diffs:
                continue

            arr_rs = np.array(score_remove_diffs, dtype=float)
            arr_rm = np.array(margin_remove_diffs, dtype=float)
            arr_is = np.array(score_inject_diffs, dtype=float)
            arr_im = np.array(margin_inject_diffs, dtype=float)
            arr_as = np.array(asym_score_diffs, dtype=float)
            arr_am = np.array(asym_margin_diffs, dtype=float)

            rs_ci = bootstrap_mean_ci(arr_rs, n_resamples=args.bootstrap_n, rng=rng)
            rm_ci = bootstrap_mean_ci(arr_rm, n_resamples=args.bootstrap_n, rng=rng)
            is_ci = bootstrap_mean_ci(arr_is, n_resamples=args.bootstrap_n, rng=rng)
            im_ci = bootstrap_mean_ci(arr_im, n_resamples=args.bootstrap_n, rng=rng)
            as_ci = bootstrap_mean_ci(arr_as, n_resamples=args.bootstrap_n, rng=rng)
            am_ci = bootstrap_mean_ci(arr_am, n_resamples=args.bootstrap_n, rng=rng)

            p_rs, _, _ = paired_sign_test(arr_rs)
            p_rm, _ = wilcoxon_signed_rank_safe(arr_rm)
            p_is, _, _ = paired_sign_test(arr_is)
            p_im, _ = wilcoxon_signed_rank_safe(arr_im)
            p_as, _, _ = paired_sign_test(arr_as)
            p_am, _ = wilcoxon_signed_rank_safe(arr_am)

            summary_rows.append(
                {
                    "arm": arm_name,
                    "n_pairs": len(arr_as),
                    "remove_score_delta": _rounded(float(np.mean(arr_rs))),
                    "remove_score_ci_low": _rounded(rs_ci.ci_low),
                    "remove_score_ci_high": _rounded(rs_ci.ci_high),
                    "remove_margin_delta": _rounded(float(np.mean(arr_rm))),
                    "remove_margin_ci_low": _rounded(rm_ci.ci_low),
                    "remove_margin_ci_high": _rounded(rm_ci.ci_high),
                    "inject_score_delta": _rounded(float(np.mean(arr_is))),
                    "inject_score_ci_low": _rounded(is_ci.ci_low),
                    "inject_score_ci_high": _rounded(is_ci.ci_high),
                    "inject_margin_delta": _rounded(float(np.mean(arr_im))),
                    "inject_margin_ci_low": _rounded(im_ci.ci_low),
                    "inject_margin_ci_high": _rounded(im_ci.ci_high),
                    "asymmetry_score_delta": _rounded(float(np.mean(arr_as))),
                    "asymmetry_score_ci_low": _rounded(as_ci.ci_low),
                    "asymmetry_score_ci_high": _rounded(as_ci.ci_high),
                    "asymmetry_margin_delta": _rounded(float(np.mean(arr_am))),
                    "asymmetry_margin_ci_low": _rounded(am_ci.ci_low),
                    "asymmetry_margin_ci_high": _rounded(am_ci.ci_high),
                    "p_remove_score": _rounded(p_rs),
                    "p_remove_margin": _rounded(p_rm),
                    "p_inject_score": _rounded(p_is),
                    "p_inject_margin": _rounded(p_im),
                    "p_asymmetry_score": _rounded(p_as),
                    "p_asymmetry_margin": _rounded(p_am),
                    "q_remove_score": "",
                    "q_remove_margin": "",
                    "q_inject_score": "",
                    "q_inject_margin": "",
                    "q_asymmetry_score": "",
                    "q_asymmetry_margin": "",
                }
            )

        _apply_fdr(summary_rows, "p_remove_score", "q_remove_score")
        _apply_fdr(summary_rows, "p_remove_margin", "q_remove_margin")
        _apply_fdr(summary_rows, "p_inject_score", "q_inject_score")
        _apply_fdr(summary_rows, "p_inject_margin", "q_inject_margin")
        _apply_fdr(summary_rows, "p_asymmetry_score", "q_asymmetry_score")
        _apply_fdr(summary_rows, "p_asymmetry_margin", "q_asymmetry_margin")

        pair_path = ctx.tables_dir / "same_position_validity_pairs.csv"
        write_csv(
            pair_path,
            pair_rows,
            fieldnames=[
                "arm",
                "pair_id",
                "axis",
                "source",
                "remove_score_delta",
                "remove_margin_delta",
                "inject_score_delta",
                "inject_margin_delta",
                "asymmetry_score_delta",
                "asymmetry_margin_delta",
            ],
        )
        ctx.register_artifact(pair_path, artifact_type="table", description="Per-pair Exp20 deltas.")

        summary_path = ctx.tables_dir / "same_position_validity_summary.csv"
        write_csv(
            summary_path,
            summary_rows,
            fieldnames=[
                "arm",
                "n_pairs",
                "remove_score_delta",
                "remove_score_ci_low",
                "remove_score_ci_high",
                "remove_margin_delta",
                "remove_margin_ci_low",
                "remove_margin_ci_high",
                "inject_score_delta",
                "inject_score_ci_low",
                "inject_score_ci_high",
                "inject_margin_delta",
                "inject_margin_ci_low",
                "inject_margin_ci_high",
                "asymmetry_score_delta",
                "asymmetry_score_ci_low",
                "asymmetry_score_ci_high",
                "asymmetry_margin_delta",
                "asymmetry_margin_ci_low",
                "asymmetry_margin_ci_high",
                "p_remove_score",
                "p_remove_margin",
                "p_inject_score",
                "p_inject_margin",
                "p_asymmetry_score",
                "p_asymmetry_margin",
                "q_remove_score",
                "q_remove_margin",
                "q_inject_score",
                "q_inject_margin",
                "q_asymmetry_score",
                "q_asymmetry_margin",
            ],
        )
        ctx.register_artifact(summary_path, artifact_type="table", description="Exp20 summary by arm.")

        complete_run(
            ctx,
            metrics={
                "heldout_pairs": len(heldout),
                "arm_rows": len(summary_rows),
                "pair_rows": len(pair_rows),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
