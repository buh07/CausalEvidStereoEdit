#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.attention_heads import build_attention_projection_specs, make_attention_head_zero_hook
from stereacl.data import ContrastPair
from stereacl.interventions import make_direction_projection_at_position_hook, make_zero_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 22: pathway and head-level decomposition around top causal layers."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--max-heads", type=int, default=0, help="0 = evaluate all heads in selected layers")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument("--exp10-run-dir", default="")
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
    for mp in candidates:
        payload = json.loads(mp.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        if model_name is not None and payload.get("parameters", {}).get("model") != model_name:
            continue
        rd = Path(payload["run_dir"])
        if required_relpaths and any(not (rd / rel).exists() for rel in required_relpaths):
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, rd)
    if best is None:
        raise FileNotFoundError(f"No completed run found for {experiment_slug} for model={model_name}")
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


def _margin_from_logits(logits: torch.Tensor, pair: AlignedPair) -> float | None:
    pos = pair.prediction_position
    if pos >= logits.shape[1]:
        return None
    return compute_score_from_logits(
        logits,
        position=pos,
        pos_token=pair.stereo_token,
        neg_token=pair.anti_token,
    )


def _forward_logits_with_preproj_hook(
    *,
    model,
    encoded_inputs: dict[str, torch.Tensor],
    layer_to_preproj_hook: dict[int, Callable[[torch.Tensor], torch.Tensor]],
    head_specs: dict[int, Any],
) -> torch.Tensor:
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for layer_idx, patch_hook in layer_to_preproj_hook.items():
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
            capture_attention=False,
            capture_mlp=False,
            attention_patch_map=None,
            mlp_patch_map=None,
            residual_patch_map=None,
        )
        return cap.logits
    finally:
        for h in hooks:
            h.remove()


def _select_top_layers(exp10_path: Path, top_layers: int) -> list[int]:
    df = pd.read_csv(exp10_path)
    if df.empty:
        return []
    work = df.copy()
    work["abs_score_delta"] = pd.to_numeric(work["stereotype_score_delta"], errors="coerce").abs()
    agg = work.groupby("layer", dropna=True)["abs_score_delta"].mean().reset_index()
    agg = agg.sort_values("abs_score_delta", ascending=False).head(top_layers)
    layers = [int(x) for x in agg["layer"].tolist() if not pd.isna(x)]
    return sorted(set(layers))


def main() -> None:
    args = parse_args()
    ctx = start_run("22", parameters=vars(args), project_root=PROJECT_ROOT)
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
        exp10_dir = (
            Path(args.exp10_run_dir)
            if args.exp10_run_dir
            else _latest_run_dir(
                "10_path_mediation",
                required_relpaths=["tables/path_mediation.csv"],
                model_name=args.model,
            )
        )

        aligned = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = split.get("test_indices", [])
        heldout = [aligned[i] for i in test_indices if 0 <= i < len(aligned)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        directions = load_directions_npz(exp1_dir / "artifacts" / "directions_layerwise.npz")
        exp10_table = exp10_dir / "tables" / "path_mediation.csv"
        top_layers = _select_top_layers(exp10_table, top_layers=args.top_layers)

        refs = {
            "exp1_run_dir": str(exp1_dir),
            "exp10_run_dir": str(exp10_dir),
            "top_layers": top_layers,
            "heldout_pairs": len(heldout),
            "direction_count": len(directions),
        }
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **refs})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        head_specs = build_attention_projection_specs(bundle.model)
        rng = np.random.default_rng(args.seed)

        baseline_margin: dict[str, float] = {}
        encoded_cache: dict[str, dict[str, torch.Tensor]] = {}
        for pair in heldout:
            pid = pair.pair.pair_id
            enc = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)
            encoded_cache[pid] = enc
            cap = forward_with_component_capture(
                bundle.model,
                enc,
                output_hidden_states=False,
                capture_attention=False,
                capture_mlp=False,
            )
            m = _margin_from_logits(cap.logits, pair)
            if m is not None:
                baseline_margin[pid] = float(m)

        layer_rows: list[dict[str, Any]] = []
        head_rows: list[dict[str, Any]] = []

        for layer in top_layers:
            idx = layer - 1
            res_deltas: list[float] = []
            attn_deltas: list[float] = []
            mlp_deltas: list[float] = []
            per_head_deltas: dict[int, list[float]] = defaultdict(list)

            for pair in heldout:
                pid = pair.pair.pair_id
                base = baseline_margin.get(pid)
                if base is None:
                    continue
                pos = pair.prediction_position
                axis = pair.pair.axis
                d_np = directions.get((axis, layer))
                enc = encoded_cache[pid]

                if d_np is not None:
                    d = torch.tensor(d_np, device=bundle.device, dtype=torch.float32)
                    res_hook = make_direction_projection_at_position_hook(pos, d)
                    logits_res = forward_with_component_capture(
                        bundle.model,
                        enc,
                        output_hidden_states=False,
                        capture_attention=False,
                        capture_mlp=False,
                        residual_patch_map={idx: res_hook},
                    ).logits
                    m_res = _margin_from_logits(logits_res, pair)
                    if m_res is not None:
                        res_deltas.append(float(m_res - base))

                z_hook = make_zero_position_hook(pos)
                logits_attn = forward_with_component_capture(
                    bundle.model,
                    enc,
                    output_hidden_states=False,
                    capture_attention=True,
                    capture_mlp=False,
                    attention_patch_map={idx: z_hook},
                ).logits
                m_attn = _margin_from_logits(logits_attn, pair)
                if m_attn is not None:
                    attn_deltas.append(float(m_attn - base))

                logits_mlp = forward_with_component_capture(
                    bundle.model,
                    enc,
                    output_hidden_states=False,
                    capture_attention=False,
                    capture_mlp=True,
                    mlp_patch_map={idx: z_hook},
                ).logits
                m_mlp = _margin_from_logits(logits_mlp, pair)
                if m_mlp is not None:
                    mlp_deltas.append(float(m_mlp - base))

                spec = head_specs.get(idx)
                if spec is None:
                    continue
                n_heads = spec.num_heads if args.max_heads <= 0 else min(spec.num_heads, args.max_heads)
                for h in range(n_heads):
                    h_hook = make_attention_head_zero_hook(spec=spec, position=pos, head_index=h)
                    logits_h = _forward_logits_with_preproj_hook(
                        model=bundle.model,
                        encoded_inputs=enc,
                        layer_to_preproj_hook={idx: h_hook},
                        head_specs=head_specs,
                    )
                    m_h = _margin_from_logits(logits_h, pair)
                    if m_h is not None:
                        per_head_deltas[h].append(float(m_h - base))

            if not (res_deltas or attn_deltas or mlp_deltas):
                continue

            arr_r = np.array(res_deltas, dtype=float) if res_deltas else np.array([], dtype=float)
            arr_a = np.array(attn_deltas, dtype=float) if attn_deltas else np.array([], dtype=float)
            arr_m = np.array(mlp_deltas, dtype=float) if mlp_deltas else np.array([], dtype=float)

            r_ci = bootstrap_mean_ci(arr_r, n_resamples=args.bootstrap_n, rng=rng) if arr_r.size else None
            a_ci = bootstrap_mean_ci(arr_a, n_resamples=args.bootstrap_n, rng=rng) if arr_a.size else None
            m_ci = bootstrap_mean_ci(arr_m, n_resamples=args.bootstrap_n, rng=rng) if arr_m.size else None

            p_r = paired_sign_test(arr_r)[0] if arr_r.size else np.nan
            p_a = paired_sign_test(arr_a)[0] if arr_a.size else np.nan
            p_m = paired_sign_test(arr_m)[0] if arr_m.size else np.nan
            w_r = wilcoxon_signed_rank_safe(arr_r)[0] if arr_r.size else np.nan
            w_a = wilcoxon_signed_rank_safe(arr_a)[0] if arr_a.size else np.nan
            w_m = wilcoxon_signed_rank_safe(arr_m)[0] if arr_m.size else np.nan

            layer_rows.append(
                {
                    "layer": layer,
                    "n_pairs_residual": int(arr_r.size),
                    "n_pairs_attention": int(arr_a.size),
                    "n_pairs_mlp": int(arr_m.size),
                    "mean_margin_delta_residual": _rounded(float(np.mean(arr_r)) if arr_r.size else np.nan),
                    "mean_margin_delta_attention": _rounded(float(np.mean(arr_a)) if arr_a.size else np.nan),
                    "mean_margin_delta_mlp": _rounded(float(np.mean(arr_m)) if arr_m.size else np.nan),
                    "mean_abs_margin_delta_residual": _rounded(float(np.mean(np.abs(arr_r))) if arr_r.size else np.nan),
                    "mean_abs_margin_delta_attention": _rounded(float(np.mean(np.abs(arr_a))) if arr_a.size else np.nan),
                    "mean_abs_margin_delta_mlp": _rounded(float(np.mean(np.abs(arr_m))) if arr_m.size else np.nan),
                    "residual_ci_low": _rounded(r_ci.ci_low) if r_ci else "",
                    "residual_ci_high": _rounded(r_ci.ci_high) if r_ci else "",
                    "attention_ci_low": _rounded(a_ci.ci_low) if a_ci else "",
                    "attention_ci_high": _rounded(a_ci.ci_high) if a_ci else "",
                    "mlp_ci_low": _rounded(m_ci.ci_low) if m_ci else "",
                    "mlp_ci_high": _rounded(m_ci.ci_high) if m_ci else "",
                    "p_sign_residual": _rounded(p_r),
                    "p_sign_attention": _rounded(p_a),
                    "p_sign_mlp": _rounded(p_m),
                    "p_wilcoxon_residual": _rounded(w_r),
                    "p_wilcoxon_attention": _rounded(w_a),
                    "p_wilcoxon_mlp": _rounded(w_m),
                }
            )

            # Per-head rows with ranking by mean absolute delta
            tmp = []
            for h, vals in per_head_deltas.items():
                arr_h = np.array(vals, dtype=float)
                if arr_h.size == 0:
                    continue
                tmp.append((h, float(np.mean(arr_h)), float(np.mean(np.abs(arr_h))), int(arr_h.size)))
            tmp.sort(key=lambda x: x[2], reverse=True)
            for rank, (h, mean_d, mean_abs_d, nvals) in enumerate(tmp, start=1):
                head_rows.append(
                    {
                        "layer": layer,
                        "head_index": h,
                        "head_rank_by_abs_delta": rank,
                        "n_pairs": nvals,
                        "mean_margin_delta": _rounded(mean_d),
                        "mean_abs_margin_delta": _rounded(mean_abs_d),
                    }
                )

        layer_path = ctx.tables_dir / "head_path_layer_summary.csv"
        write_csv(
            layer_path,
            layer_rows,
            fieldnames=[
                "layer",
                "n_pairs_residual",
                "n_pairs_attention",
                "n_pairs_mlp",
                "mean_margin_delta_residual",
                "mean_margin_delta_attention",
                "mean_margin_delta_mlp",
                "mean_abs_margin_delta_residual",
                "mean_abs_margin_delta_attention",
                "mean_abs_margin_delta_mlp",
                "residual_ci_low",
                "residual_ci_high",
                "attention_ci_low",
                "attention_ci_high",
                "mlp_ci_low",
                "mlp_ci_high",
                "p_sign_residual",
                "p_sign_attention",
                "p_sign_mlp",
                "p_wilcoxon_residual",
                "p_wilcoxon_attention",
                "p_wilcoxon_mlp",
            ],
        )
        ctx.register_artifact(layer_path, artifact_type="table", description="Exp22 layer/path decomposition summary.")

        head_path = ctx.tables_dir / "head_path_head_summary.csv"
        write_csv(
            head_path,
            head_rows,
            fieldnames=[
                "layer",
                "head_index",
                "head_rank_by_abs_delta",
                "n_pairs",
                "mean_margin_delta",
                "mean_abs_margin_delta",
            ],
        )
        ctx.register_artifact(head_path, artifact_type="table", description="Exp22 per-head effects.")

        complete_run(
            ctx,
            metrics={
                "heldout_pairs": len(heldout),
                "top_layers": len(top_layers),
                "layer_rows": len(layer_rows),
                "head_rows": len(head_rows),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
