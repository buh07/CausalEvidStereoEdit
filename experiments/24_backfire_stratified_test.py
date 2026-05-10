#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, write_csv, write_json
from stereacl.attention_heads import build_attention_projection_specs, make_attention_head_zero_hook
from stereacl.data import ContrastPair
from stereacl.interventions import make_zero_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 24: suppressor-stratified multi-site backfire confirmation test."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=241)
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument("--exp09-run-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _latest_run_dir(slug: str, model: str, required: list[str]) -> Path:
    root = PROJECT_ROOT / "results" / slug
    candidates = sorted(root.glob("*/*/manifest.json"))
    best: tuple[str, Path] | None = None
    for mp in candidates:
        payload = json.loads(mp.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        if payload.get("parameters", {}).get("model") != model:
            continue
        rd = Path(payload["run_dir"])
        if any(not (rd / rel).exists() for rel in required):
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, rd)
    if best is None:
        raise FileNotFoundError(f"No completed run for {slug} model={model}")
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


def _compose(
    first: Callable[[torch.Tensor], torch.Tensor] | None,
    second: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if first is None:
        return second

    def _c(x: torch.Tensor) -> torch.Tensor:
        return second(first(x))

    return _c


def _is_attention(component_type: str) -> bool:
    return component_type.startswith("attention")


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

                def _mk(hf: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
                    def _hook(_module, inputs: tuple[torch.Tensor, ...]):
                        if not inputs:
                            return None
                        patched = hf(inputs[0])
                        if len(inputs) == 1:
                            return (patched,)
                        return (patched, *inputs[1:])

                    return _hook

                hooks.append(module.register_forward_pre_hook(_mk(patch_hook)))

        cap = forward_with_component_capture(
            model=model,
            encoded_inputs=encoded_inputs,
            output_hidden_states=False,
            capture_attention=bool(attention_patch_map),
            capture_mlp=bool(mlp_patch_map),
            attention_patch_map=attention_patch_map if attention_patch_map else None,
            mlp_patch_map=mlp_patch_map if mlp_patch_map else None,
        )
        return cap.logits
    finally:
        for h in hooks:
            h.remove()


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


def _one_sided_backfire_p(diffs: np.ndarray) -> tuple[float, int, int]:
    nz = diffs[np.abs(diffs) > 1e-12]
    n = int(nz.size)
    if n == 0:
        return float("nan"), 0, 0
    n_pos = int(np.sum(nz > 0))
    p = float(binomtest(k=n_pos, n=n, p=0.5, alternative="greater").pvalue)
    return p, n, n_pos


def _build_stratum_components(
    axis_df: pd.DataFrame,
    top_k: int,
) -> dict[str, list[tuple[str, int, str, int | None]]]:
    work = axis_df.copy()
    work = work.sort_values("ranking_source", ascending=True)
    # causal labels from single ablation:
    # score_delta < 0 => promoter (ablation reduces stereotype score)
    # score_delta > 0 => suppressor (ablation increases stereotype score)
    promoters = work[pd.to_numeric(work["stereotype_score_delta"], errors="coerce") < 0]
    suppressors = work[pd.to_numeric(work["stereotype_score_delta"], errors="coerce") > 0]

    def _to_comp(df: pd.DataFrame) -> list[tuple[str, int, str, int | None]]:
        out = []
        for _, r in df.iterrows():
            layer = int(r["layer"])
            out.append(
                (
                    str(r["component_type"]),
                    layer,
                    str(r.get("component_id", f"L{layer}")),
                    _optional_int(r.get("head_index")),
                )
            )
        return out

    prom_list = _to_comp(promoters)
    supp_list = _to_comp(suppressors)

    def _mix(frac_supp: float) -> list[tuple[str, int, str, int | None]]:
        n_supp = int(math.ceil(top_k * frac_supp))
        n_prom = max(0, top_k - n_supp)
        chosen = supp_list[:n_supp] + prom_list[:n_prom]
        if len(chosen) < top_k:
            rem = [c for c in prom_list[n_prom:] + supp_list[n_supp:] if c not in chosen]
            chosen.extend(rem[: top_k - len(chosen)])
        return chosen[:top_k]

    return {
        "low_suppressor": _mix(0.0),
        "mid_suppressor": _mix(0.25),
        "high_suppressor": _mix(0.5),
    }


def main() -> None:
    args = parse_args()
    ctx = start_run("24", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = (
            Path(args.exp1_run_dir)
            if args.exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                args.model,
                ["artifacts/aligned_pairs.jsonl", "artifacts/train_test_split.json"],
            )
        )
        exp09_dir = (
            Path(args.exp09_run_dir)
            if args.exp09_run_dir
            else _latest_run_dir(
                "09_dla_atp_adjudication",
                args.model,
                ["tables/adjudication_single_ablation.csv"],
            )
        )

        aligned = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = split.get("test_indices", [])
        heldout = [aligned[i] for i in test_indices if 0 <= i < len(aligned)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        adjud = pd.read_csv(exp09_dir / "tables" / "adjudication_single_ablation.csv")
        adjud = adjud[adjud["ranking_source"].astype(str) == "union"].copy()
        axes = sorted(set(adjud["axis"].astype(str).tolist()) & {p.pair.axis for p in heldout})

        refs = {
            "exp1_run_dir": str(exp1_dir),
            "exp09_run_dir": str(exp09_dir),
            "heldout_pairs": len(heldout),
            "axes": axes,
            "top_k": args.top_k,
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

        baseline_map: dict[str, float] = {}
        for pair in heldout:
            enc = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)
            cap = forward_with_component_capture(
                bundle.model,
                enc,
                output_hidden_states=False,
                capture_attention=False,
                capture_mlp=False,
            )
            pos = pair.prediction_position
            if pos < cap.logits.shape[1]:
                baseline_map[pair.pair.pair_id] = compute_score_from_logits(
                    cap.logits,
                    position=pos,
                    pos_token=pair.stereo_token,
                    neg_token=pair.anti_token,
                )

        by_axis_pairs: dict[str, list[AlignedPair]] = {}
        for axis in axes:
            by_axis_pairs[axis] = [p for p in heldout if p.pair.axis == axis]

        rows: list[dict[str, Any]] = []
        for axis in axes:
            axis_df = adjud[adjud["axis"].astype(str) == axis]
            if axis_df.empty:
                continue
            strata = _build_stratum_components(axis_df, top_k=args.top_k)
            axis_pairs = by_axis_pairs.get(axis, [])
            if not axis_pairs:
                continue

            for stratum, comps in strata.items():
                if len(comps) < args.top_k:
                    continue
                diffs_score: list[float] = []
                diffs_margin: list[float] = []

                for pair in axis_pairs:
                    pid = pair.pair.pair_id
                    base = baseline_map.get(pid)
                    if base is None:
                        continue
                    pos = pair.prediction_position
                    enc = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)

                    attn_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
                    mlp_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
                    preproj_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}

                    for ct, layer, _cid, head_index in comps:
                        idx = layer - 1
                        h = make_zero_position_hook(pos)
                        if ct == "attention_head" and head_index is not None and idx in head_specs:
                            try:
                                hh = make_attention_head_zero_hook(
                                    spec=head_specs[idx],
                                    position=pos,
                                    head_index=head_index,
                                )
                            except ValueError:
                                continue
                            preproj_patch[idx] = _compose(preproj_patch.get(idx), hh)
                        elif _is_attention(ct):
                            attn_patch[idx] = _compose(attn_patch.get(idx), h)
                        else:
                            mlp_patch[idx] = _compose(mlp_patch.get(idx), h)

                    logits = _forward_logits_with_patches(
                        model=bundle.model,
                        encoded_inputs=enc,
                        attention_patch_map=attn_patch,
                        mlp_patch_map=mlp_patch,
                        preproj_patch_map=preproj_patch,
                        head_specs=head_specs,
                    )
                    if pos >= logits.shape[1]:
                        continue
                    edited = compute_score_from_logits(
                        logits,
                        position=pos,
                        pos_token=pair.stereo_token,
                        neg_token=pair.anti_token,
                    )
                    d_margin = float(edited - base)
                    d_score = float(float(edited > 0) - float(base > 0))
                    diffs_margin.append(d_margin)
                    diffs_score.append(d_score)

                if not diffs_score:
                    continue
                arr_s = np.array(diffs_score, dtype=float)
                arr_m = np.array(diffs_margin, dtype=float)
                ci_s = bootstrap_mean_ci(arr_s, n_resamples=args.bootstrap_n, rng=rng)
                ci_m = bootstrap_mean_ci(arr_m, n_resamples=args.bootstrap_n, rng=rng)
                p_backfire, n_nonzero, k_pos = _one_sided_backfire_p(arr_s)
                p_margin, _ = wilcoxon_signed_rank_safe(arr_m)

                suppressor_count = int(sum(1 for c in comps if float(axis_df[axis_df["component_id"].astype(str) == str(c[2])]["stereotype_score_delta"].head(1).fillna(0).astype(float).mean()) > 0))
                rows.append(
                    {
                        "axis": axis,
                        "stratum": stratum,
                        "top_k": args.top_k,
                        "n_components": len(comps),
                        "suppressor_count": suppressor_count,
                        "suppressor_fraction": _rounded(suppressor_count / max(1, len(comps))),
                        "n_pairs": len(arr_s),
                        "mean_score_delta": _rounded(float(np.mean(arr_s))),
                        "score_ci_low": _rounded(ci_s.ci_low),
                        "score_ci_high": _rounded(ci_s.ci_high),
                        "mean_margin_delta": _rounded(float(np.mean(arr_m))),
                        "margin_ci_low": _rounded(ci_m.ci_low),
                        "margin_ci_high": _rounded(ci_m.ci_high),
                        "p_backfire_one_sided": _rounded(p_backfire),
                        "n_nonzero_score": n_nonzero,
                        "k_positive_score": k_pos,
                        "p_margin_wilcoxon": _rounded(p_margin),
                        "q_backfire_one_sided": "",
                        "q_margin_wilcoxon": "",
                    }
                )

        # FDR over all stratum-axis tests
        p_back = [float(r["p_backfire_one_sided"]) if r["p_backfire_one_sided"] != "" else float("nan") for r in rows]
        p_marg = [float(r["p_margin_wilcoxon"]) if r["p_margin_wilcoxon"] != "" else float("nan") for r in rows]
        q_back = benjamini_hochberg(p_back)
        q_marg = benjamini_hochberg(p_marg)
        for i in range(len(rows)):
            rows[i]["q_backfire_one_sided"] = _rounded(q_back[i])
            rows[i]["q_margin_wilcoxon"] = _rounded(q_marg[i])

        out_path = ctx.tables_dir / "backfire_stratified_results.csv"
        write_csv(
            out_path,
            rows,
            fieldnames=[
                "axis",
                "stratum",
                "top_k",
                "n_components",
                "suppressor_count",
                "suppressor_fraction",
                "n_pairs",
                "mean_score_delta",
                "score_ci_low",
                "score_ci_high",
                "mean_margin_delta",
                "margin_ci_low",
                "margin_ci_high",
                "p_backfire_one_sided",
                "n_nonzero_score",
                "k_positive_score",
                "p_margin_wilcoxon",
                "q_backfire_one_sided",
                "q_margin_wilcoxon",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Exp24 suppressor-stratified backfire results.")

        complete_run(
            ctx,
            metrics={
                "rows": len(rows),
                "axes": len({r["axis"] for r in rows}) if rows else 0,
                "strata": len({r["stratum"] for r in rows}) if rows else 0,
                "significant_backfire_rows_q05": int(sum(1 for r in rows if r["q_backfire_one_sided"] != "" and float(r["q_backfire_one_sided"]) < 0.05)),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
