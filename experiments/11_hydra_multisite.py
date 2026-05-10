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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 11: hydra/self-repair multi-site ablation test."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--n-sites", default="1,4,8", help="Comma-separated site counts to test.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--promoters-only",
        action="store_true",
        help="Only include stereotype-promoting components (signed score > 0) before ranking.",
    )
    parser.add_argument(
        "--ranking-source",
        choices=["union", "dla", "atp"],
        default="union",
        help="Component ranking source to evaluate.",
    )
    parser.add_argument(
        "--eval-sources",
        default="",
        help="Comma-separated pair sources for heldout evaluation (e.g. stereoset_intrasentence,crows_pairs).",
    )
    parser.add_argument(
        "--axes",
        default="",
        help="Optional comma-separated axis filter (e.g. gender,profession).",
    )
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument("--exp2-run-dir", default="")
    parser.add_argument("--exp3-run-dir", default="")
    return parser.parse_args()


def _parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


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


def _load_ranked_components(
    path: Path,
    axis: str,
    score_col: str,
    sign_col: str,
    top_k: int,
    promoters_only: bool,
) -> list[tuple[str, int, str, int | None]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    df_axis = df[df["axis"] == axis]
    if promoters_only and sign_col in df_axis.columns:
        df_axis = df_axis[df_axis[sign_col] > 0]
    df_axis = df_axis.sort_values(score_col, ascending=False).head(top_k)
    comps: list[tuple[str, int, str, int | None]] = []
    for _, row in df_axis.iterrows():
        layer = int(row["layer"])
        comps.append(
            (
                str(row["component_type"]),
                layer,
                str(row["component_id"]) if "component_id" in row.index and not pd.isna(row["component_id"]) else f"L{layer}",
                _optional_int(row["head_index"]) if "head_index" in row.index else None,
            )
        )
    return comps


def _load_top_components(
    dla_path: Path,
    atp_path: Path,
    top_k: int,
    axis: str,
    ranking_source: str,
    promoters_only: bool = False,
) -> list[tuple[str, int, str, int | None]]:
    components: list[tuple[str, int, str, int | None]] = []
    seen: set[tuple[str, int, str, int | None]] = set()

    dla_promoters = promoters_only and ranking_source in {"union", "dla"}
    atp_promoters = promoters_only and ranking_source in {"union", "atp"}

    if ranking_source in {"union", "dla"}:
        for comp in _load_ranked_components(
            path=dla_path,
            axis=axis,
            score_col="mean_abs_dla_score",
            sign_col="mean_dla_score",
            top_k=top_k,
            promoters_only=dla_promoters,
        ):
            if comp not in seen:
                seen.add(comp)
                components.append(comp)

    if ranking_source in {"union", "atp"}:
        for comp in _load_ranked_components(
            path=atp_path,
            axis=axis,
            score_col="mean_abs_attr_score",
            sign_col="mean_attr_score",
            top_k=top_k,
            promoters_only=atp_promoters,
        ):
            # A1 default: pure AtP block-level sites only (no synthetic head fallback)
            if ranking_source == "atp" and comp[0] == "attention_head":
                continue
            if comp not in seen:
                seen.add(comp)
                components.append(comp)

    return components


def _compose(
    first: Callable[[torch.Tensor], torch.Tensor] | None,
    second: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if first is None:
        return second

    def _c(x: torch.Tensor) -> torch.Tensor:
        return second(first(x))

    return _c


def _is_attention_component_type(component_type: str) -> bool:
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


def _rounded(x: float | int | None) -> float | str:
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return ""
    return round(float(x), 8)


def main() -> None:
    args = parse_args()
    ctx = start_run("11", parameters=vars(args), project_root=PROJECT_ROOT)
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
        exp2_dir = (
            Path(args.exp2_run_dir)
            if args.exp2_run_dir
            else _latest_run_dir(
                "02_component_dla",
                required_relpaths=["tables/component_dla_scores.csv"],
                model_name=args.model,
            )
        )
        exp3_dir = (
            Path(args.exp3_run_dir)
            if args.exp3_run_dir
            else _latest_run_dir(
                "03_attribution_patching",
                required_relpaths=["tables/attribution_patch_scores.csv"],
                model_name=args.model,
            )
        )

        eval_sources = _parse_csv_set(args.eval_sources)
        axis_filter = _parse_csv_set(args.axes)

        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text())
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if eval_sources:
            heldout = [p for p in heldout if p.pair.source in eval_sources]
        if axis_filter:
            heldout = [p for p in heldout if p.pair.axis in axis_filter]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        n_sites_list = [int(x) for x in args.n_sites.split(",") if x.strip()]
        max_k = max(n_sites_list)

        dla_path = exp2_dir / "tables" / "component_dla_scores.csv"
        atp_path = exp3_dir / "tables" / "attribution_patch_scores.csv"
        axes = sorted({p.pair.axis for p in heldout})

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs_path,
            {
                "exp1_run_dir": str(exp1_dir),
                "exp2_run_dir": str(exp2_dir),
                "exp3_run_dir": str(exp3_dir),
                "heldout_pairs": len(heldout),
                "n_sites_list": n_sites_list,
                "promoters_only": args.promoters_only,
                "ranking_source": args.ranking_source,
                "eval_sources": sorted(eval_sources),
                "axes": sorted(axis_filter),
                "bootstrap_n": args.bootstrap_n,
            },
        )
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"heldout_pairs": len(heldout), "dry_run": True})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        head_specs = build_attention_projection_specs(bundle.model)
        rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(args.seed)

        for axis in axes:
            top_components = _load_top_components(
                dla_path=dla_path,
                atp_path=atp_path,
                top_k=max_k,
                axis=axis,
                ranking_source=args.ranking_source,
                promoters_only=args.promoters_only,
            )
            if not top_components:
                continue
            axis_pairs = [p for p in heldout if p.pair.axis == axis]

            base_by_pair: dict[str, float] = {}
            for pair in axis_pairs:
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
                    base_by_pair[pair.pair.pair_id] = compute_score_from_logits(
                        cap.logits,
                        position=pos,
                        pos_token=pair.stereo_token,
                        neg_token=pair.anti_token,
                    )
            if not base_by_pair:
                continue

            base_arr = np.array(list(base_by_pair.values()), dtype=float)
            base_score = float(np.mean(base_arr > 0))
            base_margin = float(np.mean(base_arr))

            rows.append(
                {
                    "axis": axis,
                    "n_sites": 0,
                    "ranking_source": args.ranking_source,
                    "stereotype_score": round(base_score, 8),
                    "stereotype_score_baseline": round(base_score, 8),
                    "stereotype_score_delta": 0.0,
                    "stereotype_score_delta_ci_low": "",
                    "stereotype_score_delta_ci_high": "",
                    "mean_margin": round(base_margin, 8),
                    "mean_margin_baseline": round(base_margin, 8),
                    "mean_margin_delta": 0.0,
                    "mean_margin_delta_ci_low": "",
                    "mean_margin_delta_ci_high": "",
                    "paired_p_score_sign": "",
                    "paired_p_margin_wilcoxon": "",
                    "q_score_sign": "",
                    "q_margin_wilcoxon": "",
                    "per_site_score_gain": "",
                    "n_score_pos": "",
                    "n_score_neg": "",
                    "n_score_zero": "",
                    "n_pairs": len(base_by_pair),
                }
            )

            for n_sites in n_sites_list:
                sites = top_components[:n_sites]
                if not sites:
                    continue

                paired_base: list[float] = []
                paired_abl: list[float] = []
                for pair in axis_pairs:
                    pos = pair.prediction_position
                    base_val = base_by_pair.get(pair.pair.pair_id)
                    if base_val is None:
                        continue
                    encoded = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)
                    attn_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
                    mlp_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
                    preproj_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
                    for (ct, layer, _component_id, head_index) in sites:
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

                if paired_abl:
                    arr_base = np.array(paired_base, dtype=float)
                    arr_abl = np.array(paired_abl, dtype=float)

                    abl_score = float(np.mean(arr_abl > 0))
                    abl_margin = float(np.mean(arr_abl))
                    score_delta = abl_score - float(np.mean(arr_base > 0))
                    margin_delta = abl_margin - float(np.mean(arr_base))

                    score_pair_diffs = (arr_abl > 0).astype(float) - (arr_base > 0).astype(float)
                    margin_pair_diffs = arr_abl - arr_base
                    n_pos = int(np.sum(score_pair_diffs > 0))
                    n_neg = int(np.sum(score_pair_diffs < 0))
                    n_zero = int(np.sum(score_pair_diffs == 0))

                    score_ci = bootstrap_mean_ci(score_pair_diffs, n_resamples=args.bootstrap_n, rng=rng)
                    margin_ci = bootstrap_mean_ci(margin_pair_diffs, n_resamples=args.bootstrap_n, rng=rng)
                    p_score_sign, _, _ = paired_sign_test(score_pair_diffs)
                    p_margin_w, _ = wilcoxon_signed_rank_safe(margin_pair_diffs)

                    gain = (base_score - abl_score) / n_sites if n_sites > 0 else 0.0
                    rows.append(
                        {
                            "axis": axis,
                            "n_sites": n_sites,
                            "ranking_source": args.ranking_source,
                            "stereotype_score": round(abl_score, 8),
                            "stereotype_score_baseline": round(float(np.mean(arr_base > 0)), 8),
                            "stereotype_score_delta": round(score_delta, 8),
                            "stereotype_score_delta_ci_low": _rounded(score_ci.ci_low),
                            "stereotype_score_delta_ci_high": _rounded(score_ci.ci_high),
                            "mean_margin": round(abl_margin, 8),
                            "mean_margin_baseline": round(float(np.mean(arr_base)), 8),
                            "mean_margin_delta": round(margin_delta, 8),
                            "mean_margin_delta_ci_low": _rounded(margin_ci.ci_low),
                            "mean_margin_delta_ci_high": _rounded(margin_ci.ci_high),
                            "paired_p_score_sign": _rounded(p_score_sign),
                            "paired_p_margin_wilcoxon": _rounded(p_margin_w),
                            "q_score_sign": "",
                            "q_margin_wilcoxon": "",
                            "per_site_score_gain": round(gain, 8),
                            "n_score_pos": n_pos,
                            "n_score_neg": n_neg,
                            "n_score_zero": n_zero,
                            "n_pairs": len(arr_abl),
                        }
                    )

        def _to_float_or_nan(value: Any) -> float:
            try:
                if value == "":
                    return float("nan")
                return float(value)
            except Exception:
                return float("nan")

        non_baseline = [i for i, r in enumerate(rows) if int(r["n_sites"]) > 0]
        if non_baseline:
            p_score = [_to_float_or_nan(rows[i]["paired_p_score_sign"]) for i in non_baseline]
            p_margin = [_to_float_or_nan(rows[i]["paired_p_margin_wilcoxon"]) for i in non_baseline]
            q_score = benjamini_hochberg(p_score)
            q_margin = benjamini_hochberg(p_margin)
            for j, row_idx in enumerate(non_baseline):
                rows[row_idx]["q_score_sign"] = _rounded(q_score[j])
                rows[row_idx]["q_margin_wilcoxon"] = _rounded(q_margin[j])

        out_path = ctx.tables_dir / "hydra_multisite.csv"
        write_csv(
            out_path,
            rows,
            fieldnames=[
                "axis",
                "n_sites",
                "ranking_source",
                "stereotype_score",
                "stereotype_score_baseline",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_margin",
                "mean_margin_baseline",
                "mean_margin_delta",
                "mean_margin_delta_ci_low",
                "mean_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
                "per_site_score_gain",
                "n_score_pos",
                "n_score_neg",
                "n_score_zero",
                "n_pairs",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Hydra multi-site ablation results.")

        complete_run(ctx, metrics={"rows": len(rows), "axes": len(axes), "dry_run": False})
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
