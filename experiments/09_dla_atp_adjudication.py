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
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 09: single-component causal ablation to adjudicate DLA vs AtP rankings."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
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


def _load_ranked(
    path: Path,
    score_col: str,
    sign_col: str,
    top_k: int,
    promoters_only: bool = False,
) -> dict[str, list[tuple[str, int, str, int | None, int]]]:
    """Return ranked components per axis, preserving head identity where available."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}

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

    result: dict[str, list[tuple[str, int, str, int | None, int]]] = {}
    for axis, group in df.groupby("axis"):
        filtered = group
        if promoters_only and sign_col in filtered.columns:
            filtered = filtered[filtered[sign_col] > 0]
        ranked = filtered.sort_values(score_col, ascending=False).head(top_k).reset_index(drop=True)
        result[str(axis)] = [
            (
                str(row["component_type"]),
                int(row["layer"]),
                str(row["component_id"]) if "component_id" in row.index and not pd.isna(row["component_id"]) else f"L{int(row['layer'])}",
                _optional_int(row["head_index"]) if "head_index" in row.index else None,
                int(i + 1),
            )
            for i, (_, row) in enumerate(ranked.iterrows())
        ]
    return result


def _is_attention_component_type(component_type: str) -> bool:
    return component_type.startswith("attention")


def _forward_with_preproj_patch(
    model,
    encoded_inputs: dict[str, torch.Tensor],
    preproj_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]],
    head_specs: dict[int, Any],
) -> torch.Tensor:
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    try:
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

        with torch.no_grad():
            outputs = model(
                **encoded_inputs,
                output_hidden_states=False,
                use_cache=False,
            )
        return outputs.logits
    finally:
        for handle in hooks:
            handle.remove()


def _build_component_set(
    ranking_source: str,
    dla_ranked: dict[str, list[tuple[str, int, str, int | None, int]]],
    atp_ranked: dict[str, list[tuple[str, int, str, int | None, int]]],
) -> dict[str, dict[tuple[str, int, str, int | None], dict[str, int | None]]]:
    component_set: dict[str, dict[tuple[str, int, str, int | None], dict[str, int | None]]] = {}
    if ranking_source == "dla":
        axes = sorted(dla_ranked.keys())
    elif ranking_source == "atp":
        axes = sorted(atp_ranked.keys())
    else:
        axes = sorted(set(dla_ranked.keys()) | set(atp_ranked.keys()))

    for axis in axes:
        component_set[axis] = {}
        if ranking_source in {"union", "dla"}:
            for (ct, layer, component_id, head_index, rank) in dla_ranked.get(axis, []):
                key = (ct, layer, component_id, head_index)
                component_set[axis].setdefault(key, {"dla_rank": None, "atp_rank": None})
                component_set[axis][key]["dla_rank"] = rank
        if ranking_source in {"union", "atp"}:
            for (ct, layer, component_id, head_index, rank) in atp_ranked.get(axis, []):
                key = (ct, layer, component_id, head_index)
                component_set[axis].setdefault(key, {"dla_rank": None, "atp_rank": None})
                component_set[axis][key]["atp_rank"] = rank
    return component_set


def main() -> None:
    args = parse_args()
    ctx = start_run("09", parameters=vars(args), project_root=PROJECT_ROOT)
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

        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text())
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if eval_sources:
            heldout = [p for p in heldout if p.pair.source in eval_sources]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        dla_promoters = args.promoters_only and args.ranking_source in {"union", "dla"}
        atp_promoters = args.promoters_only and args.ranking_source in {"union", "atp"}

        dla_ranked = _load_ranked(
            exp2_dir / "tables" / "component_dla_scores.csv",
            score_col="mean_abs_dla_score",
            sign_col="mean_dla_score",
            top_k=args.top_k,
            promoters_only=dla_promoters,
        )
        atp_ranked = _load_ranked(
            exp3_dir / "tables" / "attribution_patch_scores.csv",
            score_col="mean_abs_attr_score",
            sign_col="mean_attr_score",
            top_k=args.top_k,
            promoters_only=atp_promoters,
        )

        component_set = _build_component_set(args.ranking_source, dla_ranked, atp_ranked)

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs_path,
            {
                "exp1_run_dir": str(exp1_dir),
                "exp2_run_dir": str(exp2_dir),
                "exp3_run_dir": str(exp3_dir),
                "heldout_pairs": len(heldout),
                "component_count": sum(len(v) for v in component_set.values()),
                "promoters_only": args.promoters_only,
                "ranking_source": args.ranking_source,
                "eval_sources": sorted(eval_sources),
            },
        )
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"heldout_pairs": len(heldout), "dry_run": True})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        head_specs = build_attention_projection_specs(bundle.model)

        baseline_margins: dict[str, float] = {}
        for pair in heldout:
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
                baseline_margins[pair.pair.pair_id] = compute_score_from_logits(
                    cap.logits,
                    position=pos,
                    pos_token=pair.stereo_token,
                    neg_token=pair.anti_token,
                )

        rows: list[dict[str, Any]] = []

        for axis, components in sorted(component_set.items()):
            axis_pairs = [p for p in heldout if p.pair.axis == axis]
            if not axis_pairs:
                continue
            for (component_type, layer, component_id, head_index), ranks in components.items():
                idx = layer - 1

                margins_ablated: list[float] = []
                margins_base: list[float] = []
                for pair in axis_pairs:
                    pos = pair.prediction_position
                    if pair.pair.pair_id not in baseline_margins:
                        continue
                    encoded = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)
                    if _is_attention_component_type(component_type):
                        if component_type == "attention_head" and head_index is not None and idx in head_specs:
                            try:
                                head_hook = make_attention_head_zero_hook(
                                    spec=head_specs[idx],
                                    position=pos,
                                    head_index=head_index,
                                )
                            except ValueError:
                                continue
                            logits = _forward_with_preproj_patch(
                                model=bundle.model,
                                encoded_inputs=encoded,
                                preproj_patch_map={idx: head_hook},
                                head_specs=head_specs,
                            )
                        else:
                            pos_hook = make_zero_position_hook(pos)
                            cap = forward_with_component_capture(
                                bundle.model,
                                encoded,
                                output_hidden_states=False,
                                capture_attention=True,
                                capture_mlp=False,
                                attention_patch_map={idx: pos_hook},
                                mlp_patch_map=None,
                            )
                            logits = cap.logits
                    else:
                        pos_hook = make_zero_position_hook(pos)
                        cap = forward_with_component_capture(
                            bundle.model,
                            encoded,
                            output_hidden_states=False,
                            capture_attention=False,
                            capture_mlp=True,
                            attention_patch_map=None,
                            mlp_patch_map={idx: pos_hook},
                        )
                        logits = cap.logits
                    if pos >= logits.shape[1]:
                        continue
                    ablated = compute_score_from_logits(
                        logits,
                        position=pos,
                        pos_token=pair.stereo_token,
                        neg_token=pair.anti_token,
                    )
                    margins_ablated.append(ablated)
                    margins_base.append(baseline_margins[pair.pair.pair_id])

                if margins_ablated:
                    arr_abl = np.array(margins_ablated, dtype=float)
                    arr_base = np.array(margins_base, dtype=float)
                    score_base = float(np.mean(arr_base > 0))
                    score_abl = float(np.mean(arr_abl > 0))
                    rows.append(
                        {
                            "axis": axis,
                            "component_type": component_type,
                            "component_id": component_id,
                            "head_index": head_index if head_index is not None else "",
                            "layer": layer,
                            "dla_rank": ranks["dla_rank"] if ranks["dla_rank"] is not None else "",
                            "atp_rank": ranks["atp_rank"] if ranks["atp_rank"] is not None else "",
                            "ranking_source": args.ranking_source,
                            "stereotype_score_baseline": round(score_base, 8),
                            "stereotype_score_ablated": round(score_abl, 8),
                            "stereotype_score_delta": round(score_abl - score_base, 8),
                            "mean_margin_baseline": round(float(np.mean(arr_base)), 8),
                            "mean_margin_ablated": round(float(np.mean(arr_abl)), 8),
                            "mean_margin_delta": round(float(np.mean(arr_abl) - np.mean(arr_base)), 8),
                            "n_pairs": len(margins_ablated),
                        }
                    )

        out_path = ctx.tables_dir / "adjudication_single_ablation.csv"
        write_csv(
            out_path,
            rows,
            fieldnames=[
                "axis",
                "component_type",
                "component_id",
                "head_index",
                "layer",
                "dla_rank",
                "atp_rank",
                "ranking_source",
                "stereotype_score_baseline",
                "stereotype_score_ablated",
                "stereotype_score_delta",
                "mean_margin_baseline",
                "mean_margin_ablated",
                "mean_margin_delta",
                "n_pairs",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Single-component adjudication ablation.")

        complete_run(ctx, metrics={"rows": len(rows), "dry_run": False})
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
