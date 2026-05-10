#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import cosine_similarity, load_directions_npz, write_csv, write_json
from stereacl.attention_heads import (
    AttentionProjectionSpec,
    attention_head_writes_from_preproj,
    build_attention_projection_specs,
)
from stereacl.data import ContrastPair
from stereacl.modeling import (
    encode_text,
    extract_unembedding_matrix,
    forward_with_component_capture,
    load_model_bundle,
)
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.sampling import stratified_axis_sample
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 02: component-wise direct logit attribution with true per-head decomposition."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pairs-limit", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--exp1-run-dir",
        default="",
        help="Optional explicit path to an Experiment 01 run directory. "
        "If omitted, latest completed run is used.",
    )
    parser.add_argument(
        "--top-components-source",
        choices=["block", "head", "mixed"],
        default="block",
        help="Which component granularity to use for top_components.csv.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _latest_exp1_run_dir(project_root: Path, model_name: str) -> Path:
    root = project_root / "results" / "01_layerwise_probing"
    candidates = sorted(root.glob("*/*/manifest.json"))
    best: tuple[str, Path] | None = None
    for manifest_path in candidates:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        if payload.get("parameters", {}).get("model") != model_name:
            continue
        run_dir = Path(payload["run_dir"])
        directions = run_dir / "artifacts" / "directions_layerwise.npz"
        aligned = run_dir / "artifacts" / "aligned_pairs.jsonl"
        if not directions.exists() or not aligned.exists():
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        raise FileNotFoundError(
            "No completed Experiment 01 run with directions and aligned pairs found."
        )
    return best[1]


def _load_aligned_pairs(path: Path) -> list[AlignedPair]:
    pairs: list[AlignedPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _forward_with_preproj_capture(
    model,
    encoded_inputs: dict[str, torch.Tensor],
    specs: dict[int, AttentionProjectionSpec],
    preproj_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    preproj_inputs: dict[int, torch.Tensor] = {}

    for layer_idx, spec in specs.items():
        module = spec.projection_module

        def _make_hook(idx: int) -> Callable:
            def _hook(_module, inputs: tuple[torch.Tensor, ...]):
                if not inputs:
                    return None
                x = inputs[0]
                patched = (
                    preproj_patch_map[idx](x)
                    if preproj_patch_map and idx in preproj_patch_map
                    else x
                )
                preproj_inputs[idx] = patched
                if len(inputs) == 1:
                    return (patched,)
                return (patched, *inputs[1:])

            return _hook

        hooks.append(module.register_forward_pre_hook(_make_hook(layer_idx)))

    try:
        with torch.no_grad():
            outputs = model(
                **encoded_inputs,
                output_hidden_states=False,
                use_cache=False,
            )
    finally:
        for handle in hooks:
            handle.remove()

    return outputs.logits, preproj_inputs


def _select_rows_for_top_components(
    block_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    if source == "block":
        return block_rows
    if source == "head":
        return head_rows
    return [*block_rows, *head_rows]


def main() -> None:
    args = parse_args()
    ctx = start_run("02", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_run_dir = (
            Path(args.exp1_run_dir)
            if args.exp1_run_dir
            else _latest_exp1_run_dir(PROJECT_ROOT, model_name=args.model)
        )
        directions_path = exp1_run_dir / "artifacts" / "directions_layerwise.npz"
        aligned_path = exp1_run_dir / "artifacts" / "aligned_pairs.jsonl"
        directions = load_directions_npz(directions_path)
        aligned_pairs = _load_aligned_pairs(aligned_path)
        aligned_pairs = stratified_axis_sample(aligned_pairs, limit=args.pairs_limit, seed=args.seed)

        run_ref_path = ctx.artifacts_dir / "exp1_dependency.json"
        write_json(
            run_ref_path,
            {
                "exp1_run_dir": str(exp1_run_dir),
                "directions_path": str(directions_path),
                "aligned_pairs_path": str(aligned_path),
                "pairs_selected": len(aligned_pairs),
            },
        )
        ctx.register_artifact(
            run_ref_path, artifact_type="artifact", description="Experiment 01 dependency reference."
        )

        if args.dry_run:
            metrics = {
                "pairs_loaded": len(aligned_pairs),
                "directions_loaded": len(directions),
                "dry_run": True,
            }
            complete_run(ctx, metrics=metrics)
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        unembed = extract_unembedding_matrix(bundle.model).to(bundle.device).float()
        head_specs = build_attention_projection_specs(bundle.model)

        block_scores: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
        block_cosines: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
        head_scores: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
        head_cosines: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
        pair_rows: list[dict[str, Any]] = []

        for idx, pair in enumerate(aligned_pairs, start=1):
            encoded_stereo = encode_text(
                tokenizer=bundle.tokenizer,
                text=pair.pair.stereotype_text,
                device=bundle.device,
                max_length=args.max_length,
            )
            encoded_anti = encode_text(
                tokenizer=bundle.tokenizer,
                text=pair.pair.antistereotype_text,
                device=bundle.device,
                max_length=args.max_length,
            )

            cap_s = forward_with_component_capture(
                model=bundle.model,
                encoded_inputs=encoded_stereo,
                output_hidden_states=False,
                capture_attention=True,
                capture_mlp=True,
            )
            cap_a = forward_with_component_capture(
                model=bundle.model,
                encoded_inputs=encoded_anti,
                output_hidden_states=False,
                capture_attention=True,
                capture_mlp=True,
            )
            _, preproj_s = _forward_with_preproj_capture(bundle.model, encoded_stereo, specs=head_specs)
            _, preproj_a = _forward_with_preproj_capture(bundle.model, encoded_anti, specs=head_specs)

            # Use trait_token_position so component writes are measured at the
            # same position used for direction extraction in Experiment 01.
            # prediction_position collapses stereo==anti for template-style
            # pairs (SeeGULL, StereoSet), giving zero difference vectors.
            pos = pair.trait_token_position
            if pos >= cap_s.logits.shape[1] or pos >= cap_a.logits.shape[1]:
                continue

            u = (unembed[:, pair.stereo_token] - unembed[:, pair.anti_token]).float()

            # Block-level attention contribution.
            for layer_zero_idx in sorted(set(cap_s.attention_outputs) & set(cap_a.attention_outputs)):
                layer = layer_zero_idx + 1
                attn_s = cap_s.attention_outputs[layer_zero_idx]
                attn_a = cap_a.attention_outputs[layer_zero_idx]
                if pos >= attn_s.shape[1] or pos >= attn_a.shape[1]:
                    continue
                write_vec = 0.5 * (attn_s[0, pos, :].float() + attn_a[0, pos, :].float())
                dla = float(torch.dot(write_vec, u).detach().cpu())
                key = (pair.pair.axis, layer, "attention_block", f"L{layer}")
                block_scores[key].append(dla)
                direction = directions.get((pair.pair.axis, layer))
                if direction is not None:
                    block_cosines[key].append(cosine_similarity(write_vec.detach().cpu().numpy(), direction))

            # True per-head attention contributions from pre-projection capture.
            for layer_zero_idx, spec in head_specs.items():
                if layer_zero_idx not in preproj_s or layer_zero_idx not in preproj_a:
                    continue
                pre_s = preproj_s[layer_zero_idx]
                pre_a = preproj_a[layer_zero_idx]
                if pos >= pre_s.shape[1] or pos >= pre_a.shape[1]:
                    continue
                avg_pre = 0.5 * (pre_s + pre_a)
                head_writes = attention_head_writes_from_preproj(avg_pre, spec=spec)  # [B,S,H,O]
                layer = layer_zero_idx + 1
                direction = directions.get((pair.pair.axis, layer))
                for head_idx in range(spec.num_heads):
                    write_vec = head_writes[0, pos, head_idx, :].float()
                    dla = float(torch.dot(write_vec, u).detach().cpu())
                    key = (pair.pair.axis, layer, "attention_head", f"L{layer}H{head_idx}")
                    head_scores[key].append(dla)
                    if direction is not None:
                        head_cosines[key].append(cosine_similarity(write_vec.detach().cpu().numpy(), direction))

            # MLP block-level contribution.
            for layer_zero_idx in sorted(set(cap_s.mlp_outputs) & set(cap_a.mlp_outputs)):
                layer = layer_zero_idx + 1
                mlp_s = cap_s.mlp_outputs[layer_zero_idx]
                mlp_a = cap_a.mlp_outputs[layer_zero_idx]
                if pos >= mlp_s.shape[1] or pos >= mlp_a.shape[1]:
                    continue
                write_vec = 0.5 * (mlp_s[0, pos, :].float() + mlp_a[0, pos, :].float())
                dla = float(torch.dot(write_vec, u).detach().cpu())
                key = (pair.pair.axis, layer, "mlp_block", f"L{layer}")
                block_scores[key].append(dla)
                direction = directions.get((pair.pair.axis, layer))
                if direction is not None:
                    block_cosines[key].append(cosine_similarity(write_vec.detach().cpu().numpy(), direction))

            pair_rows.append(
                {
                    "pair_id": pair.pair.pair_id,
                    "axis": pair.pair.axis,
                    "source": pair.pair.source,
                    "processed_index": idx,
                }
            )

        block_rows: list[dict[str, Any]] = []
        for key in sorted(block_scores):
            axis, layer, component_type, component_id = key
            dla_values = block_scores[key]
            cos_values = block_cosines.get(key, [])
            block_rows.append(
                {
                    "axis": axis,
                    "layer": layer,
                    "component_type": component_type,
                    "component_id": component_id,
                    "head_index": "",
                    "mean_dla_score": round(_mean(dla_values), 8),
                    "mean_abs_dla_score": round(_mean([abs(v) for v in dla_values]), 8),
                    "std_dla_score": round(float(np.std(dla_values)), 8),
                    "mean_direction_cosine": round(_mean(cos_values), 8) if cos_values else "",
                    "n_pairs": len(dla_values),
                }
            )

        head_rows: list[dict[str, Any]] = []
        for key in sorted(head_scores):
            axis, layer, component_type, component_id = key
            head_index = int(component_id.split("H")[1])
            dla_values = head_scores[key]
            cos_values = head_cosines.get(key, [])
            head_rows.append(
                {
                    "axis": axis,
                    "layer": layer,
                    "component_type": component_type,
                    "component_id": component_id,
                    "head_index": head_index,
                    "mean_dla_score": round(_mean(dla_values), 8),
                    "mean_abs_dla_score": round(_mean([abs(v) for v in dla_values]), 8),
                    "std_dla_score": round(float(np.std(dla_values)), 8),
                    "mean_direction_cosine": round(_mean(cos_values), 8) if cos_values else "",
                    "n_pairs": len(dla_values),
                }
            )

        # Unified table (blocks + heads).
        component_rows = [*block_rows, *head_rows]
        table_path = ctx.tables_dir / "component_dla_scores.csv"
        write_csv(
            table_path,
            rows=component_rows,
            fieldnames=[
                "axis",
                "layer",
                "component_type",
                "component_id",
                "head_index",
                "mean_dla_score",
                "mean_abs_dla_score",
                "std_dla_score",
                "mean_direction_cosine",
                "n_pairs",
            ],
        )
        ctx.register_artifact(table_path, artifact_type="table", description="Unified DLA score table.")

        # Dedicated per-head table for explicit decomposition output.
        head_table_path = ctx.tables_dir / "attention_head_dla_scores.csv"
        write_csv(
            head_table_path,
            rows=head_rows,
            fieldnames=[
                "axis",
                "layer",
                "component_type",
                "component_id",
                "head_index",
                "mean_dla_score",
                "mean_abs_dla_score",
                "std_dla_score",
                "mean_direction_cosine",
                "n_pairs",
            ],
        )
        ctx.register_artifact(
            head_table_path,
            artifact_type="table",
            description="True per-head attention DLA decomposition scores.",
        )

        top_rows: list[dict[str, Any]] = []
        top_source_rows = _select_rows_for_top_components(
            block_rows=block_rows,
            head_rows=head_rows,
            source=args.top_components_source,
        )
        by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in top_source_rows:
            by_axis[str(row["axis"])].append(row)
        for axis in sorted(by_axis):
            ranked = sorted(by_axis[axis], key=lambda r: float(r["mean_abs_dla_score"]), reverse=True)
            for rank, row in enumerate(ranked[: args.top_k], start=1):
                top_rows.append(
                    {
                        "axis": axis,
                        "rank": rank,
                        "layer": row["layer"],
                        "component_type": row["component_type"],
                        "component_id": row["component_id"],
                        "head_index": row["head_index"],
                        "mean_dla_score": row["mean_dla_score"],
                        "mean_abs_dla_score": row["mean_abs_dla_score"],
                    }
                )

        top_path = ctx.tables_dir / "top_components.csv"
        write_csv(
            top_path,
            rows=top_rows,
            fieldnames=[
                "axis",
                "rank",
                "layer",
                "component_type",
                "component_id",
                "head_index",
                "mean_dla_score",
                "mean_abs_dla_score",
            ],
        )
        ctx.register_artifact(
            top_path,
            artifact_type="table",
            description=f"Top-K components by axis from source={args.top_components_source}.",
        )

        pair_path = ctx.tables_dir / "processed_pairs.csv"
        write_csv(
            pair_path,
            rows=pair_rows,
            fieldnames=["pair_id", "axis", "source", "processed_index"],
        )
        ctx.register_artifact(pair_path, artifact_type="table", description="Pairs processed in Experiment 02.")

        spec_path = ctx.artifacts_dir / "attention_head_specs.json"
        write_json(
            spec_path,
            {
                str(layer + 1): {
                    "projection_name": spec.projection_name,
                    "projection_kind": spec.projection_kind,
                    "num_heads": spec.num_heads,
                    "head_dim": spec.head_dim,
                    "in_features": spec.in_features,
                    "out_features": spec.out_features,
                }
                for layer, spec in head_specs.items()
            },
        )
        ctx.register_artifact(
            spec_path,
            artifact_type="artifact",
            description="Attention head decomposition specs by layer.",
        )

        metrics = {
            "pairs_loaded": len(aligned_pairs),
            "pairs_processed": len(pair_rows),
            "directions_loaded": len(directions),
            "block_component_rows": len(block_rows),
            "head_component_rows": len(head_rows),
            "component_rows": len(component_rows),
            "top_component_rows": len(top_rows),
            "dry_run": False,
        }
        complete_run(ctx, metrics=metrics)
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
