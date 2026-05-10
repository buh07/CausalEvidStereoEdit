#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_direction, cosine_similarity, write_csv, write_json
from stereacl.attention_heads import (
    AttentionProjectionSpec,
    attention_head_writes_from_preproj,
    build_attention_projection_specs,
)
from stereacl.data import (
    ContrastPair,
    load_crows_pairs,
    load_seegull_indian_state_pairs,
    load_seegull_pairs,
    load_stereoset_intrasentence_pairs,
    save_pairs_jsonl,
)
from stereacl.modeling import (
    encode_text,
    extract_unembedding_matrix,
    forward_with_component_capture,
    load_model_bundle,
)
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.sampling import stratified_axis_sample
from stereacl.token_alignment import AlignedPair, filter_aligned_pairs


LATAM_IDENTITIES = {
    "Argentine",
    "Bolivian",
    "Brazilian",
    "Chilean",
    "Colombian",
    "Costa Rican",
    "Cuban",
    "Dominican",
    "Ecuadorian",
    "Guatemalan",
    "Honduran",
    "Hondurans",
    "Mexican",
    "Nicaraguan",
    "Panamanian",
    "Paraguayan",
    "Peruvian",
    "Puerto Rican",
    "Salvadoran",
    "Dominicans",
    "Costa Ricans",
    "Uruguayan",
    "Venezuelan",
}

SOUTH_ASIA_IDENTITIES = {
    "Afghan",
    "Afghans",
    "Bangladeshi",
    "Bhutanese",
    "Indian",
    "Maldivian",
    "Maldivians",
    "Nepali",
    "Nepalese",
    "Pakistani",
    "Sri Lankan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 05: cross-cultural direction and component-shift analysis."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--pairs-per-culture", type=int, default=150)
    parser.add_argument("--per-source-limit", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k-components", type=int, default=20)
    parser.add_argument(
        "--seegull-pairs-per-identity",
        type=int,
        default=4,
        help="Number of stereotype/anti pairs to construct per SeeGULL identity.",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--cosine-bootstrap-n",
        type=int,
        default=500,
        help="Bootstrap resamples for cross-culture direction cosine CIs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def _build_culture_pairs(args: argparse.Namespace) -> dict[str, list[ContrastPair]]:
    culture_pairs: dict[str, list[ContrastPair]] = {}

    us_pairs = []
    us_pairs.extend(load_stereoset_intrasentence_pairs(limit=args.per_source_limit))
    us_pairs.extend(load_crows_pairs(limit=args.per_source_limit))
    culture_pairs["us_english"] = us_pairs

    latam_pairs = load_seegull_pairs(
        limit=args.per_source_limit,
        include_identities=LATAM_IDENTITIES,
        pairs_per_identity=args.seegull_pairs_per_identity,
    )
    culture_pairs["latam_spanish_proxy"] = latam_pairs

    south_asia_pairs = load_seegull_pairs(
        limit=args.per_source_limit,
        include_identities=SOUTH_ASIA_IDENTITIES,
        pairs_per_identity=args.seegull_pairs_per_identity,
    )
    south_asia_pairs.extend(
        load_seegull_indian_state_pairs(
            limit=args.per_source_limit,
            pairs_per_identity=args.seegull_pairs_per_identity,
        )
    )
    culture_pairs["south_asia_hindi_proxy"] = south_asia_pairs
    return culture_pairs


def _compute_culture_outputs(
    aligned_pairs: list[AlignedPair],
    bundle,
    head_specs: dict[int, AttentionProjectionSpec],
    max_length: int,
) -> tuple[
    dict[int, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, tuple[np.ndarray, np.ndarray]],
]:
    unembed = extract_unembedding_matrix(bundle.model).to(bundle.device).float()
    stereo_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    anti_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    component_scores: dict[tuple[int, str, str], list[float]] = defaultdict(list)

    for pair in aligned_pairs:
        encoded_s = encode_text(
            tokenizer=bundle.tokenizer,
            text=pair.pair.stereotype_text,
            device=bundle.device,
            max_length=max_length,
        )
        encoded_a = encode_text(
            tokenizer=bundle.tokenizer,
            text=pair.pair.antistereotype_text,
            device=bundle.device,
            max_length=max_length,
        )
        cap_s = forward_with_component_capture(
            model=bundle.model,
            encoded_inputs=encoded_s,
            output_hidden_states=True,
            capture_attention=True,
            capture_mlp=True,
        )
        cap_a = forward_with_component_capture(
            model=bundle.model,
            encoded_inputs=encoded_a,
            output_hidden_states=True,
            capture_attention=True,
            capture_mlp=True,
        )
        _, preproj_s = _forward_with_preproj_capture(bundle.model, encoded_s, specs=head_specs)
        _, preproj_a = _forward_with_preproj_capture(bundle.model, encoded_a, specs=head_specs)

        pos = pair.prediction_position
        direction_pos = pair.trait_token_position
        if pos >= cap_s.logits.shape[1] or pos >= cap_a.logits.shape[1]:
            continue

        u = (unembed[:, pair.stereo_token] - unembed[:, pair.anti_token]).float()

        for layer_idx, hidden in enumerate(cap_s.hidden_states[1:], start=1):
            if direction_pos >= hidden.shape[1] or direction_pos >= cap_a.hidden_states[layer_idx].shape[1]:
                continue
            stereo_by_layer[layer_idx].append(hidden[0, direction_pos, :].detach().float().cpu().numpy())
            anti_by_layer[layer_idx].append(
                cap_a.hidden_states[layer_idx][0, direction_pos, :].detach().float().cpu().numpy()
            )

        for layer_zero_idx in sorted(set(cap_s.mlp_outputs) & set(cap_a.mlp_outputs)):
            layer = layer_zero_idx + 1
            mlp_s = cap_s.mlp_outputs[layer_zero_idx]
            mlp_a = cap_a.mlp_outputs[layer_zero_idx]
            if pos >= mlp_s.shape[1] or pos >= mlp_a.shape[1]:
                continue
            write_vec = 0.5 * (mlp_s[0, pos, :].float() + mlp_a[0, pos, :].float())
            score = float(torch.dot(write_vec, u).detach().cpu())
            component_scores[(layer, "mlp_block", f"L{layer}")].append(score)

        for layer_zero_idx, spec in head_specs.items():
            if layer_zero_idx not in preproj_s or layer_zero_idx not in preproj_a:
                continue
            pre_s = preproj_s[layer_zero_idx]
            pre_a = preproj_a[layer_zero_idx]
            if pos >= pre_s.shape[1] or pos >= pre_a.shape[1]:
                continue
            head_writes = attention_head_writes_from_preproj(0.5 * (pre_s + pre_a), spec=spec)
            layer = layer_zero_idx + 1
            for head_idx in range(spec.num_heads):
                write_vec = head_writes[0, pos, head_idx, :].float()
                score = float(torch.dot(write_vec, u).detach().cpu())
                component_scores[(layer, "attention_head", f"L{layer}H{head_idx}")].append(score)

    directions: dict[int, np.ndarray] = {}
    for layer in sorted(set(stereo_by_layer) & set(anti_by_layer)):
        if len(stereo_by_layer[layer]) < 2 or len(anti_by_layer[layer]) < 2:
            continue
        stereo = np.stack(stereo_by_layer[layer])
        anti = np.stack(anti_by_layer[layer])
        directions[layer] = compute_direction(stereo, anti)

    rows: list[dict[str, Any]] = []
    for key in sorted(component_scores):
        layer, component_type, component_id = key
        vals = component_scores[key]
        rows.append(
            {
                "layer": layer,
                "component_type": component_type,
                "component_id": component_id,
                "mean_dla_score": round(float(np.mean(vals)), 8),
                "mean_abs_dla_score": round(float(np.mean(np.abs(vals))), 8),
                "n_pairs": len(vals),
            }
        )
    top_rows = sorted(rows, key=lambda r: float(r["mean_abs_dla_score"]), reverse=True)

    layer_vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer in sorted(set(stereo_by_layer) & set(anti_by_layer)):
        s = stereo_by_layer[layer]
        a = anti_by_layer[layer]
        if len(s) < 2 or len(a) < 2:
            continue
        layer_vectors[layer] = (np.stack(s), np.stack(a))

    return directions, rows, top_rows, layer_vectors


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def _bootstrap_mean_cosine_ci(
    *,
    layers_a: dict[int, tuple[np.ndarray, np.ndarray]],
    layers_b: dict[int, tuple[np.ndarray, np.ndarray]],
    common_layers: list[int],
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float] | tuple[None, None]:
    if not common_layers or n_resamples <= 0:
        return (None, None)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        layer_cosines: list[float] = []
        for layer in common_layers:
            s_a, a_a = layers_a[layer]
            s_b, a_b = layers_b[layer]
            idx_sa = rng.integers(0, s_a.shape[0], size=s_a.shape[0])
            idx_aa = rng.integers(0, a_a.shape[0], size=a_a.shape[0])
            idx_sb = rng.integers(0, s_b.shape[0], size=s_b.shape[0])
            idx_ab = rng.integers(0, a_b.shape[0], size=a_b.shape[0])
            dir_a = compute_direction(s_a[idx_sa], a_a[idx_aa])
            dir_b = compute_direction(s_b[idx_sb], a_b[idx_ab])
            layer_cosines.append(cosine_similarity(dir_a, dir_b))
        draws[i] = float(np.mean(layer_cosines)) if layer_cosines else float("nan")
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return (None, None)
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def main() -> None:
    args = parse_args()
    ctx = start_run("05", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        head_specs = build_attention_projection_specs(bundle.model)
        raw_by_culture = _build_culture_pairs(args)

        raw_jsonl_by_culture: dict[str, Path] = {}
        aligned_by_culture: dict[str, list[AlignedPair]] = {}
        alignment_stats: dict[str, dict[str, Any]] = {}
        pair_count_rows: list[dict[str, Any]] = []

        for culture, pairs in raw_by_culture.items():
            raw_path = ctx.artifacts_dir / f"{culture}_raw_pairs.jsonl"
            save_pairs_jsonl(pairs, raw_path)
            raw_jsonl_by_culture[culture] = raw_path
            aligned, stats = filter_aligned_pairs(pairs, tokenizer=bundle.tokenizer)
            aligned = stratified_axis_sample(aligned, limit=args.pairs_per_culture, seed=args.seed)
            aligned_by_culture[culture] = aligned
            alignment_stats[culture] = stats
            pair_count_rows.append(
                {
                    "culture": culture,
                    "pairs_raw": len(pairs),
                    "pairs_aligned": len(aligned),
                    "axes_count": len({p.pair.axis for p in aligned}),
                }
            )

        pair_counts_path = ctx.tables_dir / "culture_pair_counts.csv"
        write_csv(
            pair_counts_path,
            rows=pair_count_rows,
            fieldnames=["culture", "pairs_raw", "pairs_aligned", "axes_count"],
        )
        ctx.register_artifact(pair_counts_path, artifact_type="table", description="Per-culture pair counts.")

        align_stats_path = ctx.artifacts_dir / "alignment_stats_by_culture.json"
        write_json(align_stats_path, alignment_stats)
        ctx.register_artifact(align_stats_path, artifact_type="artifact", description="Alignment stats by culture.")

        for culture, path in raw_jsonl_by_culture.items():
            ctx.register_artifact(
                path,
                artifact_type="artifact",
                description=f"Raw contrast pairs for culture={culture}.",
            )

        if args.dry_run:
            metrics = {
                "cultures_compared": len(aligned_by_culture),
                "pairs_per_culture": args.pairs_per_culture,
                "total_aligned_pairs": sum(len(v) for v in aligned_by_culture.values()),
                "dry_run": True,
            }
            complete_run(ctx, metrics=metrics)
            return

        directions_by_culture: dict[str, dict[int, np.ndarray]] = {}
        layer_vectors_by_culture: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        component_rows: list[dict[str, Any]] = []
        top_rows: list[dict[str, Any]] = []

        for culture, aligned_pairs in aligned_by_culture.items():
            directions, rows, ranked, layer_vectors = _compute_culture_outputs(
                aligned_pairs=aligned_pairs,
                bundle=bundle,
                head_specs=head_specs,
                max_length=args.max_length,
            )
            directions_by_culture[culture] = directions
            layer_vectors_by_culture[culture] = layer_vectors
            for row in rows:
                component_rows.append({"culture": culture, **row})
            for rank, row in enumerate(ranked[: args.top_k_components], start=1):
                top_rows.append({"culture": culture, "rank": rank, **row})

        component_path = ctx.tables_dir / "culture_component_scores.csv"
        write_csv(
            component_path,
            rows=component_rows,
            fieldnames=[
                "culture",
                "layer",
                "component_type",
                "component_id",
                "mean_dla_score",
                "mean_abs_dla_score",
                "n_pairs",
            ],
        )
        ctx.register_artifact(component_path, artifact_type="table", description="Per-culture component DLA scores.")

        top_path = ctx.tables_dir / "culture_top_components.csv"
        write_csv(
            top_path,
            rows=top_rows,
            fieldnames=[
                "culture",
                "rank",
                "layer",
                "component_type",
                "component_id",
                "mean_dla_score",
                "mean_abs_dla_score",
                "n_pairs",
            ],
        )
        ctx.register_artifact(top_path, artifact_type="table", description="Top-K components per culture.")

        # Direction similarity across cultures.
        direction_rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(args.seed + 10_005)
        for culture_a, culture_b in itertools.combinations(sorted(directions_by_culture), 2):
            dirs_a = directions_by_culture[culture_a]
            dirs_b = directions_by_culture[culture_b]
            common_layers = sorted(set(dirs_a) & set(dirs_b))
            layer_cosines: list[float] = []
            for layer in common_layers:
                layer_cosines.append(cosine_similarity(dirs_a[layer], dirs_b[layer]))
            ci_lo, ci_hi = _bootstrap_mean_cosine_ci(
                layers_a=layer_vectors_by_culture.get(culture_a, {}),
                layers_b=layer_vectors_by_culture.get(culture_b, {}),
                common_layers=common_layers,
                n_resamples=args.cosine_bootstrap_n,
                rng=rng,
            )
            direction_rows.append(
                {
                    "culture_a": culture_a,
                    "culture_b": culture_b,
                    "mean_cosine_similarity": round(float(np.mean(layer_cosines)), 8)
                    if layer_cosines
                    else "",
                    "min_cosine_similarity": round(float(np.min(layer_cosines)), 8)
                    if layer_cosines
                    else "",
                    "max_cosine_similarity": round(float(np.max(layer_cosines)), 8)
                    if layer_cosines
                    else "",
                    "mean_cosine_ci_low": round(float(ci_lo), 8) if ci_lo is not None else "",
                    "mean_cosine_ci_high": round(float(ci_hi), 8) if ci_hi is not None else "",
                    "n_common_layers": len(common_layers),
                }
            )

        sim_path = ctx.tables_dir / "direction_similarity.csv"
        write_csv(
            sim_path,
            rows=direction_rows,
            fieldnames=[
                "culture_a",
                "culture_b",
                "mean_cosine_similarity",
                "min_cosine_similarity",
                "max_cosine_similarity",
                "mean_cosine_ci_low",
                "mean_cosine_ci_high",
                "n_common_layers",
            ],
        )
        ctx.register_artifact(sim_path, artifact_type="table", description="Cross-culture direction similarity.")

        # Top-component overlap.
        top_sets: dict[str, set[str]] = defaultdict(set)
        for row in top_rows:
            culture = str(row["culture"])
            comp_key = f"{row['component_type']}::{row['component_id']}"
            top_sets[culture].add(comp_key)
        overlap_rows: list[dict[str, Any]] = []
        for culture_a, culture_b in itertools.combinations(sorted(top_sets), 2):
            overlap_rows.append(
                {
                    "culture_a": culture_a,
                    "culture_b": culture_b,
                    "jaccard_overlap": round(_jaccard(top_sets[culture_a], top_sets[culture_b]), 8),
                    "intersection_size": len(top_sets[culture_a] & top_sets[culture_b]),
                    "union_size": len(top_sets[culture_a] | top_sets[culture_b]),
                }
            )

        overlap_path = ctx.tables_dir / "top_component_overlap.csv"
        write_csv(
            overlap_path,
            rows=overlap_rows,
            fieldnames=[
                "culture_a",
                "culture_b",
                "jaccard_overlap",
                "intersection_size",
                "union_size",
            ],
        )
        ctx.register_artifact(overlap_path, artifact_type="table", description="Cross-culture top-component overlap.")

        directions_npz_path = ctx.artifacts_dir / "culture_directions.json"
        write_json(
            directions_npz_path,
            {
                culture: {
                    str(layer): {
                        "norm": float(np.linalg.norm(vec)),
                    }
                    for layer, vec in dirs.items()
                }
                for culture, dirs in directions_by_culture.items()
            },
        )
        ctx.register_artifact(
            directions_npz_path,
            artifact_type="artifact",
            description="Direction metadata by culture/layer.",
        )

        metrics = {
            "cultures_compared": len(aligned_by_culture),
            "pairs_per_culture": args.pairs_per_culture,
            "total_aligned_pairs": sum(len(v) for v in aligned_by_culture.values()),
            "component_rows": len(component_rows),
            "top_component_rows": len(top_rows),
            "direction_similarity_rows": len(direction_rows),
            "pairwise_comparisons": len(overlap_rows),
            "dry_run": False,
        }
        complete_run(ctx, metrics=metrics)
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
