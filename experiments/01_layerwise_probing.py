#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import (
    LayerAxisBuckets,
    compute_direction,
    cosine_similarity,
    fit_logistic_probe_auc,
    save_directions_npz,
    write_csv,
    write_json,
)
from stereacl.data import (
    ContrastPair,
    build_contrast_pairs,
    deterministic_split_indices,
    save_pairs_jsonl,
    summarize_pairs,
)
from stereacl.modeling import ModelBundle, encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.sampling import stratified_axis_sample
from stereacl.token_alignment import AlignedPair, filter_aligned_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 01: layer-wise probing and stereotype-direction extraction."
    )
    parser.add_argument("--model", default="gpt2", help="HF causal LM model name.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype or auto.")
    parser.add_argument("--pairs-limit", type=int, default=300, help="Final aligned pair limit.")
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=1200,
        help="Raw pair limit per dataset source before token filtering.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--direction-position",
        choices=["trait", "prediction"],
        default="trait",
        help=(
            "Token position used to collect hidden vectors for direction extraction. "
            "'trait' uses trait_token_position; 'prediction' uses prediction_position."
        ),
    )
    parser.add_argument("--no-stereoset", action="store_true")
    parser.add_argument("--no-crows", action="store_true")
    parser.add_argument("--no-seegull", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _save_aligned_pairs(path: Path, pairs: list[AlignedPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in pairs:
            payload = {
                "pair_id": item.pair.pair_id,
                "source": item.pair.source,
                "axis": item.pair.axis,
                "stereotype_text": item.pair.stereotype_text,
                "antistereotype_text": item.pair.antistereotype_text,
                "stereo_token": item.stereo_token,
                "anti_token": item.anti_token,
                "trait_token_position": item.trait_token_position,
                "prediction_position": item.prediction_position,
                "stereo_input_ids": item.stereo_input_ids,
                "anti_input_ids": item.anti_input_ids,
                "metadata": item.pair.metadata,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _collect_hidden_vectors(
    train_pairs: list[AlignedPair],
    bundle: ModelBundle,
    max_length: int,
    direction_position: str,
) -> tuple[LayerAxisBuckets, int]:
    buckets = LayerAxisBuckets.empty()
    processed = 0

    for pair in train_pairs:
        for is_stereo in (True, False):
            text = pair.pair.stereotype_text if is_stereo else pair.pair.antistereotype_text
            encoded = encode_text(
                tokenizer=bundle.tokenizer,
                text=text,
                device=bundle.device,
                max_length=max_length,
            )
            capture = forward_with_component_capture(
                model=bundle.model,
                encoded_inputs=encoded,
                capture_attention=False,
                capture_mlp=False,
                output_hidden_states=True,
            )
            if not capture.hidden_states:
                continue
            if direction_position == "prediction":
                position = pair.prediction_position
            else:
                # Default: trait-token position. On strictly causal models,
                # prediction_position can be identical across stereo/anti
                # templated pairs, collapsing difference vectors to zero.
                position = pair.trait_token_position
            for layer_idx, hidden in enumerate(capture.hidden_states[1:], start=1):
                if position >= hidden.shape[1]:
                    continue
                vec = hidden[0, position, :].detach().float().cpu().numpy()
                axis = pair.pair.axis
                source = pair.pair.source
                if is_stereo:
                    buckets.stereo[axis][layer_idx].append(vec)
                    buckets.source_stereo[source][axis][layer_idx].append(vec)
                else:
                    buckets.anti[axis][layer_idx].append(vec)
                    buckets.source_anti[source][axis][layer_idx].append(vec)
        processed += 1
    return buckets, processed


def _compute_outputs(
    buckets: LayerAxisBuckets,
    output_dir: Path,
    seed: int,
) -> tuple[dict[tuple[str, int], np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    directions: dict[tuple[str, int], np.ndarray] = {}
    auc_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []

    axes = sorted(set(buckets.stereo) | set(buckets.anti))
    for axis in axes:
        layers = sorted(set(buckets.stereo[axis]) | set(buckets.anti[axis]))
        for layer in layers:
            stereo_list = buckets.stereo[axis][layer]
            anti_list = buckets.anti[axis][layer]
            if len(stereo_list) < 2 or len(anti_list) < 2:
                continue
            stereo = np.stack(stereo_list)
            anti = np.stack(anti_list)
            direction = compute_direction(stereo, anti)
            auc = fit_logistic_probe_auc(stereo, anti, seed=seed)
            directions[(axis, layer)] = direction.astype(np.float32)
            auc_rows.append(
                {
                    "axis": axis,
                    "layer": layer,
                    "auc": round(float(auc), 6) if auc is not None else "",
                    "n_stereo": len(stereo),
                    "n_anti": len(anti),
                }
            )
            direction_rows.append(
                {
                    "axis": axis,
                    "layer": layer,
                    "direction_norm": round(float(np.linalg.norm(direction)), 6),
                }
            )

    sources = sorted(set(buckets.source_stereo) | set(buckets.source_anti))
    for axis, layer in directions:
        per_source_dir: dict[str, np.ndarray] = {}
        for source in sources:
            stereo_list = buckets.source_stereo[source][axis][layer]
            anti_list = buckets.source_anti[source][axis][layer]
            if len(stereo_list) < 2 or len(anti_list) < 2:
                continue
            stereo = np.stack(stereo_list)
            anti = np.stack(anti_list)
            per_source_dir[source] = compute_direction(stereo, anti)
        for source_a, source_b in itertools.combinations(sorted(per_source_dir), 2):
            cos = cosine_similarity(per_source_dir[source_a], per_source_dir[source_b])
            cosine_rows.append(
                {
                    "axis": axis,
                    "layer": layer,
                    "source_a": source_a,
                    "source_b": source_b,
                    "cosine_similarity": round(float(cos), 6),
                }
            )

    directions_path = output_dir / "directions_layerwise.npz"
    save_directions_npz(directions_path, directions)
    return directions, auc_rows, direction_rows, cosine_rows


def main() -> None:
    args = parse_args()
    ctx = start_run("01", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        include_stereoset = not args.no_stereoset
        include_crows = not args.no_crows
        include_seegull = not args.no_seegull

        pairs: list[ContrastPair] = build_contrast_pairs(
            include_stereoset=include_stereoset,
            include_crows=include_crows,
            include_seegull=include_seegull,
            per_source_limit=args.per_source_limit,
        )
        raw_pairs_path = ctx.artifacts_dir / "contrast_pairs_raw.jsonl"
        save_pairs_jsonl(pairs, raw_pairs_path)
        ctx.register_artifact(raw_pairs_path, artifact_type="artifact", description="Raw contrast pairs.")

        pair_summary = summarize_pairs(pairs)
        pair_summary_path = ctx.tables_dir / "pair_counts_by_source_axis.csv"
        pair_summary.to_csv(pair_summary_path, index=False)
        ctx.register_artifact(
            pair_summary_path,
            artifact_type="table",
            description="Pair counts by dataset source and axis.",
        )

        # Build tokenizer from selected model to enforce single-token-difference filtering.
        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        aligned_pairs, align_stats = filter_aligned_pairs(pairs, tokenizer=bundle.tokenizer)
        aligned_pairs = stratified_axis_sample(aligned_pairs, limit=args.pairs_limit, seed=args.seed)

        aligned_path = ctx.artifacts_dir / "aligned_pairs.jsonl"
        _save_aligned_pairs(aligned_path, aligned_pairs)
        ctx.register_artifact(
            aligned_path,
            artifact_type="artifact",
            description="Model-tokenizer-aligned single-token contrast pairs.",
        )
        write_json(ctx.artifacts_dir / "alignment_stats.json", align_stats)
        ctx.register_artifact(
            ctx.artifacts_dir / "alignment_stats.json",
            artifact_type="artifact",
            description="Alignment and filtering statistics.",
        )

        train_idx, test_idx = deterministic_split_indices(
            n_items=len(aligned_pairs),
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        train_pairs = [aligned_pairs[i] for i in train_idx]
        test_pairs = [aligned_pairs[i] for i in test_idx]
        split_path = ctx.artifacts_dir / "train_test_split.json"
        write_json(
            split_path,
            {
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "n_train": len(train_pairs),
                "n_test": len(test_pairs),
            },
        )
        ctx.register_artifact(split_path, artifact_type="artifact", description="Deterministic train/test split.")

        if args.dry_run:
            metrics = {
                "pairs_raw": len(pairs),
                "pairs_aligned": len(aligned_pairs),
                "pairs_train": len(train_pairs),
                "pairs_test": len(test_pairs),
                "axes_count": int(pair_summary["axis"].nunique()) if not pair_summary.empty else 0,
                "dry_run": True,
            }
            complete_run(ctx, metrics=metrics)
            return

        buckets, processed_pairs = _collect_hidden_vectors(
            train_pairs=train_pairs,
            bundle=bundle,
            max_length=args.max_length,
            direction_position=args.direction_position,
        )
        directions, auc_rows, direction_rows, cosine_rows = _compute_outputs(
            buckets=buckets,
            output_dir=ctx.artifacts_dir,
            seed=args.seed,
        )

        auc_path = ctx.tables_dir / "layer_probe_auc.csv"
        write_csv(
            auc_path,
            rows=sorted(auc_rows, key=lambda r: (str(r["axis"]), int(r["layer"]))),
            fieldnames=["axis", "layer", "auc", "n_stereo", "n_anti"],
        )
        ctx.register_artifact(auc_path, artifact_type="table", description="Layer-wise probe AUC by axis.")

        direction_path = ctx.tables_dir / "layer_direction_norm.csv"
        write_csv(
            direction_path,
            rows=sorted(direction_rows, key=lambda r: (str(r["axis"]), int(r["layer"]))),
            fieldnames=["axis", "layer", "direction_norm"],
        )
        ctx.register_artifact(direction_path, artifact_type="table", description="Layer direction norms.")

        cosine_path = ctx.tables_dir / "cross_dataset_direction_cosine.csv"
        write_csv(
            cosine_path,
            rows=sorted(
                cosine_rows,
                key=lambda r: (str(r["axis"]), int(r["layer"]), str(r["source_a"]), str(r["source_b"])),
            ),
            fieldnames=["axis", "layer", "source_a", "source_b", "cosine_similarity"],
        )
        ctx.register_artifact(
            cosine_path,
            artifact_type="table",
            description="Cross-dataset direction cosine similarities.",
        )

        directions_path = ctx.artifacts_dir / "directions_layerwise.npz"
        ctx.register_artifact(
            directions_path,
            artifact_type="artifact",
            description="Saved stereotype direction vectors per (axis, layer).",
        )

        metrics = {
            "pairs_raw": len(pairs),
            "pairs_aligned": len(aligned_pairs),
            "pairs_train": len(train_pairs),
            "pairs_test": len(test_pairs),
            "pairs_processed": processed_pairs,
            "directions_computed": len(directions),
            "auc_rows": len(auc_rows),
            "cross_dataset_cosine_rows": len(cosine_rows),
            "dry_run": False,
        }
        complete_run(ctx, metrics=metrics)
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
