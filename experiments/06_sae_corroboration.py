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
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import cosine_similarity, load_directions_npz, write_csv, write_json
from stereacl.attention_heads import (
    AttentionProjectionSpec,
    attention_head_writes_from_preproj,
    build_attention_projection_specs,
)
from stereacl.data import ContrastPair
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.sampling import stratified_axis_sample
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 06: sparse-feature corroboration for localized components."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--top-components", type=int, default=30)
    parser.add_argument("--pairs-limit", type=int, default=120)
    parser.add_argument("--dictionary-features", type=int, default=128)
    parser.add_argument("--dictionary-alpha", type=float, default=1.0)
    parser.add_argument("--dictionary-max-iter", type=int, default=400)
    parser.add_argument("--top-features-per-component", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
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
        ended = payload.get("ended_at_utc") or ""
        run_dir = Path(payload["run_dir"])
        if required_relpaths:
            missing = False
            for rel in required_relpaths:
                if not (run_dir / rel).exists():
                    missing = True
                    break
            if missing:
                continue
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        required_msg = f" with required files {required_relpaths}" if required_relpaths else ""
        raise FileNotFoundError(f"No completed run found for {experiment_slug}{required_msg}.")
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


def _parse_component_row(row: pd.Series) -> tuple[str, int, int | None, str]:
    component_type = str(row["component_type"])
    layer = int(row["layer"])
    component_id = str(row["component_id"])
    head_index: int | None = None
    if component_type == "attention_head":
        if "H" not in component_id:
            raise ValueError(f"Cannot parse head index from {component_id}")
        head_index = int(component_id.split("H")[1])
    return component_type, layer, head_index, component_id


def _component_key(component_type: str, layer: int, head_index: int | None) -> str:
    if component_type == "attention_head":
        assert head_index is not None
        return f"{component_type}::L{layer}H{head_index}"
    return f"{component_type}::L{layer}"


def main() -> None:
    args = parse_args()
    ctx = start_run("06", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = _latest_run_dir(
            "01_layerwise_probing",
            required_relpaths=["artifacts/aligned_pairs.jsonl", "artifacts/directions_layerwise.npz"],
            model_name=args.model,
        )
        exp2_dir = _latest_run_dir(
            "02_component_dla",
            required_relpaths=["tables/top_components.csv"],
            model_name=args.model,
        )

        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        aligned_pairs = stratified_axis_sample(aligned_pairs, limit=args.pairs_limit, seed=args.seed)
        directions = load_directions_npz(exp1_dir / "artifacts" / "directions_layerwise.npz")

        component_table = pd.read_csv(exp2_dir / "tables" / "top_components.csv")
        if component_table.empty:
            raise ValueError("Experiment 02 top_components.csv is empty.")
        component_table = component_table.sort_values("mean_abs_dla_score", ascending=False)

        selected_components: list[tuple[str, int, int | None, str]] = []
        seen: set[str] = set()
        for _, row in component_table.iterrows():
            ctype, layer, head_idx, cid = _parse_component_row(row)
            key = _component_key(ctype, layer, head_idx)
            if key in seen:
                continue
            selected_components.append((ctype, layer, head_idx, cid))
            seen.add(key)
            if len(selected_components) >= args.top_components:
                break

        dep_path = ctx.artifacts_dir / "dependencies.json"
        write_json(
            dep_path,
            {
                "exp1_run_dir": str(exp1_dir),
                "exp2_run_dir": str(exp2_dir),
                "selected_components": [
                    {
                        "component_type": ctype,
                        "layer": layer,
                        "head_index": head_idx,
                        "component_id": cid,
                    }
                    for (ctype, layer, head_idx, cid) in selected_components
                ],
                "pairs_selected": len(aligned_pairs),
            },
        )
        ctx.register_artifact(dep_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            metrics = {
                "components_selected": len(selected_components),
                "pairs_loaded": len(aligned_pairs),
                "dry_run": True,
            }
            complete_run(ctx, metrics=metrics)
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        head_specs = build_attention_projection_specs(bundle.model)

        sample_vectors: list[np.ndarray] = []
        sample_meta: list[dict[str, Any]] = []

        for pair in aligned_pairs:
            encoded_s = encode_text(
                tokenizer=bundle.tokenizer,
                text=pair.pair.stereotype_text,
                device=bundle.device,
                max_length=args.max_length,
            )
            encoded_a = encode_text(
                tokenizer=bundle.tokenizer,
                text=pair.pair.antistereotype_text,
                device=bundle.device,
                max_length=args.max_length,
            )
            cap_s = forward_with_component_capture(
                model=bundle.model,
                encoded_inputs=encoded_s,
                output_hidden_states=False,
                capture_attention=True,
                capture_mlp=True,
            )
            cap_a = forward_with_component_capture(
                model=bundle.model,
                encoded_inputs=encoded_a,
                output_hidden_states=False,
                capture_attention=True,
                capture_mlp=True,
            )
            _, pre_s = _forward_with_preproj_capture(bundle.model, encoded_s, specs=head_specs)
            _, pre_a = _forward_with_preproj_capture(bundle.model, encoded_a, specs=head_specs)

            pos = pair.prediction_position
            if pos >= cap_s.logits.shape[1] or pos >= cap_a.logits.shape[1]:
                continue

            for component_type, layer, head_idx, component_id in selected_components:
                layer_zero = layer - 1
                vec_tensor: torch.Tensor | None = None
                if component_type == "mlp_block":
                    if layer_zero not in cap_s.mlp_outputs or layer_zero not in cap_a.mlp_outputs:
                        continue
                    ms = cap_s.mlp_outputs[layer_zero]
                    ma = cap_a.mlp_outputs[layer_zero]
                    if pos >= ms.shape[1] or pos >= ma.shape[1]:
                        continue
                    vec_tensor = 0.5 * (ms[0, pos, :].float() + ma[0, pos, :].float())
                elif component_type == "attention_block":
                    if layer_zero not in cap_s.attention_outputs or layer_zero not in cap_a.attention_outputs:
                        continue
                    ats = cap_s.attention_outputs[layer_zero]
                    ata = cap_a.attention_outputs[layer_zero]
                    if pos >= ats.shape[1] or pos >= ata.shape[1]:
                        continue
                    vec_tensor = 0.5 * (ats[0, pos, :].float() + ata[0, pos, :].float())
                elif component_type == "attention_head":
                    if head_idx is None:
                        continue
                    if layer_zero not in pre_s or layer_zero not in pre_a or layer_zero not in head_specs:
                        continue
                    ps = pre_s[layer_zero]
                    pa = pre_a[layer_zero]
                    if pos >= ps.shape[1] or pos >= pa.shape[1]:
                        continue
                    spec = head_specs[layer_zero]
                    head_writes = attention_head_writes_from_preproj(0.5 * (ps + pa), spec=spec)
                    if head_idx >= head_writes.shape[2]:
                        continue
                    vec_tensor = head_writes[0, pos, head_idx, :].float()
                else:
                    continue

                if vec_tensor is None:
                    continue
                vec = vec_tensor.detach().cpu().numpy()
                sample_vectors.append(vec)
                sample_meta.append(
                    {
                        "pair_id": pair.pair.pair_id,
                        "axis": pair.pair.axis,
                        "source": pair.pair.source,
                        "component_type": component_type,
                        "component_id": component_id,
                        "layer": layer,
                        "head_index": head_idx if head_idx is not None else "",
                    }
                )

        if not sample_vectors:
            raise ValueError("No activation vectors collected for sparse feature corroboration.")

        X = np.stack(sample_vectors)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xz = scaler.fit_transform(X)

        dict_model = DictionaryLearning(
            n_components=args.dictionary_features,
            alpha=args.dictionary_alpha,
            max_iter=args.dictionary_max_iter,
            transform_algorithm="lasso_lars",
            random_state=args.seed,
        )
        codes = dict_model.fit_transform(Xz)
        atoms = dict_model.components_  # [n_features, d_model]

        # Per-component feature salience.
        grouped_codes: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_meta: dict[str, dict[str, Any]] = {}
        for row_meta, code in zip(sample_meta, codes):
            key = f"{row_meta['component_type']}::{row_meta['component_id']}"
            grouped_codes[key].append(np.abs(code))
            grouped_meta[key] = row_meta

        correspond_rows: list[dict[str, Any]] = []
        for comp_key in sorted(grouped_codes):
            abs_codes = np.stack(grouped_codes[comp_key])
            mean_abs = abs_codes.mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][: args.top_features_per_component]
            meta = grouped_meta[comp_key]
            for rank, feature_idx in enumerate(top_idx, start=1):
                correspond_rows.append(
                    {
                        "component_id": meta["component_id"],
                        "component_type": meta["component_type"],
                        "sae_layer": meta["layer"],
                        "head_index": meta["head_index"],
                        "feature_rank": rank,
                        "feature_id": int(feature_idx),
                        "activation_score": round(float(mean_abs[feature_idx]), 8),
                        "annotation_hint": "dictionary-learning sparse feature",
                    }
                )

        correspond_path = ctx.tables_dir / "sae_feature_correspondence.csv"
        write_csv(
            correspond_path,
            rows=correspond_rows,
            fieldnames=[
                "component_id",
                "component_type",
                "sae_layer",
                "head_index",
                "feature_rank",
                "feature_id",
                "activation_score",
                "annotation_hint",
            ],
        )
        ctx.register_artifact(
            correspond_path,
            artifact_type="table",
            description="Sparse-feature correspondence (SAE proxy) by component.",
        )

        # Correlate learned atoms with Exp1 directions by layer.
        atom_rows: list[dict[str, Any]] = []
        for layer in sorted({int(m["layer"]) for m in sample_meta}):
            layer_dirs = [vec for (axis, lyr), vec in directions.items() if lyr == layer]
            if not layer_dirs:
                continue
            mean_direction = np.mean(np.stack(layer_dirs), axis=0)
            for feature_idx, atom in enumerate(atoms):
                atom_rows.append(
                    {
                        "layer": layer,
                        "feature_id": feature_idx,
                        "cosine_to_layer_direction": round(cosine_similarity(atom, mean_direction), 8),
                    }
                )

        atom_path = ctx.tables_dir / "dictionary_atoms_vs_directions.csv"
        write_csv(
            atom_path,
            rows=atom_rows,
            fieldnames=["layer", "feature_id", "cosine_to_layer_direction"],
        )
        ctx.register_artifact(
            atom_path,
            artifact_type="table",
            description="Dictionary atom cosine similarity against Exp1 layer directions.",
        )

        model_artifact = ctx.artifacts_dir / "dictionary_model_summary.json"
        write_json(
            model_artifact,
            {
                "n_samples": int(X.shape[0]),
                "embedding_dim": int(X.shape[1]),
                "n_features": int(atoms.shape[0]),
                "reconstruction_mse": float(np.mean((Xz - codes @ atoms) ** 2)),
                "selected_components": len(selected_components),
            },
        )
        ctx.register_artifact(
            model_artifact,
            artifact_type="artifact",
            description="Dictionary-learning model summary.",
        )

        notes_path = ctx.artifacts_dir / "sae_lookup_notes.md"
        notes_path.write_text(
            "# SAE Corroboration Notes\n\n"
            "This run uses dictionary learning as a sparse-feature proxy for SAE corroboration "
            "when pre-trained SAE checkpoints are not wired in this environment.\n",
            encoding="utf-8",
        )
        ctx.register_artifact(
            notes_path,
            artifact_type="artifact",
            description="Run notes for sparse-feature corroboration method.",
        )

        metrics = {
            "components_examined": len(selected_components),
            "pairs_loaded": len(aligned_pairs),
            "activation_samples": int(X.shape[0]),
            "dictionary_features": int(atoms.shape[0]),
            "correspondence_rows": len(correspond_rows),
            "dry_run": False,
        }
        complete_run(ctx, metrics=metrics)
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
