#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.interventions import make_replace_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.sampling import stratified_axis_sample
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 03: gradient attribution and activation patch validation."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--pairs-limit", type=int, default=220)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--validation-pairs-per-component", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--exp1-run-dir",
        default="",
        help="Explicit Experiment 01 run directory. If omitted, latest completed run is used.",
    )
    parser.add_argument(
        "--exp2-run-dir",
        default="",
        help="Explicit Experiment 02 run directory. If omitted, latest completed run is used.",
    )
    parser.add_argument(
        "--split-scope",
        choices=["all", "train", "test"],
        default="all",
        help="Which Exp01 split partition to sample from before attribution ranking/validation.",
    )
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


def _component_kind_and_layer(row: pd.Series) -> tuple[str, int]:
    component_type = str(row["component_type"])
    layer = int(row["layer"])
    if component_type not in {"attention_block", "mlp_block"}:
        raise ValueError(f"Unsupported component type from Experiment 02 for Exp03: {component_type}")
    return component_type, layer


def main() -> None:
    args = parse_args()
    ctx = start_run("03", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = (
            Path(args.exp1_run_dir)
            if args.exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=["artifacts/aligned_pairs.jsonl"],
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
        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_path = exp1_dir / "artifacts" / "train_test_split.json"
        if args.split_scope != "all" and split_path.exists():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            key = "train_indices" if args.split_scope == "train" else "test_indices"
            wanted = {int(i) for i in split.get(key, [])}
            aligned_pairs = [p for i, p in enumerate(aligned_pairs) if i in wanted]
        aligned_pairs = stratified_axis_sample(aligned_pairs, limit=args.pairs_limit, seed=args.seed)

        exp2_scores_path = exp2_dir / "tables" / "component_dla_scores.csv"
        exp2_scores = pd.read_csv(exp2_scores_path)
        exp2_scores = exp2_scores[exp2_scores["component_type"].isin(["attention_block", "mlp_block"])].copy()
        exp2_top = exp2_scores.sort_values(["axis", "mean_abs_dla_score"], ascending=[True, False])
        selected_components: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for axis, group in exp2_top.groupby("axis"):
            for _, row in group.head(args.top_k).iterrows():
                selected_components[str(axis)].append(_component_kind_and_layer(row))

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs_path,
            {
                "exp1_run_dir": str(exp1_dir),
                "exp2_run_dir": str(exp2_dir),
                "split_scope": args.split_scope,
                "pairs_selected": len(aligned_pairs),
                "selected_components_per_axis": {
                    axis: [{"type": ctype, "layer": layer} for (ctype, layer) in comps]
                    for axis, comps in selected_components.items()
                },
            },
        )
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            metrics = {
                "pairs_loaded": len(aligned_pairs),
                "axes_with_components": len(selected_components),
                "dry_run": True,
            }
            complete_run(ctx, metrics=metrics)
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)

        attributions: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        pair_to_axis_components: dict[str, list[tuple[str, int]]] = {}

        for pair in aligned_pairs:
            axis = pair.pair.axis
            comps = selected_components.get(axis, [])
            if not comps:
                continue
            pair_to_axis_components[pair.pair.pair_id] = comps

            encoded = encode_text(
                tokenizer=bundle.tokenizer,
                text=pair.pair.stereotype_text,
                device=bundle.device,
                max_length=args.max_length,
            )
            bundle.model.zero_grad(set_to_none=True)
            cap = forward_with_component_capture(
                model=bundle.model,
                encoded_inputs=encoded,
                require_grad=True,
                output_hidden_states=False,
                capture_attention=True,
                capture_mlp=True,
                retain_grad_on_captures=True,
            )
            # Use trait token position consistently with Exp01 direction extraction
            # and Exp02 DLA scoring to avoid causal degeneracy at the shared prefix.
            pos = pair.trait_token_position
            if pos >= cap.logits.shape[1]:
                continue
            score = cap.logits[0, pos, pair.stereo_token] - cap.logits[0, pos, pair.anti_token]
            score.backward()

            for component_type, layer in comps:
                idx = layer - 1
                if component_type == "attention_block":
                    tensor = cap.attention_outputs.get(idx)
                else:
                    tensor = cap.mlp_outputs.get(idx)
                if tensor is None or tensor.grad is None or pos >= tensor.shape[1]:
                    continue
                grad_vec = tensor.grad[0, pos, :].float()
                act_vec = tensor[0, pos, :].float()
                attr = float(torch.dot(grad_vec, act_vec).detach().cpu())
                attributions[(axis, component_type, layer)].append(attr)

        attr_rows: list[dict[str, Any]] = []
        for key in sorted(attributions):
            axis, component_type, layer = key
            vals = attributions[key]
            attr_rows.append(
                {
                    "axis": axis,
                    "component_type": component_type,
                    "layer": layer,
                    "component_id": f"L{layer}",
                    "mean_attr_score": round(float(np.mean(vals)), 8),
                    "mean_abs_attr_score": round(float(np.mean(np.abs(vals))), 8),
                    "std_attr_score": round(float(np.std(vals)), 8),
                    "n_observations": len(vals),
                }
            )

        attr_path = ctx.tables_dir / "attribution_patch_scores.csv"
        write_csv(
            attr_path,
            rows=attr_rows,
            fieldnames=[
                "axis",
                "component_type",
                "layer",
                "component_id",
                "mean_attr_score",
                "mean_abs_attr_score",
                "std_attr_score",
                "n_observations",
            ],
        )
        ctx.register_artifact(attr_path, artifact_type="table", description="Gradient attribution scores.")

        # Validate top components with direct activation replacement (anti -> stereo).
        ranked_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in attr_rows:
            ranked_by_axis[str(row["axis"])].append(row)
        for axis in ranked_by_axis:
            ranked_by_axis[axis].sort(key=lambda r: float(r["mean_abs_attr_score"]), reverse=True)

        validation_rows: list[dict[str, Any]] = []
        for axis, ranked_rows in ranked_by_axis.items():
            axis_pairs = [p for p in aligned_pairs if p.pair.axis == axis]
            for row in ranked_rows[: args.top_k]:
                component_type = str(row["component_type"])
                layer = int(row["layer"])
                idx = layer - 1
                deltas: list[float] = []
                for pair in axis_pairs[: args.validation_pairs_per_component]:
                    pos = pair.trait_token_position
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
                    cap_a = forward_with_component_capture(
                        model=bundle.model,
                        encoded_inputs=encoded_a,
                        output_hidden_states=False,
                        capture_attention=True,
                        capture_mlp=True,
                    )
                    if component_type == "attention_block":
                        source_tensor = cap_a.attention_outputs.get(idx)
                    else:
                        source_tensor = cap_a.mlp_outputs.get(idx)
                    if source_tensor is None or pos >= source_tensor.shape[1]:
                        continue
                    replacement = source_tensor[:, pos, :].detach()

                    baseline = forward_with_component_capture(
                        model=bundle.model,
                        encoded_inputs=encoded_s,
                        output_hidden_states=False,
                        capture_attention=False,
                        capture_mlp=False,
                    )
                    baseline_score = compute_score_from_logits(
                        baseline.logits,
                        position=pos,
                        pos_token=pair.stereo_token,
                        neg_token=pair.anti_token,
                    )

                    if component_type == "attention_block":
                        cap_patched = forward_with_component_capture(
                            model=bundle.model,
                            encoded_inputs=encoded_s,
                            output_hidden_states=False,
                            capture_attention=True,
                            capture_mlp=False,
                            attention_patch_map={idx: make_replace_position_hook(pos, replacement)},
                        )
                    else:
                        cap_patched = forward_with_component_capture(
                            model=bundle.model,
                            encoded_inputs=encoded_s,
                            output_hidden_states=False,
                            capture_attention=False,
                            capture_mlp=True,
                            mlp_patch_map={idx: make_replace_position_hook(pos, replacement)},
                        )
                    patched_score = compute_score_from_logits(
                        cap_patched.logits,
                        position=pos,
                        pos_token=pair.stereo_token,
                        neg_token=pair.anti_token,
                    )
                    deltas.append(baseline_score - patched_score)

                validation_rows.append(
                    {
                        "axis": axis,
                        "component_type": component_type,
                        "layer": layer,
                        "component_id": f"L{layer}",
                        "mean_delta_logit": round(float(np.mean(deltas)), 8) if deltas else "",
                        "mean_abs_delta_logit": round(float(np.mean(np.abs(deltas))), 8) if deltas else "",
                        "n_validation_pairs": len(deltas),
                    }
                )

        validation_path = ctx.tables_dir / "validation_delta_logit.csv"
        write_csv(
            validation_path,
            rows=validation_rows,
            fieldnames=[
                "axis",
                "component_type",
                "layer",
                "component_id",
                "mean_delta_logit",
                "mean_abs_delta_logit",
                "n_validation_pairs",
            ],
        )
        ctx.register_artifact(
            validation_path, artifact_type="table", description="Activation replacement validation deltas."
        )

        # DLA vs attribution ranking agreement.
        corr_rows: list[dict[str, Any]] = []
        exp2_scores["component_key"] = exp2_scores.apply(
            lambda r: f"{r['component_type']}::L{int(r['layer'])}",
            axis=1,
        )
        attr_df = pd.DataFrame(attr_rows)
        if not attr_df.empty:
            attr_df["component_key"] = attr_df.apply(
                lambda r: f"{r['component_type']}::L{int(r['layer'])}",
                axis=1,
            )
            for axis in sorted(set(exp2_scores["axis"]) & set(attr_df["axis"])):
                left = exp2_scores[exp2_scores["axis"] == axis][
                    ["component_key", "mean_abs_dla_score"]
                ].copy()
                right = attr_df[attr_df["axis"] == axis][["component_key", "mean_abs_attr_score"]].copy()
                merged = left.merge(right, on="component_key", how="inner")
                if len(merged) < 3:
                    rho = np.nan
                else:
                    rho = spearmanr(
                        merged["mean_abs_dla_score"].astype(float),
                        merged["mean_abs_attr_score"].astype(float),
                    ).correlation
                corr_rows.append(
                    {
                        "axis": axis,
                        "spearman_rho": round(float(rho), 8) if not np.isnan(rho) else "",
                        "n_components": len(merged),
                    }
                )

        corr_path = ctx.tables_dir / "dla_vs_atp_spearman.csv"
        write_csv(corr_path, corr_rows, fieldnames=["axis", "spearman_rho", "n_components"])
        ctx.register_artifact(
            corr_path,
            artifact_type="table",
            description="Spearman correlation between Experiment 02 and Experiment 03 component rankings.",
        )

        metrics = {
            "pairs_loaded": len(aligned_pairs),
            "attribution_components": len(attr_rows),
            "validation_components": len(validation_rows),
            "correlation_rows": len(corr_rows),
            "dry_run": False,
        }
        complete_run(ctx, metrics=metrics)
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
