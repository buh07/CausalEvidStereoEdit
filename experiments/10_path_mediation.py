#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.interventions import make_direction_projection_at_position_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 10: layer-wise stereotype direction analysis at prediction_position."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument(
        "--ablation-target",
        choices=["residual", "attention", "mlp"],
        default="residual",
        help="Which pathway to project: full residual stream, attention output, or MLP output.",
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


def main() -> None:
    args = parse_args()
    ctx = start_run("10", parameters=vars(args), project_root=PROJECT_ROOT)
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

        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text())
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        directions_path = exp1_dir / "artifacts" / "directions_layerwise.npz"
        directions = load_directions_npz(directions_path) if directions_path.exists() else {}

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, {
            "exp1_run_dir": str(exp1_dir),
            "heldout_pairs": len(heldout),
            "directions_loaded": len(directions),
        })
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"heldout_pairs": len(heldout), "directions_loaded": len(directions), "dry_run": True})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        n_layers = bundle.num_layers
        axes = sorted({p.pair.axis for p in heldout})

        # Accumulators: (axis, layer_idx_0based) → list of (baseline_margin, patched_margin|None, proj_coeff|None)
        by_axis_layer: dict[tuple[str, int], list[tuple[float, float | None, float | None]]] = {}
        for axis in axes:
            for layer_idx in range(n_layers):
                by_axis_layer[(axis, layer_idx)] = []

        for pair in heldout:
            pred_pos = pair.prediction_position
            axis = pair.pair.axis

            encoded_s = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)

            # Baseline forward — capture hidden states for geometric measurement.
            with torch.no_grad():
                cap_base = forward_with_component_capture(
                    bundle.model, encoded_s,
                    output_hidden_states=True,
                    capture_attention=False,
                    capture_mlp=False,
                )

            if pred_pos >= cap_base.logits.shape[1]:
                continue

            baseline_margin = compute_score_from_logits(
                cap_base.logits, position=pred_pos,
                pos_token=pair.stereo_token,
                neg_token=pair.anti_token,
            )

            for layer_idx in range(n_layers):
                layer = layer_idx + 1  # 1-indexed layer key in directions dict
                direction_np = directions.get((axis, layer))

                # Geometric measurement: projection of h[pred_pos] onto stereotype direction.
                proj_coeff: float | None = None
                # Exp01 directions are keyed from hidden_states[1:] with layer numbering
                # starting at 1, so direction (axis, L) aligns with hidden_states[L].
                if direction_np is not None and layer < len(cap_base.hidden_states):
                    hs = cap_base.hidden_states[layer]  # (1, seq_len, d_model)
                    if pred_pos < hs.shape[1]:
                        d_cpu = torch.tensor(direction_np, dtype=torch.float32)
                        h_vec = hs[0, pred_pos, :].float().cpu()
                        d_norm = d_cpu.norm().clamp(min=1e-8)
                        proj_coeff = float(torch.dot(h_vec, d_cpu) / d_norm)

                # Causal measurement: project out direction at pred_pos at this layer.
                # Patching at prediction_position (not trait_token_position) avoids the
                # causal-masking issue: pred_pos is always causally upstream of the logit.
                patched_margin: float | None = None
                if direction_np is not None:
                    d_dev = torch.tensor(direction_np, device=bundle.device, dtype=torch.float32)
                    hook = make_direction_projection_at_position_hook(pred_pos, d_dev)
                    residual_patch_map = None
                    attention_patch_map = None
                    mlp_patch_map = None
                    capture_attention = False
                    capture_mlp = False
                    if args.ablation_target == "residual":
                        residual_patch_map = {layer_idx: hook}
                    elif args.ablation_target == "attention":
                        attention_patch_map = {layer_idx: hook}
                        capture_attention = True
                    elif args.ablation_target == "mlp":
                        mlp_patch_map = {layer_idx: hook}
                        capture_mlp = True
                    else:
                        raise ValueError(f"Unsupported ablation target: {args.ablation_target}")
                    with torch.no_grad():
                        cap_patched = forward_with_component_capture(
                            bundle.model, encoded_s,
                            output_hidden_states=False,
                            capture_attention=capture_attention,
                            capture_mlp=capture_mlp,
                            attention_patch_map=attention_patch_map,
                            mlp_patch_map=mlp_patch_map,
                            residual_patch_map=residual_patch_map,
                        )
                    if pred_pos < cap_patched.logits.shape[1]:
                        patched_margin = compute_score_from_logits(
                            cap_patched.logits, position=pred_pos,
                            pos_token=pair.stereo_token,
                            neg_token=pair.anti_token,
                        )

                by_axis_layer[(axis, layer_idx)].append((baseline_margin, patched_margin, proj_coeff))

        rows: list[dict[str, Any]] = []
        for (axis, layer_idx), data in sorted(by_axis_layer.items()):
            if not data:
                continue
            base_arr = np.array([b for b, _, _ in data], dtype=float)
            patched_vals = [(b, p) for b, p, _ in data if p is not None]
            proj_vals = [c for _, _, c in data if c is not None]

            row: dict[str, Any] = {
                "ablation_target": args.ablation_target,
                "axis": axis,
                "layer": layer_idx + 1,
                "stereotype_score_baseline": round(float(np.mean(base_arr > 0)), 8),
                "mean_margin_baseline": round(float(np.mean(base_arr)), 8),
                "n_pairs": len(data),
            }

            if patched_vals:
                base_for_patch = np.array([b for b, _ in patched_vals], dtype=float)
                patch_arr = np.array([p for _, p in patched_vals], dtype=float)
                row.update({
                    "stereotype_score_ablated": round(float(np.mean(patch_arr > 0)), 8),
                    "stereotype_score_delta": round(float(np.mean(patch_arr > 0) - np.mean(base_for_patch > 0)), 8),
                    "mean_margin_ablated": round(float(np.mean(patch_arr)), 8),
                    "mean_margin_delta": round(float(np.mean(patch_arr) - np.mean(base_for_patch)), 8),
                    "n_pairs_causal": len(patched_vals),
                })
            else:
                row.update({
                    "stereotype_score_ablated": "",
                    "stereotype_score_delta": "",
                    "mean_margin_ablated": "",
                    "mean_margin_delta": "",
                    "n_pairs_causal": 0,
                })

            if proj_vals:
                proj_arr = np.array(proj_vals, dtype=float)
                row.update({
                    "mean_proj_coeff": round(float(np.mean(proj_arr)), 8),
                    "std_proj_coeff": round(float(np.std(proj_arr)), 8),
                })
            else:
                row.update({"mean_proj_coeff": "", "std_proj_coeff": ""})

            rows.append(row)

        out_path = ctx.tables_dir / "path_mediation.csv"
        write_csv(out_path, rows, fieldnames=[
            "ablation_target",
            "axis", "layer",
            "stereotype_score_baseline", "stereotype_score_ablated", "stereotype_score_delta",
            "mean_margin_baseline", "mean_margin_ablated", "mean_margin_delta",
            "mean_proj_coeff", "std_proj_coeff",
            "n_pairs", "n_pairs_causal",
        ])
        ctx.register_artifact(out_path, artifact_type="table", description="Layer-wise path mediation results.")

        complete_run(ctx, metrics={"rows": len(rows), "n_layers": n_layers, "directions_loaded": len(directions), "dry_run": False})
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
