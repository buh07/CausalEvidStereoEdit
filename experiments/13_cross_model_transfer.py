#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.interventions import make_direction_projection_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 13: cross-model direction and component ranking transfer."
    )
    parser.add_argument("--source-model", default="google/gemma-2-2b")
    parser.add_argument("--target-model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-exp1-run-dir", default="")
    parser.add_argument("--target-exp1-run-dir", default="")
    parser.add_argument("--source-exp2-run-dir", default="")
    parser.add_argument("--target-exp2-run-dir", default="")
    return parser.parse_args()


def _latest_run_dir(
    experiment_slug: str,
    model_name: str,
    required_relpaths: list[str] | None = None,
) -> Path:
    root = PROJECT_ROOT / "results" / experiment_slug
    candidates = sorted(root.glob("*/*/manifest.json"))
    best: tuple[str, Path] | None = None
    for manifest_path in candidates:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        if payload.get("parameters", {}).get("model") != model_name:
            continue
        ended = payload.get("ended_at_utc") or ""
        run_dir = Path(payload["run_dir"])
        if required_relpaths:
            if any(not (run_dir / rel).exists() for rel in required_relpaths):
                continue
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        raise FileNotFoundError(f"No completed run for {experiment_slug} / {model_name}.")
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


def _project_direction(direction: np.ndarray, target_dim: int) -> np.ndarray:
    """Align a direction vector to a different d_model by truncation or zero-padding."""
    d = direction.shape[0]
    if d == target_dim:
        return direction
    if d > target_dim:
        return direction[:target_dim]
    padded = np.zeros(target_dim, dtype=direction.dtype)
    padded[:d] = direction
    return padded


def main() -> None:
    args = parse_args()
    ctx = start_run("13", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        src_exp1_dir = (
            Path(args.source_exp1_run_dir)
            if args.source_exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing", args.source_model,
                required_relpaths=["artifacts/directions_layerwise.npz"],
            )
        )
        tgt_exp1_dir = (
            Path(args.target_exp1_run_dir)
            if args.target_exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing", args.target_model,
                required_relpaths=["artifacts/aligned_pairs.jsonl", "artifacts/train_test_split.json",
                                   "artifacts/directions_layerwise.npz"],
            )
        )
        src_exp2_dir: Path | None = None
        tgt_exp2_dir: Path | None = None
        try:
            src_exp2_dir = (
                Path(args.source_exp2_run_dir)
                if args.source_exp2_run_dir
                else _latest_run_dir(
                    "02_component_dla", args.source_model,
                    required_relpaths=["tables/component_dla_scores.csv"],
                )
            )
            tgt_exp2_dir = (
                Path(args.target_exp2_run_dir)
                if args.target_exp2_run_dir
                else _latest_run_dir(
                    "02_component_dla", args.target_model,
                    required_relpaths=["tables/component_dla_scores.csv"],
                )
            )
        except FileNotFoundError:
            pass

        src_directions = load_directions_npz(src_exp1_dir / "artifacts" / "directions_layerwise.npz")
        tgt_directions = load_directions_npz(tgt_exp1_dir / "artifacts" / "directions_layerwise.npz")

        tgt_aligned_pairs = _load_aligned_pairs(tgt_exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((tgt_exp1_dir / "artifacts" / "train_test_split.json").read_text())
        test_indices = split_info.get("test_indices", [])
        heldout = [tgt_aligned_pairs[i] for i in test_indices if 0 <= i < len(tgt_aligned_pairs)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, {
            "source_model": args.source_model,
            "target_model": args.target_model,
            "src_exp1_run_dir": str(src_exp1_dir),
            "tgt_exp1_run_dir": str(tgt_exp1_dir),
            "src_exp2_run_dir": str(src_exp2_dir) if src_exp2_dir else None,
            "tgt_exp2_run_dir": str(tgt_exp2_dir) if tgt_exp2_dir else None,
            "heldout_pairs": len(heldout),
        })
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"heldout_pairs": len(heldout), "dry_run": True})
            return

        # --- Part 1: Direction transfer causal test ---
        bundle = load_model_bundle(model_name=args.target_model, device=args.device, torch_dtype=args.torch_dtype)
        tgt_n_layers = bundle.num_layers
        src_n_layers = len({l for (_, l) in src_directions})

        # Determine target d_model from any direction vector.
        tgt_d_model = next(iter(tgt_directions.values())).shape[0]

        axes = sorted(set(a for (a, _) in src_directions) & set(a for (a, _) in tgt_directions))
        transfer_rows: list[dict[str, Any]] = []

        for axis in axes:
            # Get all source layers for this axis, map to fractional depth.
            src_layers = sorted({l for (a, l) in src_directions if a == axis})
            if not src_layers:
                continue
            # Pick most informative source layer (highest norm).
            best_src_layer = max(
                src_layers,
                key=lambda l: float(np.linalg.norm(src_directions[(axis, l)]))
            )
            layer_fraction = best_src_layer / max(src_n_layers, 1)
            # Map to target layer (round to nearest integer, clamped).
            tgt_layer = max(1, min(tgt_n_layers, round(layer_fraction * tgt_n_layers)))
            idx = tgt_layer - 1

            # Retrieve and project source direction to target d_model.
            src_dir = src_directions[(axis, best_src_layer)]
            tgt_dir_transferred = _project_direction(src_dir, tgt_d_model)
            d_tensor = torch.tensor(tgt_dir_transferred, device=bundle.device, dtype=torch.float32)
            hook = make_direction_projection_hook(d_tensor)

            # Baseline score on target model.
            axis_pairs = [p for p in heldout if p.pair.axis == axis]
            base_margins: list[float] = []
            abl_margins: list[float] = []
            for pair in axis_pairs:
                pos = pair.prediction_position
                encoded = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, args.max_length)
                with torch.no_grad():
                    cap_base = forward_with_component_capture(bundle.model, encoded,
                                                              output_hidden_states=False,
                                                              capture_attention=False, capture_mlp=False)
                    cap_abl = forward_with_component_capture(bundle.model, encoded,
                                                             output_hidden_states=False,
                                                             capture_attention=False, capture_mlp=False,
                                                             residual_patch_map={idx: hook})
                if pos >= cap_base.logits.shape[1]:
                    continue
                base_margins.append(compute_score_from_logits(cap_base.logits, pos,
                                                               pair.stereo_token, pair.anti_token))
                abl_margins.append(compute_score_from_logits(cap_abl.logits, pos,
                                                              pair.stereo_token, pair.anti_token))

            if base_margins:
                base_arr = np.array(base_margins, dtype=float)
                abl_arr = np.array(abl_margins, dtype=float)
                transfer_rows.append({
                    "source_model": args.source_model,
                    "target_model": args.target_model,
                    "axis": axis,
                    "source_layer": best_src_layer,
                    "target_layer": tgt_layer,
                    "layer_fraction": round(layer_fraction, 4),
                    "stereotype_score_baseline": round(float(np.mean(base_arr > 0)), 8),
                    "stereotype_score_transferred": round(float(np.mean(abl_arr > 0)), 8),
                    "stereotype_score_delta": round(float(np.mean(abl_arr > 0) - np.mean(base_arr > 0)), 8),
                    "mean_margin_baseline": round(float(np.mean(base_arr)), 8),
                    "mean_margin_transferred": round(float(np.mean(abl_arr)), 8),
                    "mean_margin_delta": round(float(np.mean(abl_arr) - np.mean(base_arr)), 8),
                    "n_pairs": len(base_margins),
                })

        transfer_path = ctx.tables_dir / "direction_transfer.csv"
        write_csv(transfer_path, transfer_rows, fieldnames=[
            "source_model", "target_model", "axis",
            "source_layer", "target_layer", "layer_fraction",
            "stereotype_score_baseline", "stereotype_score_transferred", "stereotype_score_delta",
            "mean_margin_baseline", "mean_margin_transferred", "mean_margin_delta",
            "n_pairs",
        ])
        ctx.register_artifact(transfer_path, artifact_type="table", description="Cross-model direction transfer causal test.")

        # --- Part 2: Component ranking transfer (Spearman correlation) ---
        ranking_rows: list[dict[str, Any]] = []
        if src_exp2_dir is not None and tgt_exp2_dir is not None:
            src_dla = pd.read_csv(src_exp2_dir / "tables" / "component_dla_scores.csv")
            tgt_dla = pd.read_csv(tgt_exp2_dir / "tables" / "component_dla_scores.csv")

            # Normalize layers to fraction of total depth for cross-model comparison.
            src_max_layer = int(src_dla["layer"].max()) if not src_dla.empty else 1
            tgt_max_layer = int(tgt_dla["layer"].max()) if not tgt_dla.empty else 1
            src_dla = src_dla.copy()
            tgt_dla = tgt_dla.copy()
            src_dla["layer_frac"] = src_dla["layer"] / src_max_layer
            tgt_dla["layer_frac"] = tgt_dla["layer"] / tgt_max_layer
            # Round to bins for alignment.
            bins = np.linspace(0, 1, 11)
            src_dla["layer_bin"] = np.digitize(src_dla["layer_frac"], bins, right=True)
            tgt_dla["layer_bin"] = np.digitize(tgt_dla["layer_frac"], bins, right=True)

            for axis in sorted(set(src_dla["axis"]) & set(tgt_dla["axis"])):
                src_ax = src_dla[src_dla["axis"] == axis][["component_type", "layer_bin", "mean_abs_dla_score"]].copy()
                tgt_ax = tgt_dla[tgt_dla["axis"] == axis][["component_type", "layer_bin", "mean_abs_dla_score"]].copy()
                src_ax["key"] = src_ax["component_type"].astype(str) + "_bin" + src_ax["layer_bin"].astype(str)
                tgt_ax["key"] = tgt_ax["component_type"].astype(str) + "_bin" + tgt_ax["layer_bin"].astype(str)
                merged = src_ax.merge(tgt_ax, on="key", suffixes=("_src", "_tgt"))
                if len(merged) >= 3:
                    rho = spearmanr(
                        merged["mean_abs_dla_score_src"].astype(float),
                        merged["mean_abs_dla_score_tgt"].astype(float),
                    ).correlation
                else:
                    rho = np.nan
                ranking_rows.append({
                    "source_model": args.source_model,
                    "target_model": args.target_model,
                    "axis": axis,
                    "spearman_rho": round(float(rho), 8) if not np.isnan(rho) else "",
                    "n_components": len(merged),
                })

        ranking_path = ctx.tables_dir / "ranking_transfer.csv"
        write_csv(ranking_path, ranking_rows, fieldnames=[
            "source_model", "target_model", "axis", "spearman_rho", "n_components",
        ])
        ctx.register_artifact(ranking_path, artifact_type="table", description="Cross-model component ranking transfer.")

        complete_run(ctx, metrics={
            "direction_transfer_rows": len(transfer_rows),
            "ranking_transfer_rows": len(ranking_rows),
            "axes": len(axes),
            "dry_run": False,
        })
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
