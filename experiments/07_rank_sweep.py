#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.interventions import make_direction_projection_hook, make_rank_k_projection_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 07: rank-sweep causal curves for stereotype subspace ablation."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--ranks", default="1,2,4,8,16,32", help="Comma-separated k values to sweep.")
    parser.add_argument(
        "--basis-mode",
        choices=["svd", "raw", "both"],
        default="both",
        help="Subspace basis for rank sweep: SVD-orthonormalized, raw per-layer directions, or both.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exp1-run-dir", default="")
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


def _build_axis_svd(
    directions: dict[tuple[str, int], np.ndarray],
    axis: str,
    n_layers: int,
) -> np.ndarray | None:
    """Stack layer direction vectors for an axis and return right singular vectors via SVD."""
    vecs = [
        directions[(a, layer)]
        for (a, layer) in sorted(directions.keys(), key=lambda k: k[1])
        if a == axis and layer <= n_layers
    ]
    if not vecs:
        return None
    D = np.stack(vecs, axis=0).astype(np.float32)  # (n_layers, d_model)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt  # (min(n_layers, d_model), d_model)


def _build_axis_layer_dirs(
    directions: dict[tuple[str, int], np.ndarray],
    axis: str,
    n_layers: int,
) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    for (a, layer), d in sorted(directions.items(), key=lambda kv: kv[0][1]):
        if a != axis or layer > n_layers:
            continue
        out.append((layer, np.asarray(d, dtype=np.float32)))
    return out


def main() -> None:
    args = parse_args()
    ctx = start_run("07", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = (
            Path(args.exp1_run_dir)
            if args.exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=["artifacts/aligned_pairs.jsonl", "artifacts/train_test_split.json",
                                   "artifacts/directions_layerwise.npz"],
                model_name=args.model,
            )
        )

        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text())
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        directions = load_directions_npz(exp1_dir / "artifacts" / "directions_layerwise.npz")
        axes = sorted({a for (a, _) in directions})
        ks = [int(x) for x in args.ranks.split(",") if x.strip()]

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, {
            "exp1_run_dir": str(exp1_dir),
            "heldout_pairs": len(heldout),
            "axes": axes,
            "ks": ks,
        })
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"heldout_pairs": len(heldout), "axes": len(axes), "ks": ks, "dry_run": True})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        n_layers = bundle.num_layers

        # Precompute per-axis SVD bases.
        axis_vt: dict[str, np.ndarray] = {}
        axis_layer_dirs: dict[str, list[tuple[int, np.ndarray]]] = {}
        for axis in axes:
            vt = _build_axis_svd(directions, axis, n_layers)
            if vt is not None:
                axis_vt[axis] = vt
            axis_layer_dirs[axis] = _build_axis_layer_dirs(directions, axis, n_layers)

        rows: list[dict[str, Any]] = []

        basis_modes = ["svd", "raw"] if args.basis_mode == "both" else [args.basis_mode]
        for basis_mode in basis_modes:
            for k in ks:
                for axis in axes:
                    residual_patch_map: dict[int, Any] = {}
                    rank = 0
                    if basis_mode == "svd":
                        vt = axis_vt.get(axis)
                        if vt is None:
                            continue
                        rank = min(k, vt.shape[0])
                        top_k_vecs = vt[:rank, :]  # (rank, d_model)
                        dir_matrix = torch.tensor(top_k_vecs.T, dtype=torch.float32, device=bundle.device)  # (d_model, rank)
                        hook = make_rank_k_projection_hook(dir_matrix)
                        layer_indices = sorted({
                            layer - 1
                            for (a, layer) in directions
                            if a == axis and layer <= n_layers
                        })
                        residual_patch_map = {idx: hook for idx in layer_indices}
                    else:
                        layer_dirs = axis_layer_dirs.get(axis, [])
                        if not layer_dirs:
                            continue
                        chosen = layer_dirs[: min(k, len(layer_dirs))]
                        rank = len(chosen)
                        for layer, d_np in chosen:
                            d_t = torch.tensor(d_np, dtype=torch.float32, device=bundle.device)
                            residual_patch_map[layer - 1] = make_direction_projection_hook(d_t)

                    if not residual_patch_map:
                        continue

                    margins: list[float] = []
                    axis_pairs = [p for p in heldout if p.pair.axis == axis]
                    for pair in axis_pairs:
                        pos = pair.prediction_position
                        encoded = encode_text(
                            tokenizer=bundle.tokenizer,
                            text=pair.pair.stereotype_text,
                            device=bundle.device,
                            max_length=args.max_length,
                        )
                        with torch.no_grad():
                            cap = forward_with_component_capture(
                                model=bundle.model,
                                encoded_inputs=encoded,
                                output_hidden_states=False,
                                capture_attention=False,
                                capture_mlp=False,
                                residual_patch_map=residual_patch_map,
                            )
                        if pos >= cap.logits.shape[1]:
                            continue
                        margin = compute_score_from_logits(cap.logits, position=pos,
                                                           pos_token=pair.stereo_token,
                                                           neg_token=pair.anti_token)
                        margins.append(margin)

                    if margins:
                        arr = np.array(margins, dtype=float)
                        rows.append({
                            "basis_mode": basis_mode,
                            "axis": axis,
                            "k": k,
                            "rank_used": rank,
                            "stereotype_score": round(float(np.mean(arr > 0)), 8),
                            "mean_margin": round(float(np.mean(arr)), 8),
                            "n_pairs": len(margins),
                        })

        out_path = ctx.tables_dir / "rank_sweep.csv"
        write_csv(
            out_path,
            rows,
            fieldnames=["basis_mode", "axis", "k", "rank_used", "stereotype_score", "mean_margin", "n_pairs"],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Rank-sweep causal curves.")

        complete_run(
            ctx,
            metrics={"rows": len(rows), "ks_swept": ks, "axes": len(axes), "basis_mode": args.basis_mode, "dry_run": False},
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
