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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.interventions import make_direction_injection_hook
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 08: signed dose-response direction injection."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--alphas",
        default="-2,-1,-0.5,-0.25,0,0.25,0.5,1,2",
        help="Comma-separated alpha values for signed injection sweep.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument("--exp2-run-dir", default="")
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


def _top_layer_per_axis(exp2_dir: Path) -> dict[str, int]:
    """Return highest-DLA layer for each axis from Exp02 component_dla_scores.csv."""
    path = exp2_dir / "tables" / "component_dla_scores.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    top: dict[str, int] = {}
    for axis, group in df.groupby("axis"):
        best_row = group.loc[group["mean_abs_dla_score"].idxmax()]
        top[str(axis)] = int(best_row["layer"])
    return top


def main() -> None:
    args = parse_args()
    ctx = start_run("08", parameters=vars(args), project_root=PROJECT_ROOT)
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
        split_info = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text())
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        directions = load_directions_npz(exp1_dir / "artifacts" / "directions_layerwise.npz")
        top_layer = _top_layer_per_axis(exp2_dir)
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        axes = sorted({a for (a, _) in directions})

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, {
            "exp1_run_dir": str(exp1_dir),
            "exp2_run_dir": str(exp2_dir),
            "heldout_pairs": len(heldout),
            "axes": axes,
            "alphas": alphas,
        })
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"heldout_pairs": len(heldout), "axes": len(axes), "alphas": alphas, "dry_run": True})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)

        rows: list[dict[str, Any]] = []

        for axis in axes:
            layer = top_layer.get(axis)
            if layer is None:
                # Fall back to layer with largest direction norm.
                axis_dirs = [(l, directions[(axis, l)]) for (a, l) in directions if a == axis]
                if not axis_dirs:
                    continue
                layer = max(axis_dirs, key=lambda x: float(np.linalg.norm(x[1])))[0]
            if (axis, layer) not in directions:
                continue

            direction_np = directions[(axis, layer)]
            idx = layer - 1  # block index for residual patch
            axis_pairs = [p for p in heldout if p.pair.axis == axis]

            for alpha in alphas:
                d = torch.tensor(direction_np, device=bundle.device, dtype=torch.float32)
                hook = make_direction_injection_hook(d, alpha=alpha)
                residual_patch_map = {idx: hook}

                margins: list[float] = []
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
                        "axis": axis,
                        "layer": layer,
                        "alpha": alpha,
                        "stereotype_score": round(float(np.mean(arr > 0)), 8),
                        "mean_margin": round(float(np.mean(arr)), 8),
                        "n_pairs": len(margins),
                    })

        out_path = ctx.tables_dir / "dose_response.csv"
        write_csv(out_path, rows, fieldnames=["axis", "layer", "alpha", "stereotype_score", "mean_margin", "n_pairs"])
        ctx.register_artifact(out_path, artifact_type="table", description="Dose-response direction injection sweep.")

        complete_run(ctx, metrics={"rows": len(rows), "axes": len(axes), "alphas": alphas, "dry_run": False})
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
