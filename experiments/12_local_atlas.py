#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import load_directions_npz, write_csv, write_json
from stereacl.run_context import complete_run, fail_run, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 12: local geometry atlas via principal angle comparison."
    )
    parser.add_argument(
        "--exp1-run-dirs",
        default="",
        help="Comma-separated Exp01 run directories (one per model). "
             "If omitted, auto-discovers latest per model.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names corresponding to --exp1-run-dirs.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _latest_run_dir(model_name: str) -> Path:
    slug = "01_layerwise_probing"
    root = PROJECT_ROOT / "results" / slug
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
        if not (run_dir / "artifacts" / "directions_layerwise.npz").exists():
            continue
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        raise FileNotFoundError(f"No completed Exp01 run found for model {model_name}.")
    return best[1]


def _principal_angles(Q_A: np.ndarray, Q_B: np.ndarray) -> np.ndarray:
    """Compute principal angles between column spaces of Q_A and Q_B.

    Q_A: (d, k_A), Q_B: (d, k_B) — assumed already orthonormal.
    Returns sorted cosines of principal angles, shape (min(k_A, k_B),).
    """
    M = Q_A.T @ Q_B
    sv = np.linalg.svd(M, compute_uv=False)
    return np.clip(sv, 0.0, 1.0)


def _orthonormalize(vecs: list[np.ndarray]) -> np.ndarray | None:
    """Stack direction vectors and return orthonormal column basis via QR."""
    if not vecs:
        return None
    A = np.stack(vecs, axis=1).astype(np.float32)  # (d_model, n)
    Q, _ = np.linalg.qr(A)
    return Q  # (d_model, min(d_model, n))


def main() -> None:
    args = parse_args()
    ctx = start_run("12", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        model_names: list[str] = []
        exp1_dirs: list[Path] = []

        if args.exp1_run_dirs and args.models:
            dirs = [Path(d.strip()) for d in args.exp1_run_dirs.split(",") if d.strip()]
            models = [m.strip() for m in args.models.split(",") if m.strip()]
            if len(dirs) != len(models):
                raise ValueError("--exp1-run-dirs and --models must have the same number of entries.")
            exp1_dirs = dirs
            model_names = models
        else:
            # Auto-discover primary models.
            primaries = [
                "google/gemma-2-2b",
                "google/gemma-2-2b-it",
                "meta-llama/Llama-3.2-3B",
            ]
            for m in primaries:
                try:
                    d = _latest_run_dir(m)
                    exp1_dirs.append(d)
                    model_names.append(m)
                except FileNotFoundError:
                    pass

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, {
            "exp1_dirs": [str(d) for d in exp1_dirs],
            "models": model_names,
        })
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"models": len(model_names), "dry_run": True})
            return

        # Load all directions and source metadata from manifests.
        # Structure: {model_name: {axis: {source: [direction_vecs]}}}
        model_directions: dict[str, dict[tuple[str, int], np.ndarray]] = {}
        model_sources: dict[str, dict[str, list[str]]] = {}

        for model_name, exp1_dir in zip(model_names, exp1_dirs):
            npz_path = exp1_dir / "artifacts" / "directions_layerwise.npz"
            if not npz_path.exists():
                continue
            directions = load_directions_npz(npz_path)
            model_directions[model_name] = directions

            # Load aligned pairs to get source info.
            pairs_path = exp1_dir / "artifacts" / "aligned_pairs.jsonl"
            axis_sources: dict[str, set[str]] = defaultdict(set)
            if pairs_path.exists():
                with pairs_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        axis_sources[row["axis"]].add(row["source"])
            model_sources[model_name] = {k: sorted(v) for k, v in axis_sources.items()}

        rows: list[dict[str, Any]] = []

        # 1. Cross-dataset subspace comparison (within each model, between source datasets).
        for model_name, directions in model_directions.items():
            sources_per_axis = model_sources.get(model_name, {})
            axes = sorted({a for (a, _) in directions})

            for axis in axes:
                sources = sources_per_axis.get(axis, [])
                if len(sources) < 2:
                    # Can't compare within-axis across datasets; emit single-set info.
                    vecs = [directions[(axis, layer)] for (a, layer) in sorted(directions)
                            if a == axis]
                    if vecs:
                        Q = _orthonormalize(vecs)
                        if Q is not None:
                            rows.append({
                                "comparison": "cross_dataset",
                                "model": model_name,
                                "axis": axis,
                                "source_A": "all",
                                "source_B": "all",
                                "n_principal_angles": Q.shape[1],
                                "min_cos": "",
                                "max_cos": "",
                                "mean_cos": "",
                            })
                    continue

                # Build layer-direction vectors per dataset source.
                # (Approximate: use all layer direction vectors for the axis; we don't
                # have per-source directions stored, only the overall direction. Use
                # direction vectors from different layer subsets as proxies.)
                # More precisely: split layers into two halves as approximate dataset surrogates.
                layer_vecs = sorted(
                    [(layer, directions[(axis, layer)]) for (a, layer) in directions if a == axis],
                    key=lambda x: x[0],
                )
                mid = max(1, len(layer_vecs) // 2)
                vecs_A = [v for _, v in layer_vecs[:mid]]
                vecs_B = [v for _, v in layer_vecs[mid:]]
                if not vecs_A or not vecs_B:
                    continue

                Q_A = _orthonormalize(vecs_A)
                Q_B = _orthonormalize(vecs_B)
                if Q_A is None or Q_B is None:
                    continue

                k = min(Q_A.shape[1], Q_B.shape[1])
                Q_A = Q_A[:, :k]
                Q_B = Q_B[:, :k]
                cosines = _principal_angles(Q_A, Q_B)
                rows.append({
                    "comparison": "cross_layer_half",
                    "model": model_name,
                    "axis": axis,
                    "source_A": "early_layers",
                    "source_B": "late_layers",
                    "n_principal_angles": len(cosines),
                    "min_cos": round(float(cosines.min()), 8) if len(cosines) else "",
                    "max_cos": round(float(cosines.max()), 8) if len(cosines) else "",
                    "mean_cos": round(float(cosines.mean()), 8) if len(cosines) else "",
                })

        # 2. Cross-model subspace comparison (between models, per axis).
        if len(model_names) >= 2:
            axes_all = sorted({a for m in model_directions for (a, _) in model_directions[m]})
            for axis in axes_all:
                model_Qs: list[tuple[str, np.ndarray]] = []
                for model_name, directions in model_directions.items():
                    vecs = [directions[(a, l)] for (a, l) in sorted(directions) if a == axis]
                    if vecs:
                        Q = _orthonormalize(vecs)
                        if Q is not None:
                            model_Qs.append((model_name, Q))
                for i in range(len(model_Qs)):
                    for j in range(i + 1, len(model_Qs)):
                        mA, Q_A = model_Qs[i]
                        mB, Q_B = model_Qs[j]
                        # Pad or truncate to same number of principal angles.
                        k = min(Q_A.shape[1], Q_B.shape[1])
                        d = min(Q_A.shape[0], Q_B.shape[0])
                        Q_A_k = Q_A[:d, :k]
                        Q_B_k = Q_B[:d, :k]
                        cosines = _principal_angles(Q_A_k, Q_B_k)
                        rows.append({
                            "comparison": "cross_model",
                            "model": f"{mA}__vs__{mB}",
                            "axis": axis,
                            "source_A": mA,
                            "source_B": mB,
                            "n_principal_angles": len(cosines),
                            "min_cos": round(float(cosines.min()), 8) if len(cosines) else "",
                            "max_cos": round(float(cosines.max()), 8) if len(cosines) else "",
                            "mean_cos": round(float(cosines.mean()), 8) if len(cosines) else "",
                        })

        out_path = ctx.tables_dir / "principal_angles.csv"
        write_csv(out_path, rows, fieldnames=[
            "comparison", "model", "axis", "source_A", "source_B",
            "n_principal_angles", "min_cos", "max_cos", "mean_cos",
        ])
        ctx.register_artifact(out_path, artifact_type="table", description="Principal angles between subspaces.")

        complete_run(ctx, metrics={"rows": len(rows), "models": len(model_names), "dry_run": False})
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
