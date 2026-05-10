#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP10_ROOT = PROJECT_ROOT / "results" / "10_path_mediation"
SUMMARY_FIG_DIR = PROJECT_ROOT / "results" / "summary" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Exp10 path-mediation figures for selected runs."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=[],
        help="Explicit Exp10 run directories. If omitted, latest completed non-dry run per model is used.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(SUMMARY_FIG_DIR),
        help="Directory for summary figures.",
    )
    return parser.parse_args()


def _model_slug(model_name: str) -> str:
    slug = model_name.lower().replace("/", "_")
    slug = re.sub(r"[^a-z0-9_-]+", "_", slug)
    return slug.strip("_")


def _load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _latest_completed_runs_by_model() -> list[Path]:
    by_model: dict[str, tuple[str, str, str, Path]] = {}
    for manifest_path in sorted(EXP10_ROOT.glob("*/*/run-*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = _load_manifest(run_dir)
        if manifest.get("status") != "completed":
            continue
        params = manifest.get("parameters", {})
        if params.get("dry_run"):
            continue
        model = str(params.get("model", "unknown"))
        key = (
            str(manifest.get("run_date_utc", "")),
            str(manifest.get("ended_at_utc", "")),
            str(manifest.get("run_id", "")),
            run_dir,
        )
        prev = by_model.get(model)
        if prev is None or key > prev:
            by_model[model] = key
    return [v[-1] for _, v in sorted(by_model.items())]


def _mean_and_std_by_layer(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (
        df.groupby("layer", as_index=False)[col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{col}_mean", "std": f"{col}_std"})
    )
    out[f"{col}_std"] = out[f"{col}_std"].fillna(0.0)
    return out


def _plot_per_run(run_dir: Path, summary_out_dir: Path) -> dict[str, str]:
    manifest = _load_manifest(run_dir)
    model = str(manifest.get("parameters", {}).get("model", "unknown"))
    run_id = str(manifest.get("run_id"))
    date = str(manifest.get("run_date_utc"))
    csv_path = run_dir / "tables" / "path_mediation.csv"
    df = pd.read_csv(csv_path)

    numeric_cols = [
        "layer",
        "stereotype_score_delta",
        "mean_margin_delta",
        "mean_proj_coeff",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["layer"]).copy()
    df["layer"] = df["layer"].astype(int)
    axes = sorted(df["axis"].dropna().unique())

    fig, axarr = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for axis_name in axes:
        sub = df[df["axis"] == axis_name].sort_values("layer")
        axarr[0].plot(sub["layer"], sub["mean_proj_coeff"], linewidth=1.4, alpha=0.85, label=axis_name)
        axarr[1].plot(sub["layer"], sub["stereotype_score_delta"], linewidth=1.4, alpha=0.85, label=axis_name)
        axarr[2].plot(sub["layer"], sub["mean_margin_delta"], linewidth=1.4, alpha=0.85, label=axis_name)

    proj_stats = _mean_and_std_by_layer(df, "mean_proj_coeff")
    score_stats = _mean_and_std_by_layer(df, "stereotype_score_delta")
    margin_stats = _mean_and_std_by_layer(df, "mean_margin_delta")

    axarr[0].plot(
        proj_stats["layer"],
        proj_stats["mean_proj_coeff_mean"],
        color="black",
        linewidth=2.3,
        label="axis mean",
    )
    axarr[1].plot(
        score_stats["layer"],
        score_stats["stereotype_score_delta_mean"],
        color="black",
        linewidth=2.3,
        label="axis mean",
    )
    axarr[2].plot(
        margin_stats["layer"],
        margin_stats["mean_margin_delta_mean"],
        color="black",
        linewidth=2.3,
        label="axis mean",
    )

    axarr[1].axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    axarr[2].axhline(0.0, color="gray", linewidth=1.0, linestyle="--")

    axarr[0].set_ylabel("Projection coeff")
    axarr[1].set_ylabel("Score delta")
    axarr[2].set_ylabel("Margin delta")
    axarr[2].set_xlabel("Layer (Exp01-aligned indexing)")
    axarr[0].set_title(f"Exp10 Path Mediation: {model} ({date} {run_id})")

    handles, labels = axarr[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.9, 1))

    model_slug = _model_slug(model)
    run_fig = run_dir / "figures" / "path_mediation_layer_curves.png"
    run_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(run_fig, dpi=160)
    plt.close(fig)

    summary_out_dir.mkdir(parents=True, exist_ok=True)
    summary_fig = summary_out_dir / f"exp10_{model_slug}_{date}_{run_id}.png"
    shutil.copy2(run_fig, summary_fig)

    return {
        "model": model,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_figure": str(summary_fig),
    }


def _plot_cross_model_summary(run_infos: list[dict[str, str]], out_dir: Path) -> Path:
    fig, axarr = plt.subplots(len(run_infos), 2, figsize=(13, 4 * len(run_infos)), sharex=False)
    if len(run_infos) == 1:
        axarr = [axarr]  # type: ignore[assignment]

    for i, info in enumerate(run_infos):
        run_dir = Path(info["run_dir"])
        df = pd.read_csv(run_dir / "tables" / "path_mediation.csv")
        for col in ["layer", "stereotype_score_delta", "mean_margin_delta"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["layer"]).copy()
        df["layer"] = df["layer"].astype(int)

        score_stats = _mean_and_std_by_layer(df, "stereotype_score_delta")
        margin_stats = _mean_and_std_by_layer(df, "mean_margin_delta")

        ax_s = axarr[i][0]
        ax_m = axarr[i][1]

        ax_s.plot(score_stats["layer"], score_stats["stereotype_score_delta_mean"], color="#1f77b4", linewidth=2.0)
        ax_s.fill_between(
            score_stats["layer"],
            score_stats["stereotype_score_delta_mean"] - score_stats["stereotype_score_delta_std"],
            score_stats["stereotype_score_delta_mean"] + score_stats["stereotype_score_delta_std"],
            color="#1f77b4",
            alpha=0.2,
            linewidth=0,
        )
        ax_s.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax_s.set_title(f"{info['model']} ({info['run_id']})")
        ax_s.set_ylabel("Score delta")
        ax_s.set_xlabel("Layer")

        ax_m.plot(margin_stats["layer"], margin_stats["mean_margin_delta_mean"], color="#d62728", linewidth=2.0)
        ax_m.fill_between(
            margin_stats["layer"],
            margin_stats["mean_margin_delta_mean"] - margin_stats["mean_margin_delta_std"],
            margin_stats["mean_margin_delta_mean"] + margin_stats["mean_margin_delta_std"],
            color="#d62728",
            alpha=0.2,
            linewidth=0,
        )
        ax_m.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax_m.set_title(f"{info['model']} ({info['run_id']})")
        ax_m.set_ylabel("Margin delta")
        ax_m.set_xlabel("Layer")

    fig.suptitle("Exp10 Cross-Model Layerwise Causal Effects (Axis Mean ± SD)", y=1.0, fontsize=14)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp10_cross_model_layerwise_summary.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.run_dirs:
        run_dirs = [Path(p).resolve() for p in args.run_dirs]
    else:
        run_dirs = _latest_completed_runs_by_model()

    if not run_dirs:
        raise RuntimeError("No completed Exp10 runs found.")

    out_dir = Path(args.out_dir).resolve()
    run_infos: list[dict[str, str]] = []
    for run_dir in run_dirs:
        run_infos.append(_plot_per_run(run_dir, out_dir))

    summary_path = _plot_cross_model_summary(run_infos, out_dir)
    index_path = out_dir / "exp10_figures_index.json"
    index_path.write_text(json.dumps({"runs": run_infos, "cross_model_summary": str(summary_path)}, indent=2), encoding="utf-8")

    print(f"Wrote {len(run_infos)} per-run figures and cross-model summary to {out_dir}")


if __name__ == "__main__":
    main()
