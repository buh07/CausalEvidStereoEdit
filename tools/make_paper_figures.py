#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/jumbo/lisp/f004ndc/StereACL")

CORE_MODELS = [
    ("google/gemma-2-2b", "Gemma-2-2B"),
    ("google/gemma-2-2b-it", "Gemma-2-2B-IT"),
    ("meta-llama/Llama-3.2-3B", "Llama-3.2-3B"),
]

ALL_MODELS_ORDER = [
    "Gemma-2-2B",
    "Gemma-2-2B-IT",
    "Llama-3.2-3B",
    "Qwen2.5-3B",
    "Qwen2.5-3B-Instruct",
    "Mistral-7B",
    "OLMo-2-7B",
]

MODEL_LABEL_FALLBACK = {
    "google/gemma-2-2b": "Gemma-2-2B",
    "google/gemma-2-2b-it": "Gemma-2-2B-IT",
    "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
    "/jumbo/lisp/f004ndc/models/mistral-7b-v0.1": "Mistral-7B",
    "/jumbo/lisp/f004ndc/models/olmo-2-7b": "OLMo-2-7B",
}

# Colorblind-safe palette.
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_GRAY = "#666666"
C_LIGHT_GRAY = "#C7C7C7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper figures from computed artifacts.")
    p.add_argument("--artifact-manifest", default="", help="Optional artifact manifest JSON.")
    p.add_argument("--run-map", default="", help="Optional run-map JSON (used if manifest not provided).")
    p.add_argument("--outdir", default=str(ROOT / "paper" / "build"))
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def latest_run(slug: str, model: str) -> Path:
    best: tuple[str, Path] | None = None
    for mf in (ROOT / "results" / slug).glob("*/*/manifest.json"):
        d = json.loads(mf.read_text(encoding="utf-8"))
        if d.get("status") != "completed":
            continue
        if d.get("parameters", {}).get("model") != model:
            continue
        t = d.get("ended_at_utc", "")
        rd = Path(d["run_dir"])
        if best is None or t > best[0]:
            best = (t, rd)
    if best is None:
        raise RuntimeError(f"No completed run for {slug} / {model}")
    return best[1]


def _load_run_map(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_manifest:
        mp = Path(args.artifact_manifest)
        payload = json.loads(mp.read_text(encoding="utf-8"))
        run_map = payload.get("run_map")
        if isinstance(run_map, dict):
            return run_map
    if args.run_map:
        rp = Path(args.run_map)
        return json.loads(rp.read_text(encoding="utf-8"))
    return {}


def _model_label(model_name: str, payload: dict[str, Any] | None = None) -> str:
    if payload and payload.get("label"):
        return str(payload["label"])
    return MODEL_LABEL_FALLBACK.get(model_name, model_name)


def _exp_run_dir_from_map(run_map: dict[str, Any], model: str, key: str) -> Path | None:
    models = run_map.get("models", {})
    if not isinstance(models, dict):
        return None
    payload = models.get(model, {})
    if not isinstance(payload, dict):
        return None
    val = payload.get(key, "")
    if not val:
        return None
    p = Path(str(val))
    return p if p.exists() else None


def _is_explicit_run_map(run_map: dict[str, Any]) -> bool:
    models = run_map.get("models", {})
    return isinstance(models, dict) and bool(models)


def _read_exp16_bundle(exp16_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(exp16_dir / "tables" / "asymmetry_2x2_matrix.csv")
    contrast = pd.read_csv(exp16_dir / "tables" / "asymmetry_contrast.csv")
    return matrix, contrast


def _collect_asymmetry_rows(run_map: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    # Prefer explicit run-map models when present; otherwise fall back to known model list and latest runs.
    model_items: list[tuple[str, dict[str, Any]]] = []
    map_models = run_map.get("models", {})
    using_explicit_map = isinstance(map_models, dict) and bool(map_models)
    if using_explicit_map:
        for model_name, payload in map_models.items():
            if isinstance(payload, dict):
                model_items.append((model_name, payload))
    else:
        for model_name, _ in CORE_MODELS:
            model_items.append((model_name, {}))

    for model_name, payload in model_items:
        exp16_dir = _exp_run_dir_from_map(run_map, model_name, "exp16_canonical_run_dir")
        if exp16_dir is None:
            seed_runs = payload.get("exp16_seed_runs", {}) if isinstance(payload, dict) else {}
            if isinstance(seed_runs, dict) and seed_runs:
                val = seed_runs.get("11") or next(iter(seed_runs.values()))
                if val:
                    exp16_dir = Path(str(val))
        if exp16_dir is None and not using_explicit_map:
            try:
                exp16_dir = latest_run("16_asymmetry_matrix", model_name)
            except Exception:
                continue
        if exp16_dir is None:
            # In explicit run-map mode, missing canonical run dirs are treated as missing artifacts.
            continue

        try:
            matrix, contrast = _read_exp16_bundle(exp16_dir)
        except Exception:
            continue

        primary = contrast[contrast["contrast"] == "primary_inject_anti_minus_remove_stereo"]
        if primary.empty:
            continue
        p = primary.iloc[0]
        label = _model_label(model_name, payload if isinstance(payload, dict) else None)

        rem = matrix[matrix["condition"] == "remove_on_stereo"]
        inj = matrix[matrix["condition"] == "inject_on_anti"]

        rec: dict[str, Any] = {
            "model": model_name,
            "label": label,
            "exp16_run_dir": str(exp16_dir),
            "score": float(p.get("mean_score_contrast", np.nan)),
            "score_lo": float(p.get("mean_score_contrast_ci_low", np.nan)),
            "score_hi": float(p.get("mean_score_contrast_ci_high", np.nan)),
            "score_q": float(p.get("q_score_sign", np.nan)),
            "margin": float(p.get("mean_margin_contrast", np.nan)),
            "margin_lo": float(p.get("mean_margin_contrast_ci_low", np.nan)),
            "margin_hi": float(p.get("mean_margin_contrast_ci_high", np.nan)),
            "margin_q": float(p.get("q_margin_wilcoxon", np.nan)),
        }

        if not rem.empty:
            rr = rem.iloc[0]
            rec.update(
                {
                    "remove": float(rr.get("stereotype_score_delta", np.nan)),
                    "remove_lo": float(rr.get("stereotype_score_delta_ci_low", np.nan)),
                    "remove_hi": float(rr.get("stereotype_score_delta_ci_high", np.nan)),
                    "remove_margin": float(rr.get("mean_margin_delta", np.nan)),
                    "remove_margin_lo": float(rr.get("mean_margin_delta_ci_low", np.nan)),
                    "remove_margin_hi": float(rr.get("mean_margin_delta_ci_high", np.nan)),
                    "remove_margin_q": float(rr.get("q_margin_wilcoxon", np.nan)),
                }
            )

        if not inj.empty:
            ii = inj.iloc[0]
            rec.update(
                {
                    "inject": float(ii.get("stereotype_score_delta", np.nan)),
                    "inject_lo": float(ii.get("stereotype_score_delta_ci_low", np.nan)),
                    "inject_hi": float(ii.get("stereotype_score_delta_ci_high", np.nan)),
                }
            )

        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["label"] = pd.Categorical(df["label"], categories=ALL_MODELS_ORDER, ordered=True)
    df = df.sort_values("label").reset_index(drop=True)
    return df


def fig_headline_asymmetry(dfm: pd.DataFrame, outdir: Path) -> None:
    if dfm.empty:
        return

    y = np.arange(len(dfm))
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.8), sharey=True)

    score_sig = int(np.sum(pd.to_numeric(dfm["score_q"], errors="coerce") < 0.05))
    margin_sig = int(np.sum(pd.to_numeric(dfm["margin_q"], errors="coerce") < 0.05))

    for ax, metric, lo_col, hi_col, q_col, xlabel, title in [
        (
            axes[0],
            "score",
            "score_lo",
            "score_hi",
            "score_q",
            "Inject − remove score contrast",
            f"Behavioral endpoint ({score_sig}/{len(dfm)} significant)",
        ),
        (
            axes[1],
            "margin",
            "margin_lo",
            "margin_hi",
            "margin_q",
            "Inject − remove margin contrast",
            f"Mechanistic endpoint ({margin_sig}/{len(dfm)} significant)",
        ),
    ]:
        ax.axvline(0.0, color=C_GRAY, linestyle="--", linewidth=1.1, zorder=0)
        q = pd.to_numeric(dfm[q_col], errors="coerce")
        sig = q < 0.05
        colors = np.where(sig, C_BLUE, C_LIGHT_GRAY)

        for i in range(len(dfm)):
            x = float(dfm.iloc[i][metric])
            lo = float(dfm.iloc[i][lo_col])
            hi = float(dfm.iloc[i][hi_col])
            if not np.isfinite([x, lo, hi]).all():
                continue
            ax.errorbar(
                x,
                i,
                xerr=[[x - lo], [hi - x]],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                elinewidth=2.0,
                capsize=3.0,
                markersize=6.0,
                zorder=3,
            )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(dfm["label"].astype(str))
        ax.invert_yaxis()
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")

    axes[0].set_ylabel("Model")
    axes[1].set_ylabel("Model")

    axes[0].plot([], [], "o", color=C_BLUE, label="q < 0.05")
    axes[0].plot([], [], "o", color=C_LIGHT_GRAY, label="q ≥ 0.05")
    axes[0].legend(loc="lower right", frameon=True)

    fig.tight_layout(h_pad=1.2)
    fig.savefig(outdir / "fig_asymmetry_contrast.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_core_asymmetry_detail(dfm: pd.DataFrame, outdir: Path) -> None:
    if dfm.empty:
        return
    core_labels = [label for _, label in CORE_MODELS]
    dfa = dfm[dfm["label"].astype(str).isin(core_labels)].copy()
    if dfa.empty:
        return
    order = ["Llama-3.2-3B", "Gemma-2-2B-IT", "Gemma-2-2B"]
    dfa["label"] = pd.Categorical(dfa["label"].astype(str), categories=order, ordered=True)
    dfa = dfa.sort_values("label")
    y = np.arange(len(dfa))

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.axvline(0.0, color=C_GRAY, linewidth=1.1, linestyle="--")

    ax.errorbar(
        dfa["remove"],
        y + 0.18,
        xerr=[dfa["remove"] - dfa["remove_lo"], dfa["remove_hi"] - dfa["remove"]],
        fmt="o",
        color=C_ORANGE,
        label="Ablation delta (main intervention test)",
        capsize=3.5,
        elinewidth=2.2,
        markersize=6.5,
    )
    ax.errorbar(
        dfa["inject"],
        y - 0.18,
        xerr=[dfa["inject"] - dfa["inject_lo"], dfa["inject_hi"] - dfa["inject"]],
        fmt="o",
        color=C_GREEN,
        label="Injection delta (main intervention test)",
        capsize=3.5,
        elinewidth=2.2,
        markersize=6.5,
    )
    ax.errorbar(
        dfa["score"],
        y,
        xerr=[dfa["score"] - dfa["score_lo"], dfa["score_hi"] - dfa["score"]],
        fmt="s",
        color=C_BLUE,
        label="Direct contrast (matched operators)",
        capsize=3.5,
        elinewidth=2.2,
        markersize=6.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(dfa["label"].astype(str))
    ax.set_xlabel("Score delta")
    ax.set_title("Core-model asymmetry detail")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "fig_asymmetry_core_detail.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_sign_reliability(run_map: dict[str, Any], outdir: Path) -> None:
    target_model = "google/gemma-2-2b-it"
    rd14 = _exp_run_dir_from_map(run_map, target_model, "exp14_run_dir")
    if rd14 is None and not _is_explicit_run_map(run_map):
        rd14 = latest_run("14_sign_reliability_audit", target_model)
    if rd14 is None:
        return

    df = pd.read_csv(rd14 / "tables" / "sign_reliability_by_axis.csv")
    df = df[pd.to_numeric(df["n_dla_sign_agreement"], errors="coerce").fillna(0) > 0].copy()
    if df.empty:
        return
    df = df.sort_values("dla_sign_agreement_rate")

    y = np.arange(len(df))
    vals = pd.to_numeric(df["dla_sign_agreement_rate"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(df["dla_sign_agreement_ci_low"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(df["dla_sign_agreement_ci_high"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.axvline(0.5, color=C_GRAY, linestyle="--", linewidth=1.1, label="Chance (50%)")
    ax.errorbar(
        vals,
        y,
        xerr=[vals - lo, hi - vals],
        fmt="o",
        color=C_PURPLE,
        ecolor=C_PURPLE,
        capsize=3.5,
        elinewidth=2.2,
        markersize=6.5,
    )

    labels = [f"{a} (n={int(n)})" for a, n in zip(df["axis"], df["n_dla_sign_agreement"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Sign agreement vs causal direction")
    ax.set_title("Gemma-2-2B-IT per-axis sign reliability")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "fig_gemmait_sign_forest.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_margin_shift_delta(dfm: pd.DataFrame, outdir: Path) -> None:
    if dfm.empty:
        return
    sub = dfm[["label", "remove_margin", "remove_margin_lo", "remove_margin_hi", "remove_margin_q"]].copy()
    sub = sub.rename(
        columns={
            "label": "model",
            "remove_margin": "delta",
            "remove_margin_lo": "lo",
            "remove_margin_hi": "hi",
            "remove_margin_q": "q",
        }
    )
    sub = sub.dropna(subset=["delta", "lo", "hi"], how="any")
    if sub.empty:
        return

    y = np.arange(len(sub))
    sig = pd.to_numeric(sub["q"], errors="coerce") < 0.05

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.axvline(0.0, color=C_GRAY, linestyle="--", linewidth=1.1)
    colors = np.where(sig, C_BLUE, C_LIGHT_GRAY)
    for i in range(len(sub)):
        x = float(sub.iloc[i]["delta"])
        lo = float(sub.iloc[i]["lo"])
        hi = float(sub.iloc[i]["hi"])
        ax.errorbar(
            x,
            i,
            xerr=[[x - lo], [hi - x]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2.2,
            capsize=3.5,
            markersize=6.5,
        )
        q = sub.iloc[i]["q"]
        if np.isfinite(q):
            ax.text(hi + 0.02, i, f"q={q:.3f}", va="center", ha="left", fontsize=9, color=C_GRAY)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["model"].astype(str))
    ax.set_xlabel("Ablation margin delta")
    ax.set_title("Decision-margin shift under prediction-position ablation")
    ax.plot([], [], "o", color=C_BLUE, label="q < 0.05")
    ax.plot([], [], "o", color=C_LIGHT_GRAY, label="q ≥ 0.05")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "fig_margin_shift_models.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_rank_sweep(run_map: dict[str, Any], outdir: Path) -> None:
    rows = []
    explicit = _is_explicit_run_map(run_map)
    for model_name, label in CORE_MODELS:
        rd7 = _exp_run_dir_from_map(run_map, model_name, "exp07_run_dir")
        if rd7 is None and not explicit:
            try:
                rd7 = latest_run("07_rank_sweep", model_name)
            except Exception:
                continue
        if rd7 is None:
            continue
        df = pd.read_csv(rd7 / "tables" / "rank_sweep.csv")
        for k, sub in df.groupby("k", sort=True):
            rows.append({"model": label, "k": int(k), "score": float(sub["stereotype_score"].mean())})

    dfr = pd.DataFrame(rows).sort_values(["model", "k"])
    if dfr.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    colors = {"Gemma-2-2B": C_GREEN, "Gemma-2-2B-IT": C_ORANGE, "Llama-3.2-3B": C_BLUE}
    for model in dfr["model"].unique():
        sub = dfr[dfr["model"] == model]
        ax.plot(sub["k"], sub["score"], marker="o", linewidth=2.0, label=model, color=colors.get(model, C_GRAY))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Axis-mean stereotype score")
    ax.set_title("Rank sweep diagnostic")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "fig_rank_sweep.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_dose_response_split(run_map: dict[str, Any], outdir: Path) -> None:
    rows = []
    explicit = _is_explicit_run_map(run_map)
    for model_name, label in CORE_MODELS:
        rd8 = _exp_run_dir_from_map(run_map, model_name, "exp08_run_dir")
        if rd8 is None and not explicit:
            try:
                rd8 = latest_run("08_dose_response", model_name)
            except Exception:
                continue
        if rd8 is None:
            continue
        df = pd.read_csv(rd8 / "tables" / "dose_response.csv")
        for alpha, sub in df.groupby("alpha", sort=True):
            rows.append(
                {
                    "model": label,
                    "alpha": float(alpha),
                    "score": float(sub["stereotype_score"].mean()),
                    "margin": float(sub["mean_margin"].mean()),
                }
            )

    if not rows:
        return
    dfd = pd.DataFrame(rows).sort_values(["model", "alpha"])
    if dfd.empty:
        return

    colors = {"Gemma-2-2B": C_GREEN, "Gemma-2-2B-IT": C_ORANGE, "Llama-3.2-3B": C_BLUE}

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.8), sharex=True)
    for model in dfd["model"].unique():
        sub = dfd[dfd["model"] == model]
        axes[0].plot(sub["alpha"], sub["score"], marker="o", linewidth=2.0, color=colors.get(model, C_GRAY), label=model)
        axes[1].plot(sub["alpha"], sub["margin"], marker="o", linewidth=2.0, color=colors.get(model, C_GRAY), label=model)

    axes[0].set_ylabel("Axis-mean score")
    axes[0].set_title("Dose response: score")
    axes[0].legend(loc="best")
    axes[1].set_ylabel("Axis-mean margin")
    axes[1].set_xlabel("Injection scale α")
    axes[1].set_title("Dose response: margin")
    fig.tight_layout()
    fig.savefig(outdir / "fig_dose_response.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    setup_style()
    run_map = _load_run_map(args)
    dfm = _collect_asymmetry_rows(run_map)

    fig_headline_asymmetry(dfm, outdir)
    fig_core_asymmetry_detail(dfm, outdir)
    fig_sign_reliability(run_map, outdir)
    fig_margin_shift_delta(dfm, outdir)
    fig_rank_sweep(run_map, outdir)
    fig_dose_response_split(run_map, outdir)
    print("Wrote figures to", outdir)


if __name__ == "__main__":
    main()
