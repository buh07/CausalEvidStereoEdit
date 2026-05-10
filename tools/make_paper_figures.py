#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/jumbo/lisp/f004ndc/StereACL")
OUTDIR = ROOT / "paper" / "build"
OUTDIR.mkdir(parents=True, exist_ok=True)

CORE_MODELS = [
    ("google/gemma-2-2b", "Gemma-2-2B"),
    ("google/gemma-2-2b-it", "Gemma-2-2B-IT"),
    ("meta-llama/Llama-3.2-3B", "Llama-3.2-3B"),
]

ALL_MODELS = [
    ("google/gemma-2-2b", "Gemma-2-2B"),
    ("google/gemma-2-2b-it", "Gemma-2-2B-IT"),
    ("meta-llama/Llama-3.2-3B", "Llama-3.2-3B"),
    ("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
    ("/jumbo/lisp/f004ndc/models/mistral-7b-v0.1", "Mistral-7B"),
    ("/jumbo/lisp/f004ndc/models/olmo-2-7b", "OLMo-2-7B"),
]

# Colorblind-safe palette.
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_GRAY = "#666666"
C_LIGHT_GRAY = "#C7C7C7"


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
        d = json.loads(mf.read_text())
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


def _pick_row(df: pd.DataFrame, condition_values: Iterable[str]) -> pd.Series:
    for cond in condition_values:
        m = df[df["condition"] == cond]
        if not m.empty:
            return m.iloc[0]
    raise RuntimeError(f"None of the conditions found: {list(condition_values)}")


def fig_headline_asymmetry_7model() -> None:
    # Use manuscript values directly so figures stay synchronized with tables.
    rows = [
        # Core three-model set (Table 2 + matched-contrast margin rows).
        {"model": "Gemma-2-2B", "score": 0.100, "score_lo": 0.042, "score_hi": 0.167, "score_q": 0.0075, "margin": 1.116, "margin_lo": 0.289, "margin_hi": 2.250, "margin_q": 0.0102},
        {"model": "Gemma-2-2B-IT", "score": 0.050, "score_lo": -0.008, "score_hi": 0.108, "score_q": 0.1800, "margin": 0.842, "margin_lo": 0.289, "margin_hi": 1.525, "margin_q": 0.0229},
        {"model": "Llama-3.2-3B", "score": 0.250, "score_lo": 0.150, "score_hi": 0.367, "score_q": 0.000009, "margin": 3.120, "margin_lo": 2.259, "margin_hi": 4.080, "margin_q": 0.000002},
        # Four-model extension (Appendix Table ext4-summary).
        {"model": "Qwen2.5-3B", "score": 0.1375, "score_lo": 0.0350, "score_hi": 0.2350, "score_q": 0.00342, "margin": 0.9572, "margin_lo": 0.2350, "margin_hi": 1.6210, "margin_q": 0.00499},
        {"model": "Qwen2.5-3B-Instruct", "score": 0.1625, "score_lo": 0.0622, "score_hi": 0.2625, "score_q": 0.00443, "margin": 1.5088, "margin_lo": 0.6803, "margin_hi": 2.3705, "margin_q": 0.000524},
        {"model": "Mistral-7B", "score": 0.0909, "score_lo": -0.0380, "score_hi": 0.2200, "score_q": 0.3880, "margin": 1.8213, "margin_lo": 0.6500, "margin_hi": 2.9500, "margin_q": 0.00425},
        {"model": "OLMo-2-7B", "score": 0.2273, "score_lo": 0.0700, "score_hi": 0.3850, "score_q": 0.0213, "margin": 1.6676, "margin_lo": 0.9500, "margin_hi": 2.5300, "margin_q": 0.000744},
    ]

    dfm = pd.DataFrame(rows)
    # Keep a stable narrative order: core first, then extension.
    desired = [label for _, label in ALL_MODELS]
    dfm["model"] = pd.Categorical(dfm["model"], categories=desired, ordered=True)
    dfm = dfm.sort_values("model")

    y = np.arange(len(dfm))

    # Stacked layout is easier to read in single-column paper placement.
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.8), sharey=True)

    for ax, metric, lo_col, hi_col, q_col, xlabel, title in [
        (
            axes[0],
            "score",
            "score_lo",
            "score_hi",
            "score_q",
            "Inject − remove score contrast",
            "Behavioral endpoint (5/7 significant)",
        ),
        (
            axes[1],
            "margin",
            "margin_lo",
            "margin_hi",
            "margin_q",
            "Inject − remove margin contrast",
            "Mechanistic endpoint (7/7 significant)",
        ),
    ]:
        ax.axvline(0.0, color=C_GRAY, linestyle="--", linewidth=1.1, zorder=0)
        sig = dfm[q_col] < 0.05
        colors = np.where(sig, C_BLUE, C_LIGHT_GRAY)
        for i in range(len(dfm)):
            x = float(dfm.iloc[i][metric])
            lo = float(dfm.iloc[i][lo_col])
            hi = float(dfm.iloc[i][hi_col])
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
        ax.set_yticklabels(dfm["model"])
        ax.invert_yaxis()
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")

    axes[0].set_ylabel("Model")
    axes[1].set_ylabel("Model")

    # Legend handles.
    axes[0].plot([], [], "o", color=C_BLUE, label="q < 0.05")
    axes[0].plot([], [], "o", color=C_LIGHT_GRAY, label="q ≥ 0.05")
    axes[0].legend(loc="lower right", frameon=True)

    fig.tight_layout(h_pad=1.2)
    fig.savefig(OUTDIR / "fig_asymmetry_contrast.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_core_asymmetry_detail() -> None:
    # Values from main table + matched-contrast table.
    rows = [
        {"model": "Gemma-2-2B", "remove": -0.025, "remove_lo": -0.083, "remove_hi": 0.025, "inject": 0.092, "inject_lo": 0.042, "inject_hi": 0.158, "contrast": 0.100, "contrast_lo": 0.042, "contrast_hi": 0.167},
        {"model": "Gemma-2-2B-IT", "remove": 0.017, "remove_lo": -0.025, "remove_hi": 0.067, "inject": 0.092, "inject_lo": 0.042, "inject_hi": 0.150, "contrast": 0.050, "contrast_lo": -0.008, "contrast_hi": 0.108},
        {"model": "Llama-3.2-3B", "remove": -0.017, "remove_lo": -0.063, "remove_hi": 0.033, "inject": 0.292, "inject_lo": 0.192, "inject_hi": 0.408, "contrast": 0.250, "contrast_lo": 0.150, "contrast_hi": 0.367},
    ]

    dfa = pd.DataFrame(rows)
    order = ["Llama-3.2-3B", "Gemma-2-2B-IT", "Gemma-2-2B"]
    dfa["model"] = pd.Categorical(dfa["model"], categories=order, ordered=True)
    dfa = dfa.sort_values("model")
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
        dfa["contrast"],
        y,
        xerr=[dfa["contrast"] - dfa["contrast_lo"], dfa["contrast_hi"] - dfa["contrast"]],
        fmt="s",
        color=C_BLUE,
        label="Direct contrast (matched operators)",
        capsize=3.5,
        elinewidth=2.2,
        markersize=6.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(dfa["model"])
    ax.set_xlabel("Score delta")
    ax.set_title("Core-model asymmetry detail")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_asymmetry_core_detail.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_sign_reliability() -> None:
    rd14 = latest_run("14_sign_reliability_audit", "google/gemma-2-2b-it")
    df = pd.read_csv(rd14 / "tables" / "sign_reliability_by_axis.csv")
    df = df[pd.to_numeric(df["n_dla_sign_agreement"], errors="coerce").fillna(0) > 0].copy()
    df = df.sort_values("dla_sign_agreement_rate")

    y = np.arange(len(df))
    vals = df["dla_sign_agreement_rate"].astype(float).to_numpy()
    lo = df["dla_sign_agreement_ci_low"].astype(float).to_numpy()
    hi = df["dla_sign_agreement_ci_high"].astype(float).to_numpy()

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
    fig.savefig(OUTDIR / "fig_gemmait_sign_forest.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_margin_shift_delta() -> None:
    # Values from Appendix main-intervention margin table.
    rows = [
        {"model": "Gemma-2-2B", "delta": -0.570, "lo": -1.069, "hi": -0.166, "q": 0.0134},
        {"model": "Gemma-2-2B-IT", "delta": -0.218, "lo": -0.502, "hi": 0.013, "q": 0.4092},
        {"model": "Llama-3.2-3B", "delta": -0.157, "lo": -0.307, "hi": -0.045, "q": 0.5324},
    ]

    dfm = pd.DataFrame(rows)
    y = np.arange(len(dfm))
    sig = dfm["q"] < 0.05

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.axvline(0.0, color=C_GRAY, linestyle="--", linewidth=1.1)
    colors = np.where(sig, C_BLUE, C_LIGHT_GRAY)
    for i in range(len(dfm)):
        x = dfm.iloc[i]["delta"]
        lo = dfm.iloc[i]["lo"]
        hi = dfm.iloc[i]["hi"]
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
        ax.text(
            hi + 0.02,
            i,
            f"q={dfm.iloc[i]['q']:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color=C_GRAY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(dfm["model"])
    ax.set_xlabel("Ablation margin delta")
    ax.set_title("Decision-margin shift under prediction-position ablation")
    ax.plot([], [], "o", color=C_BLUE, label="q < 0.05")
    ax.plot([], [], "o", color=C_LIGHT_GRAY, label="q ≥ 0.05")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_margin_shift_models.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_rank_sweep() -> None:
    rows = []
    for model_key, label in CORE_MODELS:
        rd7 = latest_run("07_rank_sweep", model_key)
        df = pd.read_csv(rd7 / "tables" / "rank_sweep.csv")
        for k, sub in df.groupby("k", sort=True):
            rows.append(
                {
                    "model": label,
                    "k": int(k),
                    "score": float(sub["stereotype_score"].mean()),
                }
            )

    dfr = pd.DataFrame(rows).sort_values(["model", "k"])
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    colors = {
        "Gemma-2-2B": C_GREEN,
        "Gemma-2-2B-IT": C_ORANGE,
        "Llama-3.2-3B": C_BLUE,
    }
    for model in dfr["model"].unique():
        sub = dfr[dfr["model"] == model]
        ax.plot(sub["k"], sub["score"], marker="o", linewidth=2.0, label=model, color=colors[model])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Axis-mean stereotype score")
    ax.set_title("Rank sweep diagnostic")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_rank_sweep.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_dose_response_split() -> None:
    rows = []
    for model_key, label in CORE_MODELS:
        rd8 = latest_run("08_dose_response", model_key)
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

    dfd = pd.DataFrame(rows).sort_values(["model", "alpha"])
    colors = {
        "Gemma-2-2B": C_GREEN,
        "Gemma-2-2B-IT": C_ORANGE,
        "Llama-3.2-3B": C_BLUE,
    }

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.8), sharex=True)
    for model in dfd["model"].unique():
        sub = dfd[dfd["model"] == model]
        axes[0].plot(sub["alpha"], sub["score"], marker="o", linewidth=2.0, color=colors[model], label=model)
        axes[1].plot(sub["alpha"], sub["margin"], marker="o", linewidth=2.0, color=colors[model], label=model)

    axes[0].set_ylabel("Axis-mean score")
    axes[0].set_title("Dose response: score")
    axes[0].legend(loc="best")
    axes[1].set_ylabel("Axis-mean margin")
    axes[1].set_xlabel("Injection scale α")
    axes[1].set_title("Dose response: margin")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_dose_response.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    fig_headline_asymmetry_7model()
    fig_core_asymmetry_detail()
    fig_sign_reliability()
    fig_margin_shift_delta()
    fig_rank_sweep()
    fig_dose_response_split()
    print("Wrote figures to", OUTDIR)


if __name__ == "__main__":
    main()
