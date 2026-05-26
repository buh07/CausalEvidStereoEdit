#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CORE_ORDER = ["Gemma-2-2B", "Gemma-2-2B-IT", "Llama-3.2-3B"]


def _fmt_num(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except Exception:
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_int(v: Any) -> str:
    try:
        x = int(round(float(v)))
    except Exception:
        return "n/a"
    return str(x)


def _model_sort_key(label: str) -> tuple[int, str]:
    if label in CORE_ORDER:
        return (CORE_ORDER.index(label), label)
    return (999, label)


def _tex(s: Any) -> str:
    t = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    return t


def _source_mix_str(vals: pd.Series) -> str:
    if vals.empty:
        return "n/a"
    c = Counter(str(v) for v in vals.tolist())
    parts = []
    for k in ["crows_pairs", "stereoset_intrasentence", "seegull", "SEEGeL"]:
        if c.get(k, 0) > 0:
            short = {
                "crows_pairs": "CrowS",
                "stereoset_intrasentence": "StereoSet",
                "seegull": "SEEGeL",
                "SEEGeL": "SEEGeL",
            }[k]
            parts.append(f"{short} {c[k]}")
    # include any unmapped sources
    for k in sorted(c):
        if k not in {"crows_pairs", "stereoset_intrasentence", "seegull", "SEEGeL"}:
            parts.append(f"{k} {c[k]}")
    return "; ".join(parts) if parts else "n/a"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_occupancy_setting(report_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(report_dir / "occupancy_setting_comparison.csv")
    rows = []
    for _, r in df.sort_values("model_label", key=lambda s: s.map(_model_sort_key)).iterrows():
        rows.append(
            f"{_tex(r['model_label'])} & {_fmt_int(r.get('n_overlap_pairs'))} & "
            f"{_fmt_num(r.get('mean_abs_true_proj_exp16'))} & {_fmt_num(r.get('mean_abs_true_proj_exp26'))} & "
            f"{_fmt_num(r.get('mean_abs_true_proj_same_minus_cross'))} \\\\"
        )
    body = "\n".join(rows) if rows else "Pending & -- & -- & -- & -- \\\\"
    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & $n_{\\text{overlap}}$ & $|h\\cdot\\hat d|$ cross & $|h\\cdot\\hat d|$ same & same$-$cross \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Direction-occupancy setting comparison from \\texttt{reports/occupancy\\_setting\\_comparison.csv}.}\n"
        "\\label{tab:occupancy-setting}\n"
        "\\end{table}\n"
    )
    _write(out_dir / "occupancy_setting.tex", tex)


def render_occupancy_coupling(report_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(report_dir / "occupancy_effect_coupling.csv")
    rows = []
    setting_map = {"exp16_cross": "Cross", "exp26_same": "Same"}
    order = {"Cross": 0, "Same": 1}
    df = df.copy()
    df["setting_label"] = df["setting"].map(setting_map).fillna(df["setting"])
    df = df.sort_values(
        ["model_label", "setting_label"],
        key=lambda s: s.map(lambda x: (_model_sort_key(str(x)), order.get(str(x), 99)) if s.name == "model_label" else order.get(str(x), 99))
        if s.name in {"model_label", "setting_label"}
        else s,
    )
    for _, r in df.iterrows():
        rows.append(
            f"{_tex(r['model_label'])} & {_tex(r['setting_label'])} & {_fmt_int(r.get('n_pairs'))} & "
            f"{_fmt_num(r.get('spearman_abs_true_vs_abs_remove_margin'))} & "
            f"{_fmt_num(r.get('spearman_abs_true_vs_abs_remove_margin_p'))} \\\\"
        )
    body = "\n".join(rows) if rows else "Pending & -- & -- & -- & -- \\\\"
    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & Setting & $n$ & $\\rho(|h\\cdot\\hat d|,|\\Delta m|)$ & $p$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Occupancy-to-effect coupling from \\texttt{reports/occupancy\\_effect\\_coupling.csv}.}\n"
        "\\label{tab:occupancy-coupling}\n"
        "\\end{table}\n"
    )
    _write(out_dir / "occupancy_coupling.tex", tex)


def render_exp26_methods(run_map: dict[str, Any], report_dir: Path, out_dir: Path) -> None:
    inter = pd.read_csv(report_dir / "cross_vs_same_interaction.csv")
    overlap_col = "n_overlap"
    if overlap_col not in inter.columns and "n_overlap_pairs" in inter.columns:
        overlap_col = "n_overlap_pairs"
    overlap_by_label = {}
    for _, r in inter.iterrows():
        try:
            overlap_by_label[str(r["model_label"])] = int(r[overlap_col])
        except Exception:
            continue

    rows = []
    models = run_map.get("models", {})
    for model_name, payload in models.items():
        if not isinstance(payload, dict):
            continue
        exp26_dir_raw = payload.get("exp26_run_dir", "")
        if not exp26_dir_raw:
            continue
        label = str(payload.get("label") or model_name)
        exp26_dir = Path(str(exp26_dir_raw))
        tm = pd.read_csv(exp26_dir / "tables" / "ar_same_position_template_meta.csv").iloc[0]
        mx = pd.read_csv(exp26_dir / "tables" / "ar_same_position_matrix.csv")
        rm = mx[mx["condition"] == "remove_on_stereo"].iloc[0]
        pairs = pd.read_csv(exp26_dir / "tables" / "ar_same_position_pair_deltas.csv")
        mix = _source_mix_str(pairs["source"]) if "source" in pairs.columns else "n/a"
        rows.append(
            {
                "model_label": label,
                "prompt_variant": str(tm.get("prompt_variant", "n/a")),
                "n_train": int(tm.get("n_train_pairs", np.nan)),
                "n_test": int(tm.get("n_test_pairs", np.nan)),
                "n_overlap": overlap_by_label.get(label, int(tm.get("n_test_pairs", np.nan))),
                "baseline_score": float(rm.get("stereotype_score_baseline", np.nan)),
                "source_mix": mix,
            }
        )

    rows = sorted(rows, key=lambda r: _model_sort_key(r["model_label"]))
    body_rows = []
    for r in rows:
        body_rows.append(
            f"{_tex(r['model_label'])} & {_tex(r['prompt_variant'])} & {r['n_train']} & {r['n_test']} & {r['n_overlap']} & "
            f"{_fmt_num(r['baseline_score'])} & {_tex(r['source_mix'])} \\\\"
        )
    body = "\n".join(body_rows) if body_rows else "Pending & -- & -- & -- & -- & -- & -- \\\\"
    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Model & Prompt variant & $n_{\\text{train}}$ & $n_{\\text{test}}$ & $n_{\\text{overlap}}$ & Baseline score & Source mix (test) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Exp26 methods box: deterministic prefix-difference templates with same-position intervention at the scored token. Rows are derived from each model's \\texttt{ar\\_same\\_position\\_template\\_meta.csv}, \\texttt{ar\\_same\\_position\\_matrix.csv}, and pair-level source counts.}\n"
        "\\label{tab:exp26-methods-box}\n"
        "\\end{table}\n"
    )
    _write(out_dir / "exp26_methods_box.tex", tex)


def render_exp30_axis(report_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(report_dir / "exp30_backfire_axis_decomposition.csv")
    rows = []
    for label in CORE_ORDER:
        sub = df[(df["model_label"] == label) & (df["condition"].astype(str) == "stereoset_to_crows")].copy()
        if sub.empty:
            sub = df[df["model_label"] == label].copy()
        if sub.empty:
            continue
        sub["abs_delta"] = sub["stereotype_score_delta"].abs()
        top = sub.sort_values("abs_delta", ascending=False).iloc[0]
        rows.append(
            f"{_tex(label)} & {_tex(top.get('condition',''))} & {_tex(top.get('axis',''))} & {_fmt_int(top.get('n_pairs'))} & {_fmt_num(top.get('stereotype_score_delta'))} \\\\"
        )
    body = "\n".join(rows) if rows else "Pending & -- & -- & -- & -- \\\\"
    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & Condition & Axis & $n_{\\text{pairs}}$ & $\\Delta$score \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Exp30 axis-level decomposition (largest absolute axis row per model) from \\texttt{reports/exp30\\_backfire\\_axis\\_decomposition.csv}.}\n"
        "\\label{tab:exp30-backfire-axis}\n"
        "\\end{table}\n"
    )
    _write(out_dir / "exp30_backfire_axis.tex", tex)


def render_exp30_alignment(report_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(report_dir / "exp30_direction_alignment_summary.csv")
    rows = []
    df = df.sort_values("model_label", key=lambda s: s.map(_model_sort_key))
    for _, r in df.iterrows():
        rows.append(
            f"{_tex(r['model_label'])} & {_fmt_int(r.get('n_shared_axis_layers'))} & "
            f"{_fmt_num(r.get('mean_cosine_stereoset_vs_crows'))} & {_fmt_num(r.get('neg_cosine_fraction'))} \\\\"
        )
    body = "\n".join(rows) if rows else "Pending & -- & -- & -- \\\\"
    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Model & Shared axis-layers & Mean cosine (StereoSet vs CrowS) & Negative-cosine fraction \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Source-direction alignment diagnostics from \\texttt{reports/exp30\\_direction\\_alignment\\_summary.csv}.}\n"
        "\\label{tab:exp30-direction-alignment}\n"
        "\\end{table}\n"
    )
    _write(out_dir / "exp30_direction_alignment.tex", tex)


def render_crossfit(report_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(report_dir / "crossfit_split_clean_summary.csv")
    rows = []
    df = df.sort_values("model_label", key=lambda s: s.map(_model_sort_key))
    for _, r in df.iterrows():
        rows.append(
            f"{_tex(r['model_label'])} & {_fmt_int(r.get('n_folds'))} & {_fmt_num(r.get('dla_sign_rate_mean'))} & "
            f"{_fmt_num(r.get('dla_sign_rate_sd'))} & {_fmt_num(r.get('suppressor_fraction_mean'))} \\\\"
        )
    body = "\n".join(rows) if rows else "Pending & -- & -- & -- & -- \\\\"
    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & Folds & Mean DLA sign rate & SD DLA sign rate & Mean suppressor fraction \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Cross-fit split-clean summary from \\texttt{reports/crossfit\\_split\\_clean\\_summary.csv}.}\n"
        "\\label{tab:crossfit-split-clean-summary}\n"
        "\\end{table}\n"
    )
    _write(out_dir / "crossfit_split_clean_summary.tex", tex)


def render_mistral(report_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(report_dir / "mistral_power_report.csv")
    r = df.iloc[0]
    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Model & $n$ & Contrast & 95\\% CI & MDE & Power vs SESOI & Verdict \\\\\n"
        "\\midrule\n"
        f"{_tex(r['model_label'])} & {_fmt_int(r.get('n_pairs'))} & {_fmt_num(r.get('observed_score_contrast'))} & "
        f"[{_fmt_num(r.get('score_ci_low'))}, {_fmt_num(r.get('score_ci_high'))}] & "
        f"{_fmt_num(r.get('mde_score_approx'))} & {_tex(r.get('power_vs_sesoi','n/a'))} & {_tex(r.get('verdict','n/a'))} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Mistral power framing from \\texttt{reports/mistral\\_power\\_report.csv}.}\n"
        "\\label{tab:mistral-power}\n"
        "\\end{table}\n"
    )
    _write(out_dir / "mistral_power.tex", tex)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render auto appendix tables from Rev3 report artifacts.")
    p.add_argument("--run-map", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--output-dir", default="paper/build/auto_tables")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_map = json.loads(Path(args.run_map).read_text(encoding="utf-8"))
    report_dir = Path(args.report_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_occupancy_setting(report_dir, out_dir)
    render_occupancy_coupling(report_dir, out_dir)
    render_exp26_methods(run_map, report_dir, out_dir)
    render_exp30_axis(report_dir, out_dir)
    render_exp30_alignment(report_dir, out_dir)
    render_crossfit(report_dir, out_dir)
    render_mistral(report_dir, out_dir)

    print(out_dir)


if __name__ == "__main__":
    main()
