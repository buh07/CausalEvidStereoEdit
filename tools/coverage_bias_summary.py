#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute pre-sampling alignability vs post-sampling inclusion summaries from Exp01 raw pairs, "
            "plus lexical-pattern coverage differences."
        )
    )
    p.add_argument("--run-map", required=True)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def _find_diff_span(tokens_a: list[int], tokens_b: list[int]) -> tuple[int, int, int, int] | None:
    min_len = min(len(tokens_a), len(tokens_b))
    start = 0
    while start < min_len and tokens_a[start] == tokens_b[start]:
        start += 1
    if start == len(tokens_a) and start == len(tokens_b):
        return None
    end_a = len(tokens_a) - 1
    end_b = len(tokens_b) - 1
    while end_a >= start and end_b >= start and tokens_a[end_a] == tokens_b[end_b]:
        end_a -= 1
        end_b -= 1
    return start, end_a, start, end_b


def _span_bin(span_len: int) -> str:
    if span_len <= 0:
        return "no_diff"
    if span_len == 1:
        return "span_tokens_1"
    if span_len == 2:
        return "span_tokens_2"
    if span_len == 3:
        return "span_tokens_3"
    return "span_tokens_4plus"


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _label(model_name: str, payload: dict[str, Any]) -> str:
    if payload.get("label"):
        return str(payload["label"])
    return {
        "google/gemma-2-2b": "Gemma-2-2B",
        "google/gemma-2-2b-it": "Gemma-2-2B-IT",
        "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    }.get(model_name, model_name)


def _load_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = str(row.get("pair_id", "")).strip()
            if pid:
                ids.add(pid)
    return ids


def main() -> None:
    args = parse_args()
    run_map_path = Path(args.run_map)
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    models = run_map.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Run map missing models.")

    out_dir = Path(args.output_dir) if args.output_dir else run_map_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    lexical_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, payload_any in sorted(models.items()):
        if not isinstance(payload_any, dict):
            continue
        payload = payload_any
        label = _label(model_name, payload)
        exp01_dir_raw = payload.get("exp01_mixed_run_dir", "")
        exp16_dir_raw = payload.get("exp16_canonical_run_dir", "")
        if not exp01_dir_raw:
            missing.append(f"{model_name}:missing_exp01")
            continue

        exp01_dir = Path(str(exp01_dir_raw))
        raw_path = exp01_dir / "artifacts" / "contrast_pairs_raw.jsonl"
        if not raw_path.exists():
            missing.append(f"{model_name}:missing_raw_pairs")
            continue

        included_ids: set[str] = set()
        if exp16_dir_raw:
            exp16_pair = Path(str(exp16_dir_raw)) / "tables" / "asymmetry_pair_deltas.csv"
            if exp16_pair.exists():
                df_ids = pd.read_csv(exp16_pair)
                if "pair_id" in df_ids.columns:
                    included_ids = set(df_ids["pair_id"].astype(str).tolist())
        if not included_ids:
            included_ids = _load_ids(exp01_dir / "artifacts" / "aligned_pairs.jsonl")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        total_raw = 0
        total_alignable = 0
        total_included = 0

        by_source_axis: dict[tuple[str, str], Counter[str]] = {}
        by_pattern: dict[str, Counter[str]] = {}
        by_source: dict[str, Counter[str]] = {}

        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                pid = str(row.get("pair_id", "")).strip()
                source = str(row.get("source", "other"))
                axis = str(row.get("axis", "other"))
                s_text = str(row.get("stereotype_text", ""))
                a_text = str(row.get("antistereotype_text", ""))
                total_raw += 1

                s_ids = tokenizer(s_text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
                a_ids = tokenizer(a_text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
                diff = _find_diff_span(s_ids, a_ids)

                alignable = False
                span_len = 0
                if diff is not None:
                    s0, s1, a0, a1 = diff
                    span_s_len = (s1 - s0 + 1)
                    span_a_len = (a1 - a0 + 1)
                    span_len = max(span_s_len, span_a_len)
                    alignable = (
                        span_s_len == 1
                        and span_a_len == 1
                        and s0 == a0
                        and s0 > 0
                        and s_text != a_text
                    )

                included = pid in included_ids

                if alignable:
                    total_alignable += 1
                if included:
                    total_included += 1

                key = (source, axis)
                ctr = by_source_axis.setdefault(key, Counter())
                ctr["raw_count"] += 1
                ctr["pre_alignable_count"] += int(alignable)
                ctr["post_included_count"] += int(included)

                src_ctr = by_source.setdefault(source, Counter())
                src_ctr["raw_count"] += 1
                src_ctr["pre_alignable_count"] += int(alignable)
                src_ctr["post_included_count"] += int(included)

                pat = _span_bin(span_len)
                pctr = by_pattern.setdefault(pat, Counter())
                pctr["raw_count"] += 1
                pctr["pre_alignable_count"] += int(alignable)
                pctr["post_included_count"] += int(included)

        summary_rows.append(
            {
                "model": model_name,
                "model_label": label,
                "exp01_run_dir": str(exp01_dir),
                "exp16_run_dir": str(exp16_dir_raw),
                "raw_pairs": total_raw,
                "pre_sampling_alignable_pairs": total_alignable,
                "pre_sampling_alignability_rate": (total_alignable / total_raw) if total_raw else float("nan"),
                "post_sampling_included_pairs": total_included,
                "post_sampling_inclusion_rate_raw": (total_included / total_raw) if total_raw else float("nan"),
                "post_sampling_inclusion_rate_given_alignable": (total_included / total_alignable)
                if total_alignable
                else float("nan"),
            }
        )

        for (source, axis), ctr in sorted(by_source_axis.items()):
            raw = _safe_int(ctr.get("raw_count", 0))
            alignable = _safe_int(ctr.get("pre_alignable_count", 0))
            included = _safe_int(ctr.get("post_included_count", 0))
            source_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "source": source,
                    "axis": axis,
                    "raw_count": raw,
                    "pre_alignable_count": alignable,
                    "post_included_count": included,
                    "pre_alignability_rate": (alignable / raw) if raw else float("nan"),
                    "post_inclusion_rate_raw": (included / raw) if raw else float("nan"),
                    "post_inclusion_rate_given_alignable": (included / alignable) if alignable else float("nan"),
                }
            )

        for pattern, ctr in sorted(by_pattern.items()):
            raw = _safe_int(ctr.get("raw_count", 0))
            alignable = _safe_int(ctr.get("pre_alignable_count", 0))
            included = _safe_int(ctr.get("post_included_count", 0))
            lexical_rows.append(
                {
                    "model": model_name,
                    "model_label": label,
                    "lexical_pattern": pattern,
                    "raw_count": raw,
                    "pre_alignable_count": alignable,
                    "post_included_count": included,
                    "pre_alignability_rate": (alignable / raw) if raw else float("nan"),
                    "post_inclusion_rate_raw": (included / raw) if raw else float("nan"),
                    "post_inclusion_rate_given_alignable": (included / alignable) if alignable else float("nan"),
                }
            )

    summary_path = out_dir / "coverage_bias_summary.csv"
    source_path = out_dir / "coverage_bias_by_source_axis.csv"
    lexical_path = out_dir / "coverage_bias_lexical_patterns.csv"

    summary_cols = [
        "model",
        "model_label",
        "exp01_run_dir",
        "exp16_run_dir",
        "raw_pairs",
        "pre_sampling_alignable_pairs",
        "pre_sampling_alignability_rate",
        "post_sampling_included_pairs",
        "post_sampling_inclusion_rate_raw",
        "post_sampling_inclusion_rate_given_alignable",
    ]
    source_cols = [
        "model",
        "model_label",
        "source",
        "axis",
        "raw_count",
        "pre_alignable_count",
        "post_included_count",
        "pre_alignability_rate",
        "post_inclusion_rate_raw",
        "post_inclusion_rate_given_alignable",
    ]
    lexical_cols = [
        "model",
        "model_label",
        "lexical_pattern",
        "raw_count",
        "pre_alignable_count",
        "post_included_count",
        "pre_alignability_rate",
        "post_inclusion_rate_raw",
        "post_inclusion_rate_given_alignable",
    ]
    pd.DataFrame(summary_rows, columns=summary_cols).to_csv(summary_path, index=False)
    pd.DataFrame(source_rows, columns=source_cols).to_csv(source_path, index=False)
    pd.DataFrame(lexical_rows, columns=lexical_cols).to_csv(lexical_path, index=False)

    meta = {
        "run_map": str(run_map_path),
        "summary": str(summary_path),
        "by_source_axis": str(source_path),
        "lexical_patterns": str(lexical_path),
        "missing": missing,
        "semantics": {
            "pre_sampling_alignable_pairs": "Pairs that satisfy single-token alignability in raw contrast pool before post-alignment subsampling.",
            "post_sampling_included_pairs": "Pairs included in the downstream analysis slice (Exp16 pair table when available).",
        },
    }
    meta_path = out_dir / "coverage_bias_summary_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(summary_path)
    print(source_path)
    print(lexical_path)
    print(meta_path)


if __name__ == "__main__":
    main()
