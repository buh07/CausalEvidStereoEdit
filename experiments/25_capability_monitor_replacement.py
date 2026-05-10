#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import load_directions_npz, write_csv, write_json
from stereacl.interventions import make_direction_projection_hook
from stereacl.modeling import forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 25: higher-sample capability monitor under direction-ablation intervention."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mmlu-samples", type=int, default=600)
    parser.add_argument("--hellaswag-samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--exp1-run-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _latest_exp1(model: str) -> Path:
    root = PROJECT_ROOT / "results" / "01_layerwise_probing"
    candidates = sorted(root.glob("*/*/manifest.json"))
    best: tuple[str, Path] | None = None
    for mp in candidates:
        payload = json.loads(mp.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        params = payload.get("parameters", {})
        if params.get("model") != model:
            continue
        if params.get("direction_position") != "trait":
            continue
        rd = Path(payload["run_dir"])
        if not (rd / "artifacts" / "directions_layerwise.npz").exists():
            continue
        ended = payload.get("ended_at_utc") or ""
        if best is None or ended > best[0]:
            best = (ended, rd)
    if best is None:
        raise FileNotFoundError(f"No trait-position Exp01 run found for model={model}")
    return best[1]


def _aggregate_global_directions(directions: dict[tuple[str, int], np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, list[np.ndarray]] = {}
    for (_axis, layer), vec in directions.items():
        out.setdefault(int(layer), []).append(vec)
    return {layer: np.mean(np.stack(vecs), axis=0) for layer, vecs in out.items()}


def _compose(
    first: Callable[[torch.Tensor], torch.Tensor] | None,
    second: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if first is None:
        return second

    def _c(x: torch.Tensor) -> torch.Tensor:
        return second(first(x))

    return _c


def _encode(tokenizer, text: str, device: torch.device, max_length: int) -> dict[str, torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in enc.items()}


def _continuation_logprob(
    *,
    bundle,
    prompt: str,
    continuation: str,
    max_length: int,
    residual_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
) -> float:
    tok = bundle.tokenizer
    dev = bundle.device
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(prompt + continuation, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return -1e9
    ids = full_ids[-max_length:]
    offset = max(0, len(prompt_ids) - (len(full_ids) - len(ids)))

    input_ids = torch.tensor([ids], device=dev)
    attention_mask = torch.ones_like(input_ids)
    cap = forward_with_component_capture(
        model=bundle.model,
        encoded_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        output_hidden_states=False,
        capture_attention=False,
        capture_mlp=False,
        residual_patch_map=residual_patch_map,
    )
    logits = cap.logits[0]  # [seq, vocab]

    # token i predicts ids[i]
    total = 0.0
    cont_start = max(1, offset)
    for i in range(cont_start, len(ids)):
        prev_logits = logits[i - 1]
        log_probs = torch.log_softmax(prev_logits.float(), dim=-1)
        total += float(log_probs[ids[i]].item())
    return total


def _mmlu_prompt(row: dict[str, Any]) -> tuple[str, list[str], int]:
    q = str(row["question"])
    choices = [str(c) for c in row["choices"]]
    ans = int(row["answer"])
    prompt = q + "\nA) " + choices[0] + "\nB) " + choices[1] + "\nC) " + choices[2] + "\nD) " + choices[3] + "\nAnswer:"
    # options are letter tokens to reduce formatting effects
    return prompt, [" A", " B", " C", " D"], ans


def _hellaswag_prompt(row: dict[str, Any]) -> tuple[str, list[str], int]:
    ctx = (str(row.get("ctx", "")) + " " + str(row.get("ctx_b", ""))).strip()
    endings = row.get("endings", [])
    options = [" " + str(e) for e in endings[:4]]
    label = int(row.get("label", 0))
    prompt = ctx
    return prompt, options, label


def _evaluate_multiple_choice(
    *,
    bundle,
    items: list[tuple[str, list[str], int]],
    max_length: int,
    residual_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None,
) -> np.ndarray:
    correct = []
    for prompt, options, gold in items:
        scores = [
            _continuation_logprob(
                bundle=bundle,
                prompt=prompt,
                continuation=opt,
                max_length=max_length,
                residual_patch_map=residual_patch_map,
            )
            for opt in options
        ]
        pred = int(np.argmax(scores))
        correct.append(1.0 if pred == gold else 0.0)
    return np.array(correct, dtype=float)


def _rounded(v: float | int | None) -> float | str:
    if v is None:
        return ""
    try:
        x = float(v)
    except Exception:
        return ""
    if np.isnan(x) or np.isinf(x):
        return ""
    return round(x, 8)


def _to_float_or_nan(x: Any) -> float:
    try:
        if x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _apply_fdr(rows: list[dict[str, Any]], p_col: str, q_col: str) -> None:
    pvals = [_to_float_or_nan(r.get(p_col, "")) for r in rows]
    qvals = benjamini_hochberg(pvals)
    for i, q in enumerate(qvals):
        rows[i][q_col] = _rounded(q)


def main() -> None:
    args = parse_args()
    ctx = start_run("25", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = Path(args.exp1_run_dir) if args.exp1_run_dir else _latest_exp1(args.model)
        directions = load_directions_npz(exp1_dir / "artifacts" / "directions_layerwise.npz")
        global_dirs = _aggregate_global_directions(directions)

        refs = {
            "exp1_run_dir": str(exp1_dir),
            "direction_layers": len(global_dirs),
            "mmlu_samples": args.mmlu_samples,
            "hellaswag_samples": args.hellaswag_samples,
        }
        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(refs_path, refs)
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            complete_run(ctx, metrics={"dry_run": True, **refs})
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)

        residual_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
        for layer, vec in global_dirs.items():
            idx = int(layer) - 1
            d = torch.tensor(vec, device=bundle.device, dtype=torch.float32)
            hook = make_direction_projection_hook(d)
            residual_patch_map[idx] = _compose(residual_patch_map.get(idx), hook)

        rng = np.random.default_rng(args.seed)

        # MMLU
        mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
        mmlu_idx = rng.choice(len(mmlu_ds), size=min(args.mmlu_samples, len(mmlu_ds)), replace=False)
        mmlu_items = [_mmlu_prompt(mmlu_ds[int(i)]) for i in mmlu_idx]

        # HellaSwag
        hs_ds = load_dataset("hellaswag", split="validation")
        hs_idx = rng.choice(len(hs_ds), size=min(args.hellaswag_samples, len(hs_ds)), replace=False)
        hs_items = [_hellaswag_prompt(hs_ds[int(i)]) for i in hs_idx]

        rows: list[dict[str, Any]] = []

        for task_name, items in [("mmlu", mmlu_items), ("hellaswag", hs_items)]:
            base = _evaluate_multiple_choice(
                bundle=bundle,
                items=items,
                max_length=args.max_length,
                residual_patch_map=None,
            )
            edited = _evaluate_multiple_choice(
                bundle=bundle,
                items=items,
                max_length=args.max_length,
                residual_patch_map=residual_patch_map,
            )
            diff = edited - base
            acc_base = float(np.mean(base))
            acc_edit = float(np.mean(edited))
            acc_delta = acc_edit - acc_base
            ci = bootstrap_mean_ci(diff, n_resamples=args.bootstrap_n, rng=rng)
            p_sign, _, _ = paired_sign_test(diff)

            rows.append(
                {
                    "task": task_name,
                    "n_items": len(items),
                    "accuracy_baseline": _rounded(acc_base),
                    "accuracy_ablated": _rounded(acc_edit),
                    "accuracy_delta": _rounded(acc_delta),
                    "delta_ci_low": _rounded(ci.ci_low),
                    "delta_ci_high": _rounded(ci.ci_high),
                    "p_sign": _rounded(p_sign),
                    "q_sign": "",
                }
            )

        _apply_fdr(rows, "p_sign", "q_sign")

        out_path = ctx.tables_dir / "capability_monitor_replacement.csv"
        write_csv(
            out_path,
            rows,
            fieldnames=[
                "task",
                "n_items",
                "accuracy_baseline",
                "accuracy_ablated",
                "accuracy_delta",
                "delta_ci_low",
                "delta_ci_high",
                "p_sign",
                "q_sign",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Exp25 paired capability deltas on above-chance tasks.")

        complete_run(
            ctx,
            metrics={
                "tasks": len(rows),
                "total_items": int(sum(int(r["n_items"]) for r in rows)),
                "dry_run": False,
            },
        )
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
