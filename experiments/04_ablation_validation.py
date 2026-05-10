#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereacl.analysis import compute_score_from_logits, load_directions_npz, write_csv, write_json
from stereacl.data import ContrastPair
from stereacl.interventions import (
    make_direction_injection_hook,
    make_direction_projection_hook,
    make_direction_projection_at_position_hook,
    make_zero_position_hook,
)
from stereacl.modeling import encode_text, forward_with_component_capture, load_model_bundle
from stereacl.run_context import complete_run, fail_run, start_run
from stereacl.stats import benjamini_hochberg, bootstrap_mean_ci, paired_sign_test, wilcoxon_signed_rank_safe
from stereacl.token_alignment import AlignedPair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 04: direction/component ablation validation on held-out pairs."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--heldout-pairs", type=int, default=200)
    parser.add_argument("--top-k-components", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bbq-samples", type=int, default=0, help="Optional extra BBQ eval samples.")
    parser.add_argument(
        "--mmlu-samples",
        type=int,
        default=0,
        help="Optional sampled MMLU test questions for capability control.",
    )
    parser.add_argument(
        "--mmlu-shots",
        type=int,
        default=5,
        help="Number of few-shot examples per MMLU question.",
    )
    parser.add_argument(
        "--strict-controls",
        action="store_true",
        help="Add random-rank, norm-matched-random, and corrupt-to-clean control conditions.",
    )
    parser.add_argument(
        "--include-label-permutation-control",
        action="store_true",
        help=(
            "Include confounded label-permutation direction control for diagnostics only. "
            "This condition is excluded by default because permuted directions can remain correlated "
            "with the original stereotype direction."
        ),
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=0,
        help="Number of bootstrap resamples for 95%% CIs on stereotype_score and mean_margin (0 = disabled).",
    )
    parser.add_argument(
        "--on-manifold",
        action="store_true",
        help="Use paired anti-state residual replacement instead of direction projection.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--exp1-run-dir",
        default="",
        help="Explicit Experiment 01 run directory. If omitted, latest completed run is used.",
    )
    parser.add_argument(
        "--exp3-run-dir",
        default="",
        help="Explicit Experiment 03 run directory. If omitted, latest completed run is used.",
    )
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
            missing = False
            for rel in required_relpaths:
                if not (run_dir / rel).exists():
                    missing = True
                    break
            if missing:
                continue
        if best is None or ended > best[0]:
            best = (ended, run_dir)
    if best is None:
        required_msg = f" with required files {required_relpaths}" if required_relpaths else ""
        raise FileNotFoundError(f"No completed run found for {experiment_slug}{required_msg}.")
    return best[1]


def _load_aligned_pairs(path: Path) -> list[AlignedPair]:
    pairs: list[AlignedPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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


def _compose_hooks(
    first: Callable[[torch.Tensor], torch.Tensor] | None,
    second: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if first is None:
        return second

    def _chained(out: torch.Tensor) -> torch.Tensor:
        return second(first(out))

    return _chained


def _load_selected_components(exp3_dir: Path, top_k: int) -> dict[str, list[tuple[str, int]]]:
    path = exp3_dir / "tables" / "attribution_patch_scores.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    selected: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for axis, group in df.groupby("axis"):
        ranked = group.sort_values("mean_abs_attr_score", ascending=False).head(top_k)
        for _, row in ranked.iterrows():
            selected[str(axis)].append((str(row["component_type"]), int(row["layer"])))
    return selected


def _aggregate_global_directions(directions: dict[tuple[str, int], np.ndarray]) -> dict[int, np.ndarray]:
    by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    for (_axis, layer), direction in directions.items():
        by_layer[int(layer)].append(direction)
    aggregated: dict[int, np.ndarray] = {}
    for layer, vecs in by_layer.items():
        aggregated[layer] = np.mean(np.stack(vecs), axis=0)
    return aggregated


def _aggregate_global_components(
    selected_components: dict[str, list[tuple[str, int]]],
) -> list[tuple[str, int]]:
    ordered: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for axis in sorted(selected_components):
        for component in selected_components[axis]:
            if component in seen:
                continue
            seen.add(component)
            ordered.append(component)
    return ordered


def _build_condition_patch_maps(
    *,
    condition: str,
    axis: str | None,
    position: int,
    bundle_device: torch.device,
    directions: dict[tuple[str, int], np.ndarray],
    selected_components: dict[str, list[tuple[str, int]]],
    global_layer_directions: dict[int, np.ndarray],
    global_components: list[tuple[str, int]],
    rng: np.random.Generator | None = None,
    all_component_types: list[tuple[str, int]] | None = None,
) -> tuple[
    dict[int, Callable[[torch.Tensor], torch.Tensor]],
    dict[int, Callable[[torch.Tensor], torch.Tensor]],
    dict[int, Callable[[torch.Tensor], torch.Tensor]],
]:
    use_direction = condition in {"direction_ablation", "combined"}
    use_components = condition in {"component_ablation", "combined"}
    use_random_same_rank = condition == "random_same_rank"
    use_norm_matched_random = condition == "norm_matched_random"
    use_label_permutation = condition == "label_permutation"

    residual_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    attn_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    mlp_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}

    if use_direction:
        if axis is None:
            direction_items = sorted(global_layer_directions.items())
            for layer, direction_np in direction_items:
                idx = int(layer) - 1
                d = torch.tensor(direction_np, device=bundle_device, dtype=torch.float32)
                hook = make_direction_projection_hook(d)
                residual_patch[idx] = _compose_hooks(residual_patch.get(idx), hook)
        else:
            for (dir_axis, layer), direction_np in directions.items():
                if dir_axis != axis:
                    continue
                idx = layer - 1
                d = torch.tensor(direction_np, device=bundle_device, dtype=torch.float32)
                hook = make_direction_projection_hook(d)
                residual_patch[idx] = _compose_hooks(residual_patch.get(idx), hook)

    if use_components:
        components = global_components if axis is None else selected_components.get(axis, [])
        for component_type, layer in components:
            idx = layer - 1
            hook = make_zero_position_hook(position)
            if component_type == "attention_block":
                attn_patch[idx] = _compose_hooks(attn_patch.get(idx), hook)
            elif component_type == "mlp_block":
                mlp_patch[idx] = _compose_hooks(mlp_patch.get(idx), hook)

    if use_random_same_rank and rng is not None and all_component_types is not None:
        # Ablate random components at same layer depths as real top-k selection.
        components = global_components if axis is None else selected_components.get(axis, [])
        target_layers = sorted({layer for (_, layer) in components})
        # Pool of all component (type, layer) pairs at those layer depths.
        pool = [(ct, ly) for (ct, ly) in all_component_types if ly in target_layers]
        chosen = pool if len(pool) <= len(components) else [
            pool[i] for i in rng.choice(len(pool), size=len(components), replace=False)
        ]
        for component_type, layer in chosen:
            idx = layer - 1
            hook = make_zero_position_hook(position)
            if component_type == "attention_block":
                attn_patch[idx] = _compose_hooks(attn_patch.get(idx), hook)
            elif component_type == "mlp_block":
                mlp_patch[idx] = _compose_hooks(mlp_patch.get(idx), hook)

    if use_norm_matched_random and rng is not None:
        # Project out a random direction with the same norm as each stereotype direction.
        if axis is None:
            direction_items_list = list(global_layer_directions.items())
        else:
            direction_items_list = [
                (layer, direction_np)
                for (dir_axis, layer), direction_np in directions.items()
                if dir_axis == axis
            ]
        for layer, direction_np in direction_items_list:
            idx = int(layer) - 1
            norm = float(np.linalg.norm(direction_np))
            if norm == 0.0:
                continue
            d_model = direction_np.shape[0]
            rand_vec = rng.standard_normal(d_model).astype(np.float32)
            rand_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-8) * norm
            d = torch.tensor(rand_vec, device=bundle_device, dtype=torch.float32)
            hook = make_direction_projection_hook(d)
            residual_patch[idx] = _compose_hooks(residual_patch.get(idx), hook)

    if use_label_permutation and rng is not None:
        # Use a shuffled-label direction (computed outside and passed via directions dict
        # with the same key structure but permuted values). Handled identically to direction_ablation
        # using whatever direction_np values are present in `directions` at call time.
        if axis is None:
            direction_items_list2 = list(global_layer_directions.items())
        else:
            direction_items_list2 = [
                (layer, direction_np)
                for (dir_axis, layer), direction_np in directions.items()
                if dir_axis == axis
            ]
        for layer, direction_np in direction_items_list2:
            idx = int(layer) - 1
            d = torch.tensor(direction_np, device=bundle_device, dtype=torch.float32)
            hook = make_direction_projection_hook(d)
            residual_patch[idx] = _compose_hooks(residual_patch.get(idx), hook)

    return residual_patch, attn_patch, mlp_patch


def _build_onmanifold_residual_patch(
    pair: AlignedPair,
    bundle,
    directions: dict[tuple[str, int], np.ndarray],
    global_layer_directions: dict[int, np.ndarray],
    axis: str | None,
    max_length: int,  # kept for API compatibility
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """Build per-layer direction-projection hooks at prediction_position only.

    This is the corrected on-manifold mode. The original design (replacing trait_token_position
    with the anti-text residual) fails for completion-style datasets because:
      1. prediction_position < trait_token_position always (stereo/anti texts diverge AFTER
         the prediction point), so causal masking prevents any change at trait_token_position
         from affecting the logit at prediction_position.
      2. Stereo and anti texts have identical tokens at all positions ≤ prediction_position,
         so the anti-text hidden state there equals the stereo hidden state — no counterfactual.

    Instead, we project out the stereotype direction from prediction_position only, at each
    direction layer. This is causally valid (pred_pos is upstream of the logit) and tests
    whether the stereotype information already encoded at the prediction point is causally
    necessary — a position-specific variant of direction_ablation.
    """
    pos = pair.prediction_position

    if axis is None:
        layer_to_dir: dict[int, np.ndarray] = dict(global_layer_directions)
    else:
        layer_to_dir = {
            layer: direction_np
            for (dir_axis, layer), direction_np in directions.items()
            if dir_axis == axis
        }

    residual_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    for layer in sorted(layer_to_dir):
        idx = layer - 1  # 0-indexed block
        direction_np = layer_to_dir[layer]
        d = torch.tensor(direction_np, device=bundle.device, dtype=torch.float32)
        hook = make_direction_projection_at_position_hook(pos, d)
        residual_patch[idx] = _compose_hooks(residual_patch.get(idx), hook)

    return residual_patch


def _compute_permuted_directions(
    pairs: list[AlignedPair],
    bundle,
    directions_orig: dict[tuple[str, int], np.ndarray],
    max_length: int,
    rng: np.random.Generator,
) -> dict[tuple[str, int], np.ndarray]:
    """Recompute directions with permuted stereo/anti labels within each axis."""
    from collections import defaultdict as _dd

    # Group pairs by axis.
    by_axis: dict[str, list[AlignedPair]] = _dd(list)
    for p in pairs:
        by_axis[p.pair.axis].append(p)

    # Collect hidden states per axis per layer to recompute directions.
    # For efficiency, only recompute at layers where we already have directions.
    layer_set = sorted({layer for (_, layer) in directions_orig})

    permuted_directions: dict[tuple[str, int], np.ndarray] = {}
    for axis, axis_pairs in by_axis.items():
        if not axis_pairs:
            continue
        # Shuffle labels.
        shuffled = list(axis_pairs)
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        fake_stereo_pairs = shuffled[:mid]
        fake_anti_pairs = shuffled[mid:]
        if not fake_stereo_pairs or not fake_anti_pairs:
            continue
        for layer in layer_set:
            if (axis, layer) not in directions_orig:
                continue
            stereo_vecs: list[np.ndarray] = []
            anti_vecs: list[np.ndarray] = []
            for pair in fake_stereo_pairs[:8]:  # small sample sufficient for a control
                enc = encode_text(bundle.tokenizer, pair.pair.stereotype_text, bundle.device, max_length)
                with torch.no_grad():
                    cap = forward_with_component_capture(
                        bundle.model, enc, output_hidden_states=True,
                        capture_attention=False, capture_mlp=False,
                    )
                if cap.hidden_states and layer < len(cap.hidden_states):
                    pos = pair.trait_token_position
                    hs = cap.hidden_states[layer]
                    if pos < hs.shape[1]:
                        stereo_vecs.append(hs[0, pos, :].float().cpu().numpy())
            for pair in fake_anti_pairs[:8]:
                enc = encode_text(bundle.tokenizer, pair.pair.antistereotype_text, bundle.device, max_length)
                with torch.no_grad():
                    cap = forward_with_component_capture(
                        bundle.model, enc, output_hidden_states=True,
                        capture_attention=False, capture_mlp=False,
                    )
                if cap.hidden_states and layer < len(cap.hidden_states):
                    pos = pair.trait_token_position
                    hs = cap.hidden_states[layer]
                    if pos < hs.shape[1]:
                        anti_vecs.append(hs[0, pos, :].float().cpu().numpy())
            if stereo_vecs and anti_vecs:
                direction = np.mean(stereo_vecs, axis=0) - np.mean(anti_vecs, axis=0)
                permuted_directions[(axis, layer)] = direction.astype(np.float32)

    return permuted_directions


def _evaluate_condition(
    condition: str,
    pairs: list[AlignedPair],
    bundle,
    directions: dict[tuple[str, int], np.ndarray],
    selected_components: dict[str, list[tuple[str, int]]],
    global_layer_directions: dict[int, np.ndarray],
    global_components: list[tuple[str, int]],
    max_length: int,
    rng: np.random.Generator | None = None,
    all_component_types: list[tuple[str, int]] | None = None,
    on_manifold: bool = False,
) -> dict[str, Any]:
    margins: list[float] = []
    pair_ids: list[str] = []
    valid_pairs = 0

    for pair in pairs:
        pos = pair.prediction_position
        axis = pair.pair.axis

        if on_manifold and condition in {"direction_ablation", "combined"}:
            residual_patch = _build_onmanifold_residual_patch(
                pair=pair,
                bundle=bundle,
                directions=directions,
                global_layer_directions=global_layer_directions,
                axis=axis,
                max_length=max_length,
            )
            # Component ablation part still uses zero hooks.
            _, attn_patch, mlp_patch = _build_condition_patch_maps(
                condition="component_ablation" if condition == "combined" else "baseline",
                axis=axis,
                position=pos,
                bundle_device=bundle.device,
                directions=directions,
                selected_components=selected_components,
                global_layer_directions=global_layer_directions,
                global_components=global_components,
                rng=rng,
                all_component_types=all_component_types,
            )
        else:
            residual_patch, attn_patch, mlp_patch = _build_condition_patch_maps(
                condition=condition,
                axis=axis,
                position=pos,
                bundle_device=bundle.device,
                directions=directions,
                selected_components=selected_components,
                global_layer_directions=global_layer_directions,
                global_components=global_components,
                rng=rng,
                all_component_types=all_component_types,
            )

        encoded = encode_text(
            tokenizer=bundle.tokenizer,
            text=pair.pair.stereotype_text,
            device=bundle.device,
            max_length=max_length,
        )
        cap = forward_with_component_capture(
            model=bundle.model,
            encoded_inputs=encoded,
            output_hidden_states=False,
            capture_attention=bool(attn_patch),
            capture_mlp=bool(mlp_patch),
            attention_patch_map=attn_patch if attn_patch else None,
            mlp_patch_map=mlp_patch if mlp_patch else None,
            residual_patch_map=residual_patch if residual_patch else None,
        )
        if pos >= cap.logits.shape[1]:
            continue
        margin = compute_score_from_logits(
            cap.logits,
            position=pos,
            pos_token=pair.stereo_token,
            neg_token=pair.anti_token,
        )
        margins.append(margin)
        pair_ids.append(pair.pair.pair_id)
        valid_pairs += 1

    if not margins:
        return {
            "condition": condition,
            "n_pairs": valid_pairs,
            "stereotype_score": "",
            "mean_margin": "",
            "median_margin": "",
        }
    arr = np.array(margins, dtype=float)
    return {
        "condition": condition,
        "n_pairs": valid_pairs,
        "stereotype_score": round(float(np.mean(arr > 0)), 8),
        "mean_margin": round(float(np.mean(arr)), 8),
        "median_margin": round(float(np.median(arr)), 8),
        "_margins": margins,  # kept for bootstrap; removed before CSV write
        "_pair_ids": pair_ids,
    }


def _evaluate_corrupt_to_clean_condition(
    pairs: list[AlignedPair],
    bundle,
    directions: dict[tuple[str, int], np.ndarray],
    global_layer_directions: dict[int, np.ndarray],
    max_length: int,
) -> dict[str, Any]:
    """Symmetry check: inject stereotype direction into anti-stereotype text, measure score."""
    margins: list[float] = []
    pair_ids: list[str] = []
    valid_pairs = 0
    for pair in pairs:
        pos = pair.prediction_position
        axis = pair.pair.axis
        encoded = encode_text(
            tokenizer=bundle.tokenizer,
            text=pair.pair.antistereotype_text,
            device=bundle.device,
            max_length=max_length,
        )
        residual_patch: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
        for (dir_axis, layer), direction_np in directions.items():
            if dir_axis != axis:
                continue
            idx = layer - 1
            d = torch.tensor(direction_np, device=bundle.device, dtype=torch.float32)
            hook = make_direction_injection_hook(d, alpha=1.0)
            residual_patch[idx] = _compose_hooks(residual_patch.get(idx), hook)
        if not residual_patch:
            continue
        cap = forward_with_component_capture(
            model=bundle.model,
            encoded_inputs=encoded,
            output_hidden_states=False,
            capture_attention=False,
            capture_mlp=False,
            residual_patch_map=residual_patch,
        )
        if pos >= cap.logits.shape[1]:
            continue
        margin = compute_score_from_logits(
            cap.logits,
            position=pos,
            pos_token=pair.stereo_token,
            neg_token=pair.anti_token,
        )
        margins.append(margin)
        pair_ids.append(pair.pair.pair_id)
        valid_pairs += 1

    if not margins:
        return {
            "condition": "corrupt_to_clean",
            "n_pairs": valid_pairs,
            "stereotype_score": "",
            "mean_margin": "",
            "median_margin": "",
        }
    arr = np.array(margins, dtype=float)
    return {
        "condition": "corrupt_to_clean",
        "n_pairs": valid_pairs,
        "stereotype_score": round(float(np.mean(arr > 0)), 8),
        "mean_margin": round(float(np.mean(arr)), 8),
        "median_margin": round(float(np.median(arr)), 8),
        "_margins": margins,
        "_pair_ids": pair_ids,
    }


def _bootstrap_ci(margins: list[float], n: int, rng: np.random.Generator) -> dict[str, float]:
    if not margins or n <= 0:
        return {
            "stereotype_score_ci_lo": "",
            "stereotype_score_ci_hi": "",
            "mean_margin_ci_lo": "",
            "mean_margin_ci_hi": "",
        }
    arr = np.array(margins, dtype=float)
    scores = []
    means = []
    for _ in range(n):
        sample = rng.choice(arr, size=len(arr), replace=True)
        scores.append(float(np.mean(sample > 0)))
        means.append(float(np.mean(sample)))
    return {
        "stereotype_score_ci_lo": round(float(np.percentile(scores, 2.5)), 8),
        "stereotype_score_ci_hi": round(float(np.percentile(scores, 97.5)), 8),
        "mean_margin_ci_lo": round(float(np.percentile(means, 2.5)), 8),
        "mean_margin_ci_hi": round(float(np.percentile(means, 97.5)), 8),
    }


def _to_float_or_nan(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _token_score(logits: torch.Tensor, tokenizer, text: str) -> float:
    token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if not token_ids:
        return float("-inf")
    return float(logits[token_ids[0]].detach().cpu())


def _evaluate_bbq_accuracy_condition(
    *,
    bundle,
    condition: str,
    n_samples: int,
    max_length: int,
    directions: dict[tuple[str, int], np.ndarray],
    selected_components: dict[str, list[tuple[str, int]]],
    global_layer_directions: dict[int, np.ndarray],
    global_components: list[tuple[str, int]],
) -> float | None:
    if n_samples <= 0:
        return None
    ds = load_dataset("lighteval/bbq_helm", "all", split=f"test[:{n_samples}]")
    correct = 0
    total = 0
    for row in ds:
        prompt = f"{row['context']}\n{row['question']}\nAnswer:"
        encoded = encode_text(
            tokenizer=bundle.tokenizer,
            text=prompt,
            device=bundle.device,
            max_length=max_length,
        )
        pos = encoded["input_ids"].shape[1] - 1
        residual_patch, attn_patch, mlp_patch = _build_condition_patch_maps(
            condition=condition,
            axis=None,
            position=pos,
            bundle_device=bundle.device,
            directions=directions,
            selected_components=selected_components,
            global_layer_directions=global_layer_directions,
            global_components=global_components,
        )
        cap = forward_with_component_capture(
            model=bundle.model,
            encoded_inputs=encoded,
            output_hidden_states=False,
            capture_attention=bool(attn_patch),
            capture_mlp=bool(mlp_patch),
            attention_patch_map=attn_patch if attn_patch else None,
            mlp_patch_map=mlp_patch if mlp_patch else None,
            residual_patch_map=residual_patch if residual_patch else None,
        )
        logits = cap.logits[0, pos, :]
        choice_scores = [_token_score(logits, bundle.tokenizer, str(choice)) for choice in row["choices"]]
        pred = int(np.argmax(choice_scores))
        gold = int(row["gold_index"])
        correct += int(pred == gold)
        total += 1
    if total == 0:
        return None
    return float(correct / total)


_MMLU_LABELS = ["A", "B", "C", "D"]


def _format_mmlu_question(row: dict[str, Any], include_answer: bool) -> str:
    lines = [f"Question: {row['question']}", "Options:"]
    for idx, choice in enumerate(row["choices"]):
        label = _MMLU_LABELS[idx] if idx < len(_MMLU_LABELS) else f"Opt{idx + 1}"
        lines.append(f"{label}. {choice}")
    if include_answer:
        answer_idx = int(row["answer"])
        answer_label = _MMLU_LABELS[answer_idx] if 0 <= answer_idx < len(_MMLU_LABELS) else str(answer_idx)
        lines.append(f"Answer: {answer_label}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def _label_score(logits: torch.Tensor, tokenizer, label: str) -> float:
    scores: list[float] = []
    for prefix in (" ", ""):
        token_ids = tokenizer(f"{prefix}{label}", add_special_tokens=False)["input_ids"]
        if token_ids:
            scores.append(float(logits[token_ids[0]].detach().cpu()))
    return max(scores) if scores else float("-inf")


def _evaluate_mmlu_5shot_accuracy_condition(
    *,
    bundle,
    condition: str,
    n_samples: int,
    n_shots: int,
    max_length: int,
    directions: dict[tuple[str, int], np.ndarray],
    selected_components: dict[str, list[tuple[str, int]]],
    global_layer_directions: dict[int, np.ndarray],
    global_components: list[tuple[str, int]],
) -> float | None:
    if n_samples <= 0:
        return None
    dev = load_dataset("cais/mmlu", "all", split="dev")
    test = load_dataset("cais/mmlu", "all", split=f"test[:{n_samples}]")
    dev_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev:
        dev_by_subject[str(row["subject"])].append(row)

    correct = 0
    total = 0
    for row in test:
        subject = str(row["subject"])
        shots = dev_by_subject.get(subject, [])[: max(0, n_shots)]
        prompt_blocks = [
            f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}."
        ]
        for shot in shots:
            prompt_blocks.append(_format_mmlu_question(shot, include_answer=True))
        prompt_blocks.append(_format_mmlu_question(row, include_answer=False))
        prompt = "\n\n".join(prompt_blocks)

        encoded = encode_text(
            tokenizer=bundle.tokenizer,
            text=prompt,
            device=bundle.device,
            max_length=max_length,
        )
        pos = encoded["input_ids"].shape[1] - 1
        residual_patch, attn_patch, mlp_patch = _build_condition_patch_maps(
            condition=condition,
            axis=None,
            position=pos,
            bundle_device=bundle.device,
            directions=directions,
            selected_components=selected_components,
            global_layer_directions=global_layer_directions,
            global_components=global_components,
        )
        cap = forward_with_component_capture(
            model=bundle.model,
            encoded_inputs=encoded,
            output_hidden_states=False,
            capture_attention=bool(attn_patch),
            capture_mlp=bool(mlp_patch),
            attention_patch_map=attn_patch if attn_patch else None,
            mlp_patch_map=mlp_patch if mlp_patch else None,
            residual_patch_map=residual_patch if residual_patch else None,
        )
        logits = cap.logits[0, pos, :]
        option_count = min(len(row["choices"]), len(_MMLU_LABELS))
        label_scores = [_label_score(logits, bundle.tokenizer, _MMLU_LABELS[i]) for i in range(option_count)]
        pred_idx = int(np.argmax(label_scores))
        gold_idx = int(row["answer"])
        correct += int(pred_idx == gold_idx)
        total += 1

    if total == 0:
        return None
    return float(correct / total)


def main() -> None:
    args = parse_args()
    ctx = start_run("04", parameters=vars(args), project_root=PROJECT_ROOT)
    try:
        exp1_dir = (
            Path(args.exp1_run_dir)
            if args.exp1_run_dir
            else _latest_run_dir(
                "01_layerwise_probing",
                required_relpaths=[
                    "artifacts/aligned_pairs.jsonl",
                    "artifacts/train_test_split.json",
                    "artifacts/directions_layerwise.npz",
                ],
                model_name=args.model,
            )
        )
        exp3_dir = (
            Path(args.exp3_run_dir)
            if args.exp3_run_dir
            else _latest_run_dir(
                "03_attribution_patching",
                required_relpaths=["tables/attribution_patch_scores.csv"],
                model_name=args.model,
            )
        )
        aligned_pairs = _load_aligned_pairs(exp1_dir / "artifacts" / "aligned_pairs.jsonl")
        split_info = json.loads((exp1_dir / "artifacts" / "train_test_split.json").read_text(encoding="utf-8"))
        test_indices = split_info.get("test_indices", [])
        heldout = [aligned_pairs[i] for i in test_indices if 0 <= i < len(aligned_pairs)]
        if args.heldout_pairs > 0:
            heldout = heldout[: args.heldout_pairs]

        directions = load_directions_npz(exp1_dir / "artifacts" / "directions_layerwise.npz")
        selected_components = _load_selected_components(exp3_dir, top_k=args.top_k_components)

        refs_path = ctx.artifacts_dir / "dependencies.json"
        write_json(
            refs_path,
            {
                "exp1_run_dir": str(exp1_dir),
                "exp3_run_dir": str(exp3_dir),
                "heldout_pairs": len(heldout),
                "directions_loaded": len(directions),
                "axes_with_selected_components": len(selected_components),
                "strict_controls": args.strict_controls,
                "include_label_permutation_control": args.include_label_permutation_control,
                "bootstrap_n": args.bootstrap_n,
                "on_manifold": args.on_manifold,
            },
        )
        ctx.register_artifact(refs_path, artifact_type="artifact", description="Dependency references.")

        if args.dry_run:
            metrics = {
                "heldout_pairs_loaded": len(heldout),
                "directions_loaded": len(directions),
                "axes_with_selected_components": len(selected_components),
                "dry_run": True,
            }
            complete_run(ctx, metrics=metrics)
            return

        bundle = load_model_bundle(model_name=args.model, device=args.device, torch_dtype=args.torch_dtype)
        global_layer_directions = _aggregate_global_directions(directions)
        global_components = _aggregate_global_components(selected_components)

        rng = np.random.default_rng(args.seed)

        # Build list of all possible (component_type, layer) combos for random-rank control.
        all_component_types: list[tuple[str, int]] = []
        if args.strict_controls:
            seen_ct: set[tuple[str, int]] = set()
            for comps in selected_components.values():
                for ct in comps:
                    if ct not in seen_ct:
                        seen_ct.add(ct)
                        all_component_types.append(ct)

        # Precompute permuted directions for label_permutation control.
        permuted_directions: dict[tuple[str, int], np.ndarray] = {}
        permuted_global_layer_directions: dict[int, np.ndarray] = {}
        if args.strict_controls and args.include_label_permutation_control:
            permuted_directions = _compute_permuted_directions(
                pairs=heldout,
                bundle=bundle,
                directions_orig=directions,
                max_length=args.max_length,
                rng=rng,
            )
            permuted_global_layer_directions = _aggregate_global_directions(permuted_directions)

        conditions = ["baseline", "direction_ablation", "component_ablation", "combined"]
        if args.strict_controls:
            conditions += ["random_same_rank", "norm_matched_random", "corrupt_to_clean"]
            if args.include_label_permutation_control:
                conditions += ["label_permutation"]

        rows: list[dict[str, Any]] = []
        baseline_pair_to_margin: dict[str, float] = {}
        for condition in conditions:
            if condition == "corrupt_to_clean":
                row = _evaluate_corrupt_to_clean_condition(
                    pairs=heldout,
                    bundle=bundle,
                    directions=directions,
                    global_layer_directions=global_layer_directions,
                    max_length=args.max_length,
                )
            elif condition == "label_permutation":
                row = _evaluate_condition(
                    condition="label_permutation",
                    pairs=heldout,
                    bundle=bundle,
                    directions=permuted_directions,
                    selected_components=selected_components,
                    global_layer_directions=permuted_global_layer_directions,
                    global_components=global_components,
                    max_length=args.max_length,
                    rng=rng,
                    all_component_types=all_component_types,
                    on_manifold=False,
                )
            else:
                row = _evaluate_condition(
                    condition=condition,
                    pairs=heldout,
                    bundle=bundle,
                    directions=directions,
                    selected_components=selected_components,
                    global_layer_directions=global_layer_directions,
                    global_components=global_components,
                    max_length=args.max_length,
                    rng=rng,
                    all_component_types=all_component_types,
                    on_manifold=args.on_manifold,
                )

            # When on-manifold is active, direction_ablation/combined operated at prediction_position
            # only (position-specific projection), which answers a different question than the
            # standard full-sequence direction ablation. Rename the condition so downstream
            # analysis can distinguish them without reading the manifest.
            if args.on_manifold and row.get("condition") in {"direction_ablation", "combined"}:
                row = dict(row)
                row["condition"] = row["condition"].replace(
                    "direction_ablation", "direction_ablation_at_pred_pos"
                ).replace("combined", "combined_at_pred_pos")

            # Bootstrap CIs on absolute condition metrics.
            margins_for_ci = row.pop("_margins", [])
            pair_ids_for_ci = row.pop("_pair_ids", [])
            if args.bootstrap_n > 0:
                ci = _bootstrap_ci(margins_for_ci, n=args.bootstrap_n, rng=rng)
            else:
                ci = {
                    "stereotype_score_ci_lo": "",
                    "stereotype_score_ci_hi": "",
                    "mean_margin_ci_lo": "",
                    "mean_margin_ci_hi": "",
            }
            row.update(ci)

            # Paired deltas and inferential tests vs baseline.
            row["stereotype_score_delta"] = ""
            row["stereotype_score_delta_ci_low"] = ""
            row["stereotype_score_delta_ci_high"] = ""
            row["mean_margin_delta"] = ""
            row["mean_margin_delta_ci_low"] = ""
            row["mean_margin_delta_ci_high"] = ""
            row["paired_p_score_sign"] = ""
            row["paired_p_margin_wilcoxon"] = ""
            row["q_score_sign"] = ""
            row["q_margin_wilcoxon"] = ""

            if row.get("condition") == "baseline":
                baseline_pair_to_margin = {
                    pid: float(m)
                    for pid, m in zip(pair_ids_for_ci, margins_for_ci, strict=False)
                }
                row["stereotype_score_delta"] = 0.0
                row["mean_margin_delta"] = 0.0
            elif baseline_pair_to_margin and margins_for_ci and pair_ids_for_ci:
                cond_map = {
                    pid: float(m)
                    for pid, m in zip(pair_ids_for_ci, margins_for_ci, strict=False)
                    if pid in baseline_pair_to_margin
                }
                common_ids = [pid for pid in pair_ids_for_ci if pid in cond_map and pid in baseline_pair_to_margin]
                if common_ids:
                    arr_base = np.array([baseline_pair_to_margin[pid] for pid in common_ids], dtype=float)
                    arr_cond = np.array([cond_map[pid] for pid in common_ids], dtype=float)
                    score_pair_diffs = (arr_cond > 0).astype(float) - (arr_base > 0).astype(float)
                    margin_pair_diffs = arr_cond - arr_base
                    row["stereotype_score_delta"] = round(float(np.mean(score_pair_diffs)), 8)
                    row["mean_margin_delta"] = round(float(np.mean(margin_pair_diffs)), 8)
                    score_ci = bootstrap_mean_ci(
                        score_pair_diffs,
                        n_resamples=args.bootstrap_n,
                        rng=rng,
                    )
                    margin_ci = bootstrap_mean_ci(
                        margin_pair_diffs,
                        n_resamples=args.bootstrap_n,
                        rng=rng,
                    )
                    row["stereotype_score_delta_ci_low"] = round(float(score_ci.ci_low), 8)
                    row["stereotype_score_delta_ci_high"] = round(float(score_ci.ci_high), 8)
                    row["mean_margin_delta_ci_low"] = round(float(margin_ci.ci_low), 8)
                    row["mean_margin_delta_ci_high"] = round(float(margin_ci.ci_high), 8)
                    p_score_sign, _, _ = paired_sign_test(score_pair_diffs)
                    p_margin_wilcoxon, _ = wilcoxon_signed_rank_safe(margin_pair_diffs)
                    row["paired_p_score_sign"] = (
                        round(float(p_score_sign), 8) if np.isfinite(p_score_sign) else ""
                    )
                    row["paired_p_margin_wilcoxon"] = (
                        round(float(p_margin_wilcoxon), 8) if np.isfinite(p_margin_wilcoxon) else ""
                    )

            bbq_accuracy = _evaluate_bbq_accuracy_condition(
                bundle=bundle,
                condition=condition if condition not in {"random_same_rank", "norm_matched_random", "label_permutation", "corrupt_to_clean"} else "baseline",
                n_samples=args.bbq_samples,
                max_length=args.max_length,
                directions=directions,
                selected_components=selected_components,
                global_layer_directions=global_layer_directions,
                global_components=global_components,
            )
            mmlu_accuracy = _evaluate_mmlu_5shot_accuracy_condition(
                bundle=bundle,
                condition=condition if condition not in {"random_same_rank", "norm_matched_random", "label_permutation", "corrupt_to_clean"} else "baseline",
                n_samples=args.mmlu_samples,
                n_shots=args.mmlu_shots,
                max_length=args.max_length,
                directions=directions,
                selected_components=selected_components,
                global_layer_directions=global_layer_directions,
                global_components=global_components,
            )
            row["bbq_accuracy"] = round(float(bbq_accuracy), 8) if bbq_accuracy is not None else ""
            row["mmlu_5shot_accuracy"] = round(float(mmlu_accuracy), 8) if mmlu_accuracy is not None else ""
            rows.append(row)

        # BH-FDR correction across non-baseline conditions for paired tests.
        score_ps = [_to_float_or_nan(r.get("paired_p_score_sign", "")) for r in rows if r.get("condition") != "baseline"]
        score_qs = benjamini_hochberg(score_ps)
        margin_ps = [
            _to_float_or_nan(r.get("paired_p_margin_wilcoxon", ""))
            for r in rows
            if r.get("condition") != "baseline"
        ]
        margin_qs = benjamini_hochberg(margin_ps)
        idx = 0
        for row in rows:
            if row.get("condition") == "baseline":
                continue
            q_score = score_qs[idx] if idx < len(score_qs) else float("nan")
            q_margin = margin_qs[idx] if idx < len(margin_qs) else float("nan")
            row["q_score_sign"] = round(float(q_score), 8) if np.isfinite(q_score) else ""
            row["q_margin_wilcoxon"] = round(float(q_margin), 8) if np.isfinite(q_margin) else ""
            idx += 1

        out_path = ctx.tables_dir / "ablation_comparison.csv"
        write_csv(
            out_path,
            rows=rows,
            fieldnames=[
                "condition",
                "n_pairs",
                "stereotype_score",
                "mean_margin",
                "median_margin",
                "stereotype_score_ci_lo",
                "stereotype_score_ci_hi",
                "mean_margin_ci_lo",
                "mean_margin_ci_hi",
                "stereotype_score_delta",
                "stereotype_score_delta_ci_low",
                "stereotype_score_delta_ci_high",
                "mean_margin_delta",
                "mean_margin_delta_ci_low",
                "mean_margin_delta_ci_high",
                "paired_p_score_sign",
                "paired_p_margin_wilcoxon",
                "q_score_sign",
                "q_margin_wilcoxon",
                "bbq_accuracy",
                "mmlu_5shot_accuracy",
            ],
        )
        ctx.register_artifact(out_path, artifact_type="table", description="Ablation comparison table.")

        metrics = {
            "heldout_pairs_evaluated": int(rows[0]["n_pairs"]) if rows else 0,
            "conditions_compared": len(rows),
            "directions_loaded": len(directions),
            "axes_with_selected_components": len(selected_components),
            "bbq_samples": args.bbq_samples,
            "mmlu_samples": args.mmlu_samples,
            "strict_controls": args.strict_controls,
            "bootstrap_n": args.bootstrap_n,
            "on_manifold": args.on_manifold,
            "dry_run": False,
        }
        complete_run(ctx, metrics=metrics)
    except Exception as exc:
        fail_run(ctx, exc)
        raise


if __name__ == "__main__":
    main()
