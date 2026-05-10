from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from stereacl.data import ContrastPair


@dataclass(frozen=True)
class AlignedPair:
    pair: ContrastPair
    stereo_input_ids: list[int]
    anti_input_ids: list[int]
    stereo_token: int
    anti_token: int
    trait_token_position: int
    prediction_position: int
    differing_span_stereo: tuple[int, int]
    differing_span_anti: tuple[int, int]


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


def align_pair_to_single_token_difference(
    pair: ContrastPair,
    tokenizer: PreTrainedTokenizerBase,
) -> AlignedPair | None:
    stereo_ids = tokenizer(
        pair.stereotype_text,
        add_special_tokens=True,
        return_attention_mask=False,
    )["input_ids"]
    anti_ids = tokenizer(
        pair.antistereotype_text,
        add_special_tokens=True,
        return_attention_mask=False,
    )["input_ids"]
    diff = _find_diff_span(stereo_ids, anti_ids)
    if diff is None:
        return None
    start_s, end_s, start_a, end_a = diff
    span_s = stereo_ids[start_s : end_s + 1]
    span_a = anti_ids[start_a : end_a + 1]
    if len(span_s) != 1 or len(span_a) != 1:
        return None
    if start_s <= 0 or start_a <= 0:
        return None
    if start_s != start_a:
        return None
    return AlignedPair(
        pair=pair,
        stereo_input_ids=stereo_ids,
        anti_input_ids=anti_ids,
        stereo_token=span_s[0],
        anti_token=span_a[0],
        trait_token_position=start_s,
        prediction_position=start_s - 1,
        differing_span_stereo=(start_s, end_s),
        differing_span_anti=(start_a, end_a),
    )


def filter_aligned_pairs(
    pairs: list[ContrastPair],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[AlignedPair], dict[str, Any]]:
    kept: list[AlignedPair] = []
    dropped_non_single = 0
    dropped_alignment = 0
    dropped_same_tokens = 0
    for pair in pairs:
        aligned = align_pair_to_single_token_difference(pair, tokenizer=tokenizer)
        if aligned is None:
            dropped_non_single += 1
            continue
        if aligned.pair.stereotype_text == aligned.pair.antistereotype_text:
            dropped_same_tokens += 1
            continue
        if aligned.prediction_position < 0:
            dropped_alignment += 1
            continue
        kept.append(aligned)
    stats = {
        "input_pairs": len(pairs),
        "kept_pairs": len(kept),
        "dropped_non_single_token_diff": dropped_non_single,
        "dropped_alignment": dropped_alignment,
        "dropped_same_text": dropped_same_tokens,
    }
    return kept, stats

