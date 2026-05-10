from __future__ import annotations

from collections import defaultdict
import random

from stereacl.token_alignment import AlignedPair


def stratified_axis_sample(
    aligned_pairs: list[AlignedPair],
    limit: int,
    seed: int = 7,
) -> list[AlignedPair]:
    if limit <= 0 or len(aligned_pairs) <= limit:
        return aligned_pairs
    rng = random.Random(seed)
    by_axis: dict[str, list[AlignedPair]] = defaultdict(list)
    for pair in aligned_pairs:
        by_axis[pair.pair.axis].append(pair)

    axes = sorted(by_axis)
    sampled: list[AlignedPair] = []
    per_axis = max(1, limit // max(1, len(axes)))
    for axis in axes:
        group = by_axis[axis]
        rng.shuffle(group)
        sampled.extend(group[:per_axis])

    if len(sampled) < limit:
        remaining = [p for p in aligned_pairs if p not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[: limit - len(sampled)])
    else:
        sampled = sampled[:limit]
    return sampled

