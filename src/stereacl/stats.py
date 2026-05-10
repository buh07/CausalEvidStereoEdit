from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon


@dataclass(frozen=True)
class Interval:
    mean: float
    ci_low: float
    ci_high: float


def bootstrap_mean_ci(
    values: Iterable[float] | np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Interval:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return Interval(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    if n_resamples <= 0:
        m = float(np.mean(arr))
        return Interval(mean=m, ci_low=m, ci_high=m)
    generator = rng or np.random.default_rng(0)
    n = arr.size
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = arr[generator.integers(0, n, size=n)]
        draws[i] = float(np.mean(sample))
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - (alpha / 2)))
    return Interval(mean=float(np.mean(arr)), ci_low=lo, ci_high=hi)


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    z2 = z * z
    denom = 1 + (z2 / total)
    centre = (p + (z2 / (2 * total))) / denom
    margin = (z / denom) * np.sqrt((p * (1 - p) / total) + (z2 / (4 * total * total)))
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def paired_sign_test(differences: Iterable[float] | np.ndarray) -> tuple[float, int, int]:
    arr = np.asarray(list(differences), dtype=float)
    nonzero = arr[np.abs(arr) > 1e-12]
    n = int(nonzero.size)
    if n == 0:
        return float("nan"), 0, 0
    n_pos = int(np.sum(nonzero > 0))
    p = float(binomtest(k=n_pos, n=n, p=0.5, alternative="two-sided").pvalue)
    return p, n, n_pos


def wilcoxon_signed_rank_safe(differences: Iterable[float] | np.ndarray) -> tuple[float, int]:
    arr = np.asarray(list(differences), dtype=float)
    nonzero = arr[np.abs(arr) > 1e-12]
    n = int(nonzero.size)
    if n < 3:
        return float("nan"), n
    try:
        p = float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
        return p, n
    except Exception:
        return float("nan"), n


def spearman_safe(x: Iterable[float] | np.ndarray, y: Iterable[float] | np.ndarray) -> tuple[float, float, int]:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    n = int(xa.size)
    if n < 3:
        return float("nan"), float("nan"), n
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return float("nan"), float("nan"), n
    res = spearmanr(xa, ya)
    rho = float(res.correlation) if res.correlation is not None else float("nan")
    p = float(res.pvalue) if res.pvalue is not None else float("nan")
    return rho, p, n


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if np.isfinite(p)]
    if not indexed:
        return [float("nan")] * m
    indexed.sort(key=lambda x: x[1])
    q_vals = [float("nan")] * m
    running = 1.0
    k = len(indexed)
    for rank_rev, (orig_idx, p) in enumerate(reversed(indexed), start=1):
        rank = k - rank_rev + 1
        q = min(running, (p * k) / rank)
        running = q
        q_vals[orig_idx] = float(min(max(q, 0.0), 1.0))
    return q_vals

