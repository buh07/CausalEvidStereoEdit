from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_direction(stereo_vectors: np.ndarray, anti_vectors: np.ndarray) -> np.ndarray:
    stereo_mean = stereo_vectors.mean(axis=0)
    anti_mean = anti_vectors.mean(axis=0)
    return stereo_mean - anti_mean


def fit_logistic_probe_auc(
    stereo_vectors: np.ndarray,
    anti_vectors: np.ndarray,
    seed: int = 7,
) -> float | None:
    if stereo_vectors.size == 0 or anti_vectors.size == 0:
        return None
    x = np.concatenate([stereo_vectors, anti_vectors], axis=0)
    y = np.concatenate(
        [np.ones(len(stereo_vectors), dtype=int), np.zeros(len(anti_vectors), dtype=int)],
        axis=0,
    )
    if len(np.unique(y)) < 2 or len(y) < 8:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = LogisticRegression(
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


@dataclass
class LayerAxisBuckets:
    stereo: dict[str, dict[int, list[np.ndarray]]]
    anti: dict[str, dict[int, list[np.ndarray]]]
    source_stereo: dict[str, dict[str, dict[int, list[np.ndarray]]]]
    source_anti: dict[str, dict[str, dict[int, list[np.ndarray]]]]

    @staticmethod
    def empty() -> "LayerAxisBuckets":
        def layer_dict() -> dict[int, list[np.ndarray]]:
            return defaultdict(list)  # type: ignore[return-value]

        stereo = defaultdict(layer_dict)  # type: ignore[arg-type]
        anti = defaultdict(layer_dict)  # type: ignore[arg-type]
        source_stereo = defaultdict(lambda: defaultdict(layer_dict))  # type: ignore[arg-type]
        source_anti = defaultdict(lambda: defaultdict(layer_dict))  # type: ignore[arg-type]
        return LayerAxisBuckets(
            stereo=stereo,
            anti=anti,
            source_stereo=source_stereo,
            source_anti=source_anti,
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_directions_npz(path: Path, directions: dict[tuple[str, int], np.ndarray]) -> None:
    arrays = {f"{axis}__L{layer:03d}": vec for (axis, layer), vec in directions.items()}
    np.savez(path, **arrays)


def load_directions_npz(path: Path) -> dict[tuple[str, int], np.ndarray]:
    loaded = np.load(path)
    directions: dict[tuple[str, int], np.ndarray] = {}
    for key in loaded.files:
        axis, layer_text = key.split("__L")
        directions[(axis, int(layer_text))] = loaded[key]
    return directions


def compute_score_from_logits(logits: torch.Tensor, position: int, pos_token: int, neg_token: int) -> float:
    vec = logits[0, position, :]
    return float((vec[pos_token] - vec[neg_token]).detach().cpu())

