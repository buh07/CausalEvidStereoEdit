from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

from stereacl.constants import CROWS_AXIS_MAP, STEREOSET_AXIS_MAP


CROWS_RAW_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
SEEGULL_GLOBAL_V2_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/seegull/main/dataset/"
    "stereotypes_global_v2.csv"
)
SEEGULL_US_STATES_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/seegull/main/dataset/"
    "stereotypes_us_states.csv"
)
SEEGULL_INDIAN_STATES_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/seegull/main/dataset/"
    "stereotypes_indian_states.csv"
)


@dataclass(frozen=True)
class ContrastPair:
    pair_id: str
    source: str
    axis: str
    stereotype_text: str
    antistereotype_text: str
    metadata: dict[str, Any]


def _safe_axis(axis: str | None, fallback: str = "other") -> str:
    if not axis:
        return fallback
    cleaned = str(axis).strip().lower().replace(" ", "_")
    return cleaned or fallback


def _serialize_pairs(pairs: list[ContrastPair]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        row = asdict(pair)
        row["metadata"] = json.dumps(pair.metadata, sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


def save_pairs_jsonl(pairs: list[ContrastPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            payload = asdict(pair)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_pairs_jsonl(path: Path) -> list[ContrastPair]:
    pairs: list[ContrastPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pairs.append(
                ContrastPair(
                    pair_id=row["pair_id"],
                    source=row["source"],
                    axis=row["axis"],
                    stereotype_text=row["stereotype_text"],
                    antistereotype_text=row["antistereotype_text"],
                    metadata=row.get("metadata", {}),
                )
            )
    return pairs


def load_stereoset_intrasentence_pairs(limit: int | None = None) -> list[ContrastPair]:
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    pairs: list[ContrastPair] = []
    for row in ds:
        labels = row["sentences"]["gold_label"]
        sents = row["sentences"]["sentence"]
        if not labels or not sents:
            continue
        try:
            stereo_idx = labels.index(0)
            anti_idx = labels.index(1)
        except ValueError:
            continue
        bias_type = _safe_axis(row.get("bias_type"))
        axis = STEREOSET_AXIS_MAP.get(bias_type, bias_type)
        stereo_text = str(sents[stereo_idx]).strip()
        anti_text = str(sents[anti_idx]).strip()
        if not stereo_text or not anti_text or stereo_text == anti_text:
            continue
        pairs.append(
            ContrastPair(
                pair_id=f"stereoset_{row['id']}",
                source="stereoset_intrasentence",
                axis=axis,
                stereotype_text=stereo_text,
                antistereotype_text=anti_text,
                metadata={
                    "target": row.get("target", ""),
                    "context": row.get("context", ""),
                    "bias_type": row.get("bias_type", ""),
                },
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def load_crows_pairs(limit: int | None = None, raw_url: str = CROWS_RAW_URL) -> list[ContrastPair]:
    df = pd.read_csv(raw_url)
    pairs: list[ContrastPair] = []
    for _, row in df.iterrows():
        direction = str(row.get("stereo_antistereo", "stereo")).strip().lower()
        sent_more = str(row.get("sent_more", "")).strip()
        sent_less = str(row.get("sent_less", "")).strip()
        if not sent_more or not sent_less or sent_more == sent_less:
            continue
        if direction == "stereo":
            stereo_text = sent_more
            anti_text = sent_less
        else:
            stereo_text = sent_less
            anti_text = sent_more
        raw_axis = _safe_axis(row.get("bias_type"))
        axis = CROWS_AXIS_MAP.get(raw_axis, raw_axis)
        pair_id = f"crows_{int(row.get('Unnamed: 0', len(pairs)))}"
        pairs.append(
            ContrastPair(
                pair_id=pair_id,
                source="crows_pairs",
                axis=axis,
                stereotype_text=stereo_text,
                antistereotype_text=anti_text,
                metadata={
                    "bias_type": row.get("bias_type", ""),
                    "direction": direction,
                    "anon_writer": row.get("anon_writer", ""),
                },
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def load_seegull_pairs_from_url(
    raw_url: str,
    source_name: str,
    axis: str,
    identity_prefix: str = "People from",
    limit: int | None = None,
    min_abs_delta: int = 1,
    include_identities: set[str] | None = None,
    pairs_per_identity: int = 1,
) -> list[ContrastPair]:
    df = pd.read_csv(raw_url)
    for col in ("region_stereo", "region_nonstereo"):
        if col not in df.columns:
            raise ValueError(f"SeeGULL file is missing expected column: {col}")
    df = df.copy()
    df["delta"] = df["region_stereo"] - df["region_nonstereo"]
    pairs: list[ContrastPair] = []

    if pairs_per_identity < 1:
        raise ValueError("pairs_per_identity must be >= 1")

    for identity, group in df.groupby("identity"):
        identity_text = str(identity).strip()
        if include_identities is not None and identity_text not in include_identities:
            continue

        pos = group[group["delta"] >= min_abs_delta].sort_values("delta", ascending=False)
        neg = group[group["delta"] <= -min_abs_delta].sort_values("delta", ascending=True)
        if pos.empty or neg.empty:
            continue

        n_pairs = min(pairs_per_identity, len(pos), len(neg))
        for rank in range(n_pairs):
            top_pos = pos.iloc[rank]
            top_neg = neg.iloc[rank]
            stereo_attr = str(top_pos["attribute"]).strip()
            anti_attr = str(top_neg["attribute"]).strip()
            if not stereo_attr or not anti_attr or not identity_text:
                continue
            if stereo_attr == anti_attr:
                continue

            stereo_text = f"{identity_prefix} {identity_text} are {stereo_attr}."
            anti_text = f"{identity_prefix} {identity_text} are {anti_attr}."
            pairs.append(
                ContrastPair(
                    pair_id=f"{source_name}_{identity_text.lower().replace(' ', '_')}_{rank}",
                    source=source_name,
                    axis=axis,
                    stereotype_text=stereo_text,
                    antistereotype_text=anti_text,
                    metadata={
                        "identity": identity_text,
                        "pair_rank_within_identity": rank,
                        "stereo_attribute": stereo_attr,
                        "anti_attribute": anti_attr,
                        "stereo_delta": float(top_pos["delta"]),
                        "anti_delta": float(top_neg["delta"]),
                        "mean_offensiveness": float(
                            top_pos.get("mean offensiveness_score", top_pos.get("offensiveness_score", 0.0))
                        ),
                    },
                )
            )
            if limit is not None and len(pairs) >= limit:
                break
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def load_seegull_pairs(
    limit: int | None = None,
    raw_url: str = SEEGULL_GLOBAL_V2_URL,
    min_abs_delta: int = 1,
    include_identities: set[str] | None = None,
    pairs_per_identity: int = 1,
) -> list[ContrastPair]:
    return load_seegull_pairs_from_url(
        raw_url=raw_url,
        source_name="seegull_global_v2",
        axis="nationality",
        identity_prefix="People from",
        limit=limit,
        min_abs_delta=min_abs_delta,
        include_identities=include_identities,
        pairs_per_identity=pairs_per_identity,
    )


def load_seegull_us_state_pairs(
    limit: int | None = None,
    raw_url: str = SEEGULL_US_STATES_URL,
    min_abs_delta: int = 1,
    pairs_per_identity: int = 1,
) -> list[ContrastPair]:
    return load_seegull_pairs_from_url(
        raw_url=raw_url,
        source_name="seegull_us_states",
        axis="regional_us",
        identity_prefix="People from",
        limit=limit,
        min_abs_delta=min_abs_delta,
        pairs_per_identity=pairs_per_identity,
    )


def load_seegull_indian_state_pairs(
    limit: int | None = None,
    raw_url: str = SEEGULL_INDIAN_STATES_URL,
    min_abs_delta: int = 1,
    pairs_per_identity: int = 1,
) -> list[ContrastPair]:
    return load_seegull_pairs_from_url(
        raw_url=raw_url,
        source_name="seegull_indian_states",
        axis="regional_india",
        identity_prefix="People from",
        limit=limit,
        min_abs_delta=min_abs_delta,
        pairs_per_identity=pairs_per_identity,
    )


def build_contrast_pairs(
    include_stereoset: bool = True,
    include_crows: bool = True,
    include_seegull: bool = True,
    per_source_limit: int | None = None,
) -> list[ContrastPair]:
    pairs: list[ContrastPair] = []
    if include_stereoset:
        pairs.extend(load_stereoset_intrasentence_pairs(limit=per_source_limit))
    if include_crows:
        pairs.extend(load_crows_pairs(limit=per_source_limit))
    if include_seegull:
        pairs.extend(load_seegull_pairs(limit=per_source_limit))
    return pairs


def summarize_pairs(pairs: list[ContrastPair]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["source", "axis", "count"])
    df = _serialize_pairs(pairs)
    grouped = (
        df.groupby(["source", "axis"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["source", "axis"])
    )
    return grouped


def deterministic_split_indices(
    n_items: int,
    test_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if n_items <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_items, dtype=int)
    rng.shuffle(all_idx)
    n_test = max(1, int(round(n_items * test_fraction))) if n_items > 1 else 0
    n_test = min(n_test, n_items - 1) if n_items > 1 else 0
    test_idx = np.sort(all_idx[:n_test])
    train_idx = np.sort(all_idx[n_test:])
    return train_idx, test_idx
