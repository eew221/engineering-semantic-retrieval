"""Engineering-semantic similarity definitions and retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SampleSemantics:
    damage_class: str
    component_class: str
    severity_score: float


def engineering_similarity(
    query: SampleSemantics,
    candidate: SampleSemantics,
    same_damage_bonus: float = 0.4,
    same_component_bonus: float = 0.2,
    severity_bonus_scale: float = 0.4,
) -> float:
    score = 0.0
    if query.damage_class == candidate.damage_class:
        score += same_damage_bonus
    if query.component_class == candidate.component_class:
        score += same_component_bonus
    severity_gap = abs(query.severity_score - candidate.severity_score)
    score += max(0.0, 1.0 - severity_gap) * severity_bonus_scale
    return score


def pair_target_weight(
    query: SampleSemantics,
    candidate: SampleSemantics,
    severity_tolerance: float = 0.15,
    same_damage_component_close_weight: float = 1.0,
    same_damage_component_far_weight: float = 0.85,
    same_damage_only_weight: float = 0.55,
    same_component_only_weight: float = 0.25,
    different_weight: float = 0.0,
) -> float:
    same_damage = query.damage_class == candidate.damage_class
    same_component = query.component_class == candidate.component_class
    close_severity = abs(query.severity_score - candidate.severity_score) <= severity_tolerance

    if same_damage and same_component and close_severity:
        return same_damage_component_close_weight
    if same_damage and same_component:
        return same_damage_component_far_weight
    if same_damage:
        return same_damage_only_weight
    if same_component:
        return same_component_only_weight
    return different_weight


def component_consistency_at_k(
    query_components: np.ndarray,
    retrieved_components: np.ndarray,
    k: int,
) -> float:
    topk = retrieved_components[:, :k]
    return float((topk == query_components[:, None]).mean())


def severity_consistency_at_k(
    query_severity: np.ndarray,
    retrieved_severity: np.ndarray,
    k: int,
    tolerance: float,
) -> float:
    topk = retrieved_severity[:, :k]
    delta = np.abs(topk - query_severity[:, None])
    return float((delta <= tolerance).mean())


def dcg(relevance: Iterable[float]) -> float:
    relevance = np.asarray(list(relevance), dtype=np.float32)
    if relevance.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum((2**relevance - 1) / denom))


def ndcg_at_k(relevance_matrix: np.ndarray, k: int) -> float:
    values = []
    for row in relevance_matrix:
        rel = row[:k]
        ideal = np.sort(row)[::-1][:k]
        ideal_dcg = dcg(ideal)
        values.append(0.0 if ideal_dcg == 0 else dcg(rel) / ideal_dcg)
    return float(np.mean(values))
