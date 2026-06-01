"""Retrieval evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from .engineering_semantics import SampleSemantics, component_consistency_at_k, engineering_similarity, ndcg_at_k, severity_consistency_at_k


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    sim = normalized @ normalized.T
    np.fill_diagonal(sim, -np.inf)
    return sim


def rank_indices(embeddings: np.ndarray) -> np.ndarray:
    sim = similarity_matrix(embeddings)
    return np.argsort(-sim, axis=1)


def build_relevance_matrix(
    metadata: list[dict[str, Any]],
    ranked_indices: np.ndarray,
    same_damage_bonus: float,
    same_component_bonus: float,
    severity_bonus_scale: float,
) -> np.ndarray:
    rel = np.zeros_like(ranked_indices, dtype=np.float32)
    for i, ordering in enumerate(ranked_indices):
        query = metadata[i]
        for j, idx in enumerate(ordering):
            candidate = metadata[idx]
            rel[i, j] = engineering_similarity(
                query=query,
                candidate=candidate,
                same_damage_bonus=same_damage_bonus,
                same_component_bonus=same_component_bonus,
                severity_bonus_scale=severity_bonus_scale,
            )
    return rel


def retrieval_metrics_from_ranks(
    ranked: np.ndarray,
    damage_labels: list[str],
    component_labels: list[str],
    severity_scores: np.ndarray,
    topk: list[int],
    severity_tolerance: float,
    same_damage_bonus: float,
    same_component_bonus: float,
    severity_bonus_scale: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    damage_labels_np = np.asarray(damage_labels)
    component_labels_np = np.asarray(component_labels)
    metadata = [
        SampleSemantics(
            damage_class=damage_labels[i],
            component_class=component_labels[i],
            severity_score=float(severity_scores[i]),
        )
        for i in range(len(damage_labels))
    ]

    relevance = build_relevance_matrix(
        metadata=metadata,
        ranked_indices=ranked,
        same_damage_bonus=same_damage_bonus,
        same_component_bonus=same_component_bonus,
        severity_bonus_scale=severity_bonus_scale,
    )

    retrieved_components = component_labels_np[ranked]
    retrieved_severity = severity_scores[ranked]

    binary_targets = (damage_labels_np[ranked] == damage_labels_np[:, None]).astype(np.float32)
    aps = []
    for i in range(binary_targets.shape[0]):
        gt = binary_targets[i]
        scores = np.linspace(1.0, 0.0, num=gt.shape[0], endpoint=False)
        if gt.sum() > 0:
            aps.append(average_precision_score(gt, scores))
    metrics["mAP"] = float(np.mean(aps)) if aps else 0.0

    for k in topk:
        correct = binary_targets[:, :k].any(axis=1).mean()
        metrics[f"Recall@{k}"] = float(correct)
        metrics[f"NDCG@{k}"] = ndcg_at_k(relevance, k)
        metrics[f"ComponentConsistency@{k}"] = component_consistency_at_k(
            query_components=component_labels_np,
            retrieved_components=retrieved_components,
            k=k,
        )
        metrics[f"SeverityConsistency@{k}"] = severity_consistency_at_k(
            query_severity=severity_scores,
            retrieved_severity=retrieved_severity,
            k=k,
            tolerance=severity_tolerance,
        )
    return metrics


def retrieval_metrics(
    embeddings: np.ndarray,
    damage_labels: list[str],
    component_labels: list[str],
    severity_scores: np.ndarray,
    topk: list[int],
    severity_tolerance: float,
    same_damage_bonus: float,
    same_component_bonus: float,
    severity_bonus_scale: float,
) -> dict[str, float]:
    ranked = rank_indices(embeddings)
    return retrieval_metrics_from_ranks(
        ranked=ranked,
        damage_labels=damage_labels,
        component_labels=component_labels,
        severity_scores=severity_scores,
        topk=topk,
        severity_tolerance=severity_tolerance,
        same_damage_bonus=same_damage_bonus,
        same_component_bonus=same_component_bonus,
        severity_bonus_scale=severity_bonus_scale,
    )
