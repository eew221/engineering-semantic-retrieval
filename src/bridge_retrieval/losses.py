"""Loss functions for engineering-semantic retrieval."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    return x @ x.T


def weighted_pair_contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)
    logits = (emb_a * emb_b).sum(dim=-1) / temperature
    targets = weights.clamp(0.0, 1.0)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred.squeeze(-1), target)


def batch_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature
    mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, float("-inf"))

    labels = labels.view(-1, 1)
    positive_mask = labels.eq(labels.T) & (~mask)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = torch.where(torch.isfinite(log_prob), log_prob, torch.zeros_like(log_prob))

    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=embeddings.device)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


def paired_supervised_contrastive_loss(
    anchor_embeddings: torch.Tensor,
    anchor_labels: torch.Tensor,
    partner_embeddings: torch.Tensor,
    partner_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)
    partner_embeddings = F.normalize(partner_embeddings, dim=-1)

    logits = anchor_embeddings @ partner_embeddings.T / temperature
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positive_mask = anchor_labels.view(-1, 1).eq(partner_labels.view(1, -1))
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=anchor_embeddings.device)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


def batch_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=-1)
    distances = 1.0 - embeddings @ embeddings.T
    labels = labels.view(-1)
    n = embeddings.size(0)
    losses = []

    for i in range(n):
        positive_mask = labels == labels[i]
        positive_mask[i] = False
        negative_mask = labels != labels[i]

        if positive_mask.any() and negative_mask.any():
            hardest_positive = distances[i][positive_mask].max()
            hardest_negative = distances[i][negative_mask].min()
            losses.append(F.relu(hardest_positive - hardest_negative + margin))

    if not losses:
        return torch.tensor(0.0, device=embeddings.device)
    return torch.stack(losses).mean()
