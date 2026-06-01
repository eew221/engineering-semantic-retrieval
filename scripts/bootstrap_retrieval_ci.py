from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_retrieval.datasets import RetrievalDataset, retrieval_collate_fn
from src.bridge_retrieval.engineering_semantics import SampleSemantics, dcg
from src.bridge_retrieval.metrics import build_relevance_matrix, similarity_matrix
from src.bridge_retrieval.modeling import BridgeRetrievalModel, build_model_for_state_dict
from src.bridge_retrieval.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def per_query_ndcg(relevance: np.ndarray, k: int) -> np.ndarray:
    values = np.zeros(relevance.shape[0], dtype=np.float32)
    for i, row in enumerate(relevance):
        rel = row[:k]
        ideal = np.sort(row)[::-1][:k]
        ideal_dcg = dcg(ideal)
        values[i] = 0.0 if ideal_dcg == 0 else dcg(rel) / ideal_dcg
    return values


def compute_query_metrics(
    ranked: np.ndarray,
    damage_labels: list[str],
    component_labels: list[str],
    severity_scores: np.ndarray,
    topk: list[int],
    severity_tolerance: float,
    same_damage_bonus: float,
    same_component_bonus: float,
    severity_bonus_scale: float,
) -> dict[str, np.ndarray]:
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

    binary_targets = (damage_labels_np[ranked] == damage_labels_np[:, None]).astype(np.float32)
    retrieved_components = component_labels_np[ranked]
    retrieved_severity = severity_scores[ranked]

    metrics: dict[str, np.ndarray] = {}
    aps = np.zeros(binary_targets.shape[0], dtype=np.float32)
    scores = np.linspace(1.0, 0.0, num=binary_targets.shape[1], endpoint=False)
    for i in range(binary_targets.shape[0]):
        gt = binary_targets[i]
        if gt.sum() > 0:
            aps[i] = average_precision_score(gt, scores)
    metrics["mAP"] = aps

    for k in topk:
        metrics[f"Recall@{k}"] = binary_targets[:, :k].any(axis=1).astype(np.float32)
        metrics[f"NDCG@{k}"] = per_query_ndcg(relevance, k)
        metrics[f"ComponentConsistency@{k}"] = (retrieved_components[:, :k] == component_labels_np[:, None]).mean(axis=1)
        metrics[f"SeverityConsistency@{k}"] = (
            np.abs(retrieved_severity[:, :k] - severity_scores[:, None]) <= severity_tolerance
        ).mean(axis=1)
    return metrics


def bootstrap_summary(values: np.ndarray, bootstrap_iters: int, seed: int) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    sample_idx = rng.integers(0, n, size=(bootstrap_iters, n))
    draws = values[sample_idx].mean(axis=1)
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    return {
        "point": float(values.mean()),
        "ci95": [float(ci_low), float(ci_high)],
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device_name = cfg.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    dataset = RetrievalDataset(
        csv_path=cfg["data"]["test_csv"],
        image_column=cfg["data"]["image_column"],
        image_size=cfg["data"]["image_size"],
        is_train=False,
        max_samples=cfg["data"].get("max_eval_samples"),
        use_full_image_fallback=cfg["data"]["use_full_image_fallback"],
        image_normalization=cfg["data"].get("image_normalization", "clip"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=retrieval_collate_fn,
    )

    state = None
    checkpoint_label = "zero_shot"
    if not args.no_checkpoint:
        checkpoint = args.checkpoint or Path(cfg["train"]["save_dir"]) / f"{cfg['experiment_name']}.pt"
        state = torch.load(checkpoint, map_location=device)
        checkpoint_label = checkpoint.stem
        model, _ = build_model_for_state_dict(
            backbone_name=cfg["model"]["backbone_name"],
            dropout=cfg["model"]["dropout"],
            freeze_vision_backbone=cfg["model"]["freeze_vision_backbone"],
            use_text_anchors=cfg["model"]["use_text_anchors"],
            state_dict=state["model_state_dict"],
        )
    else:
        model = BridgeRetrievalModel(
            backbone_name=cfg["model"]["backbone_name"],
            dropout=cfg["model"]["dropout"],
            freeze_vision_backbone=cfg["model"]["freeze_vision_backbone"],
            use_text_anchors=cfg["model"]["use_text_anchors"],
        )
    model = model.to(device)
    model.eval()

    embeddings = []
    damage_labels = []
    component_labels = []
    severity_scores = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        embeddings.append(outputs["image_embeds"].detach().cpu().numpy())
        damage_labels.extend(batch["damage_class"])
        component_labels.extend(batch["component_class"])
        severity_scores.extend(batch["severity_score"].detach().cpu().numpy().tolist())

    embeddings_np = np.concatenate(embeddings, axis=0)
    severity_np = np.asarray(severity_scores, dtype=np.float32)
    ranked = np.argsort(-similarity_matrix(embeddings_np), axis=1)

    query_metrics = compute_query_metrics(
        ranked=ranked,
        damage_labels=damage_labels,
        component_labels=component_labels,
        severity_scores=severity_np,
        topk=list(cfg["retrieval"]["topk"]),
        severity_tolerance=float(cfg["retrieval"]["severity_tolerance"]),
        same_damage_bonus=float(cfg["retrieval"]["same_damage_bonus"]),
        same_component_bonus=float(cfg["retrieval"]["same_component_bonus"]),
        severity_bonus_scale=float(cfg["retrieval"]["severity_bonus_scale"]),
    )

    summary = {
        "experiment_name": cfg["experiment_name"],
        "checkpoint": checkpoint_label,
        "bootstrap_iters": args.bootstrap_iters,
        "num_queries": int(len(damage_labels)),
        "metrics": {
            name: bootstrap_summary(values, bootstrap_iters=args.bootstrap_iters, seed=args.seed)
            for name, values in query_metrics.items()
        },
    }

    output_json = args.output_json
    if output_json is None:
        metrics_dir = ensure_dir(ROOT / cfg["output"]["metrics_dir"])
        output_json = metrics_dir / f"{cfg['experiment_name']}_bootstrap_ci.json"
    else:
        ensure_dir(output_json.parent)

    save_json(summary, output_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved bootstrap CI summary to {output_json}")


if __name__ == "__main__":
    main()
