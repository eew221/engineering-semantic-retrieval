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
    parser.add_argument("--config-a", type=Path, required=True)
    parser.add_argument("--config-b", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--no-checkpoint-a", action="store_true")
    parser.add_argument("--no-checkpoint-b", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--perm-iters", type=int, default=10000)
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


@torch.no_grad()
def compute_query_metrics(cfg: dict, checkpoint: Path | None, no_checkpoint: bool) -> dict[str, np.ndarray]:
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
        num_workers=cfg["data"].get("num_workers_eval", cfg["data"]["num_workers"]),
        collate_fn=retrieval_collate_fn,
    )

    state = None
    if not no_checkpoint:
        resolved = checkpoint or Path(cfg["train"]["save_dir"]) / f"{cfg['experiment_name']}.pt"
        state = torch.load(resolved, map_location=device)
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
        same_damage_bonus=float(cfg["retrieval"]["same_damage_bonus"]),
        same_component_bonus=float(cfg["retrieval"]["same_component_bonus"]),
        severity_bonus_scale=float(cfg["retrieval"]["severity_bonus_scale"]),
    )
    binary_targets = (damage_labels_np[ranked] == damage_labels_np[:, None]).astype(np.float32)
    retrieved_components = component_labels_np[ranked]
    retrieved_severity = severity_np[ranked]

    query_metrics: dict[str, np.ndarray] = {}
    aps = np.zeros(binary_targets.shape[0], dtype=np.float32)
    scores = np.linspace(1.0, 0.0, num=binary_targets.shape[1], endpoint=False)
    for i in range(binary_targets.shape[0]):
        gt = binary_targets[i]
        if gt.sum() > 0:
            aps[i] = average_precision_score(gt, scores)
    query_metrics["mAP"] = aps

    for k in cfg["retrieval"]["topk"]:
        query_metrics[f"Recall@{k}"] = binary_targets[:, :k].any(axis=1).astype(np.float32)
        query_metrics[f"NDCG@{k}"] = per_query_ndcg(relevance, k)
        query_metrics[f"ComponentConsistency@{k}"] = (retrieved_components[:, :k] == component_labels_np[:, None]).mean(axis=1)
        query_metrics[f"SeverityConsistency@{k}"] = (
            np.abs(retrieved_severity[:, :k] - severity_np[:, None]) <= float(cfg["retrieval"]["severity_tolerance"])
        ).mean(axis=1)
    return query_metrics


def bootstrap_diff(a: np.ndarray, b: np.ndarray, bootstrap_iters: int, seed: int) -> tuple[float, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    sample_idx = rng.integers(0, n, size=(bootstrap_iters, n))
    draws = (b[sample_idx].mean(axis=1) - a[sample_idx].mean(axis=1))
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    return float(draws.mean()), (float(ci_low), float(ci_high))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, perm_iters: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    diffs = b - a
    observed = abs(float(diffs.mean()))
    hits = 0
    for _ in range(perm_iters):
        signs = rng.choice([-1.0, 1.0], size=diffs.shape[0])
        stat = abs(float((diffs * signs).mean()))
        if stat >= observed:
            hits += 1
    return float((hits + 1) / (perm_iters + 1))


def main() -> None:
    args = parse_args()
    cfg_a = load_yaml(args.config_a)
    cfg_b = load_yaml(args.config_b)
    metrics_a = compute_query_metrics(cfg_a, args.checkpoint_a, args.no_checkpoint_a)
    metrics_b = compute_query_metrics(cfg_b, args.checkpoint_b, args.no_checkpoint_b)

    comparable_metrics = ["mAP", "Recall@1", "Recall@5", "Recall@10", "NDCG@5"]
    summary: dict[str, dict[str, object]] = {}
    for metric in comparable_metrics:
        diff_mean, diff_ci = bootstrap_diff(
            metrics_a[metric],
            metrics_b[metric],
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed,
        )
        p_value = permutation_pvalue(
            metrics_a[metric],
            metrics_b[metric],
            perm_iters=args.perm_iters,
            seed=args.seed + 1,
        )
        summary[metric] = {
            "mean_a": float(metrics_a[metric].mean()),
            "mean_b": float(metrics_b[metric].mean()),
            "diff_mean": diff_mean,
            "diff_ci95": list(diff_ci),
            "perm_p_value": p_value,
        }

    payload = {
        "config_a": str(args.config_a),
        "config_b": str(args.config_b),
        "checkpoint_a": None if args.no_checkpoint_a else str(args.checkpoint_a or Path(cfg_a["train"]["save_dir"]) / f"{cfg_a['experiment_name']}.pt"),
        "checkpoint_b": None if args.no_checkpoint_b else str(args.checkpoint_b or Path(cfg_b["train"]["save_dir"]) / f"{cfg_b['experiment_name']}.pt"),
        "bootstrap_iters": args.bootstrap_iters,
        "perm_iters": args.perm_iters,
        "summary": summary,
    }

    output_json = args.output_json
    if output_json is None:
        output_json = ensure_dir(ROOT / "outputs" / "metrics") / "paired_significance.json"
    else:
        ensure_dir(output_json.parent)
    save_json(payload, output_json)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved paired significance summary to {output_json}")


if __name__ == "__main__":
    main()
