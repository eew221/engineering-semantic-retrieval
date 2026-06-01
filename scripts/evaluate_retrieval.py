from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_retrieval.datasets import RetrievalDataset, retrieval_collate_fn
from src.bridge_retrieval.labels import make_component_prompt, make_damage_prompt
from src.bridge_retrieval.metrics import retrieval_metrics, retrieval_metrics_from_ranks, similarity_matrix
from src.bridge_retrieval.modeling import BridgeRetrievalModel
from src.bridge_retrieval.qualitative import save_retrieval_grid
from src.bridge_retrieval.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/bridge_retrieval.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--save-qualitative", action="store_true")
    return parser.parse_args()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def rerank_indices(
    base_sim: np.ndarray,
    damage_probs: np.ndarray,
    component_probs: np.ndarray,
    pred_severity: np.ndarray,
    topn: int,
    damage_scale: float,
    component_scale: float,
    severity_bonus_scale: float,
) -> np.ndarray:
    n = base_sim.shape[0]
    ranked = np.argsort(-base_sim, axis=1)
    reranked = ranked.copy()
    for i in range(n):
        head = ranked[i, :topn]
        scores = base_sim[i, head].copy()
        scores += (damage_probs[head] @ damage_probs[i]) * damage_scale
        scores += (component_probs[head] @ component_probs[i]) * component_scale
        scores += np.maximum(0.0, 1.0 - np.abs(pred_severity[head] - pred_severity[i])) * severity_bonus_scale
        order = np.argsort(-scores)
        reranked[i, :topn] = head[order]
    return reranked


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
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers_eval", 0),
        pin_memory=torch.cuda.is_available() and cfg.get("device", "cuda") != "cpu",
        collate_fn=retrieval_collate_fn,
    )

    model = BridgeRetrievalModel(
        backbone_name=cfg["model"]["backbone_name"],
        dropout=cfg["model"]["dropout"],
        freeze_vision_backbone=cfg["model"]["freeze_vision_backbone"],
        use_text_anchors=cfg["model"]["use_text_anchors"],
    ).to(device)

    if not args.no_checkpoint:
        checkpoint = args.checkpoint or Path(cfg["train"]["save_dir"]) / f"{cfg['experiment_name']}.pt"
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    model.eval()

    embeddings = []
    damage_labels = []
    component_labels = []
    severity_scores = []
    crop_paths = []
    severity_preds = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        embeddings.append(outputs["image_embeds"].cpu().numpy())
        damage_labels.extend(batch["damage_class"])
        component_labels.extend(batch["component_class"])
        severity_scores.extend(batch["severity_score"].cpu().numpy().tolist())
        severity_preds.extend(outputs["severity_pred"].squeeze(-1).cpu().numpy().tolist())
        crop_paths.extend(batch["crop_path"])

    embeddings_np = np.concatenate(embeddings, axis=0)
    severity_np = np.asarray(severity_scores, dtype=np.float32)
    metrics = retrieval_metrics(
        embeddings=embeddings_np,
        damage_labels=damage_labels,
        component_labels=component_labels,
        severity_scores=severity_np,
        topk=list(cfg["retrieval"]["topk"]),
        severity_tolerance=float(cfg["retrieval"]["severity_tolerance"]),
        same_damage_bonus=float(cfg["retrieval"]["same_damage_bonus"]),
        same_component_bonus=float(cfg["retrieval"]["same_component_bonus"]),
        severity_bonus_scale=float(cfg["retrieval"]["severity_bonus_scale"]),
    )
    result_payload: dict[str, object] = {"base": metrics}

    ranked_before = np.argsort(-similarity_matrix(embeddings_np), axis=1)

    if args.rerank:
        damage_prompts = [make_damage_prompt(x) for x in cfg["labels"]["damage_classes"]]
        component_prompts = [make_component_prompt(x) for x in cfg["labels"]["component_classes"]]
        damage_text_embeds = model.encode_text(damage_prompts, device=device).cpu().numpy()
        component_text_embeds = model.encode_text(component_prompts, device=device).cpu().numpy()

        damage_logits = embeddings_np @ damage_text_embeds.T
        component_logits = embeddings_np @ component_text_embeds.T
        damage_probs = np.exp(damage_logits - damage_logits.max(axis=1, keepdims=True))
        damage_probs = damage_probs / damage_probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        component_probs = np.exp(component_logits - component_logits.max(axis=1, keepdims=True))
        component_probs = component_probs / component_probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        pred_severity = np.asarray(severity_preds, dtype=np.float32)
        topn = int(cfg["retrieval"].get("rerank_topn", 20))

        reranked = rerank_indices(
            base_sim=similarity_matrix(embeddings_np),
            damage_probs=damage_probs,
            component_probs=component_probs,
            pred_severity=pred_severity,
            topn=topn,
            damage_scale=float(cfg["retrieval"].get("rerank_damage_scale", cfg["retrieval"]["same_damage_bonus"])),
            component_scale=float(cfg["retrieval"].get("rerank_component_scale", cfg["retrieval"]["same_component_bonus"])),
            severity_bonus_scale=float(cfg["retrieval"].get("rerank_severity_scale", cfg["retrieval"]["severity_bonus_scale"])),
        )
        rerank_metrics = retrieval_metrics_from_ranks(
            ranked=reranked,
            damage_labels=damage_labels,
            component_labels=component_labels,
            severity_scores=severity_np,
            topk=list(cfg["retrieval"]["topk"]),
            severity_tolerance=float(cfg["retrieval"]["severity_tolerance"]),
            same_damage_bonus=float(cfg["retrieval"]["same_damage_bonus"]),
            same_component_bonus=float(cfg["retrieval"]["same_component_bonus"]),
            severity_bonus_scale=float(cfg["retrieval"]["severity_bonus_scale"]),
        )
        result_payload["reranked"] = rerank_metrics
        metrics = rerank_metrics

        if args.save_qualitative:
            qualitative_dir = ensure_dir(cfg["output"]["qualitative_dir"])
            chosen = 0
            for i in range(len(crop_paths)):
                before = ranked_before[i, :5]
                after = reranked[i, :5]
                if np.array_equal(before, after):
                    continue
                title = f"q={damage_labels[i]} / {component_labels[i]}"
                save_retrieval_grid(
                    query_path=crop_paths[i],
                    before_paths=[crop_paths[j] for j in before],
                    after_paths=[crop_paths[j] for j in after],
                    out_path=qualitative_dir / f"retrieval_query_{i:04d}.png",
                    title=title,
                )
                chosen += 1
                if chosen >= 6:
                    break

    metrics_dir = ensure_dir(cfg["output"]["metrics_dir"])
    save_path = metrics_dir / f"{cfg['experiment_name']}_test_metrics.json"
    save_json(result_payload, save_path)
    print(result_payload)
    print(f"Saved metrics to {save_path}")


if __name__ == "__main__":
    main()
