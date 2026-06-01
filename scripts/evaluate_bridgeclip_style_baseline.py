from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_retrieval.datasets import RetrievalDataset, retrieval_collate_fn
from src.bridge_retrieval.labels import make_compositional_prompt
from src.bridge_retrieval.metrics import retrieval_metrics, similarity_matrix
from src.bridge_retrieval.modeling import BridgeRetrievalModel
from src.bridge_retrieval.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="bridgeclip_style_zero_shot")
    parser.add_argument("--alpha-image", type=float, default=0.7)
    parser.add_argument("--beta-text", type=float, default=0.3)
    return parser.parse_args()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


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
        num_workers=cfg["data"]["num_workers"],
        collate_fn=retrieval_collate_fn,
    )

    model = BridgeRetrievalModel(
        backbone_name=cfg["model"]["backbone_name"],
        dropout=cfg["model"]["dropout"],
        freeze_vision_backbone=cfg["model"]["freeze_vision_backbone"],
        use_text_anchors=True,
    ).to(device)
    model.eval()

    prompts = [
        make_compositional_prompt(damage, component)
        for damage in cfg["labels"]["damage_classes"]
        for component in cfg["labels"]["component_classes"]
    ]
    text_embeds = model.encode_text(prompts, device=device).cpu().numpy()

    fused_embeddings = []
    damage_labels = []
    component_labels = []
    severity_scores = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        image_embeds = outputs["image_embeds"].cpu().numpy()
        prompt_logits = image_embeds @ text_embeds.T
        prompt_probs = np.exp(prompt_logits - prompt_logits.max(axis=1, keepdims=True))
        prompt_probs = prompt_probs / prompt_probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        weighted_text = prompt_probs @ text_embeds
        fused = args.alpha_image * image_embeds + args.beta_text * weighted_text
        fused = fused / np.linalg.norm(fused, axis=1, keepdims=True).clip(min=1e-8)
        fused_embeddings.append(fused)
        damage_labels.extend(batch["damage_class"])
        component_labels.extend(batch["component_class"])
        severity_scores.extend(batch["severity_score"].cpu().numpy().tolist())

    embeddings = np.concatenate(fused_embeddings, axis=0)
    metrics = retrieval_metrics(
        embeddings=embeddings,
        damage_labels=damage_labels,
        component_labels=component_labels,
        severity_scores=np.asarray(severity_scores, dtype=np.float32),
        topk=list(cfg["retrieval"]["topk"]),
        severity_tolerance=float(cfg["retrieval"]["severity_tolerance"]),
        same_damage_bonus=float(cfg["retrieval"]["same_damage_bonus"]),
        same_component_bonus=float(cfg["retrieval"]["same_component_bonus"]),
        severity_bonus_scale=float(cfg["retrieval"]["severity_bonus_scale"]),
    )

    payload = {
        "method": "bridgeclip_style_prompt_conditioned_zeroshot",
        "alpha_image": args.alpha_image,
        "beta_text": args.beta_text,
        "num_prompts": len(prompts),
        "base": metrics,
    }
    out_dir = ensure_dir(cfg["output"]["metrics_dir"])
    save_path = out_dir / f"{args.output_name}_test_metrics.json"
    save_json(payload, save_path)
    print(json.dumps(payload, indent=2))
    print(f"Saved metrics to {save_path}")


if __name__ == "__main__":
    main()
