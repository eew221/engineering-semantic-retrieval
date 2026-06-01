from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_retrieval.datasets import RetrievalDataset, retrieval_collate_fn
from src.bridge_retrieval.modeling import BridgeRetrievalModel
from src.bridge_retrieval.utils import ensure_dir, load_yaml

try:
    import umap  # type: ignore
except Exception:
    umap = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True)
    return parser.parse_args()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def scatter_plot(coords: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(unique):
        mask = [x == label for x in labels]
        pts = coords[mask]
        plt.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.75, label=label, color=cmap(idx % 20))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=7, loc="best", ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


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
        max_samples=args.max_samples,
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
        use_text_anchors=cfg["model"]["use_text_anchors"],
    ).to(device)

    if not args.no_checkpoint:
        checkpoint = args.checkpoint or Path(cfg["train"]["save_dir"]) / f"{cfg['experiment_name']}.pt"
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    model.eval()

    embeds = []
    damage = []
    component = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        embeds.append(outputs["image_embeds"].cpu().numpy())
        damage.extend(batch["damage_class"])
        component.extend(batch["component_class"])
    embeds = np.concatenate(embeds, axis=0)

    out_dir = ensure_dir(args.output_dir)
    payload = {
        "tag": args.tag,
        "num_samples": int(embeds.shape[0]),
        "embedding_dim": int(embeds.shape[1]),
        "checkpoint": None if args.no_checkpoint else str(args.checkpoint or Path(cfg["train"]["save_dir"]) / f"{cfg['experiment_name']}.pt"),
    }
    (out_dir / f"{args.tag}_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(embeds).to_csv(out_dir / f"{args.tag}_embeddings.csv", index=False)
    pd.DataFrame({"damage_class": damage, "component_class": component}).to_csv(out_dir / f"{args.tag}_labels.csv", index=False)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    tsne_coords = tsne.fit_transform(embeds)
    scatter_plot(tsne_coords, damage, out_dir / f"{args.tag}_tsne_damage.png", f"{args.tag}: t-SNE by damage")
    scatter_plot(tsne_coords, component, out_dir / f"{args.tag}_tsne_component.png", f"{args.tag}: t-SNE by component")

    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_coords = reducer.fit_transform(embeds)
        scatter_plot(umap_coords, damage, out_dir / f"{args.tag}_umap_damage.png", f"{args.tag}: UMAP by damage")
        scatter_plot(umap_coords, component, out_dir / f"{args.tag}_umap_component.png", f"{args.tag}: UMAP by component")

    print(f"Saved embedding exports to {out_dir}")


if __name__ == "__main__":
    main()
