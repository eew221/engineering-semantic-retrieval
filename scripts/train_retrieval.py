from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_retrieval.datasets import RetrievalDataset, retrieval_collate_fn
from src.bridge_retrieval.losses import (
    batch_supervised_contrastive_loss,
    paired_supervised_contrastive_loss,
    batch_triplet_loss,
    regression_loss,
    weighted_pair_contrastive_loss,
)
from src.bridge_retrieval.modeling import BridgeRetrievalModel
from src.bridge_retrieval.utils import ensure_dir, load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/bridge_retrieval.yaml"))
    return parser.parse_args()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def encode_damage_labels(batch_damage: list[str], cfg: dict) -> torch.Tensor:
    name_to_idx = {name: idx for idx, name in enumerate(cfg["labels"]["damage_classes"])}
    indices = [name_to_idx.get(name, 0) for name in batch_damage]
    return torch.tensor(indices, dtype=torch.long)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    device_name = cfg.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_ds = RetrievalDataset(
        csv_path=cfg["data"]["train_csv"],
        image_column=cfg["data"]["image_column"],
        image_size=cfg["data"]["image_size"],
        is_train=True,
        max_samples=cfg["data"].get("max_train_samples"),
        use_full_image_fallback=cfg["data"]["use_full_image_fallback"],
        image_normalization=cfg["data"].get("image_normalization", "clip"),
        pair_weight_mode=cfg["train"].get("pair_weight_mode", "engineering"),
        partner_sampling_strategy=cfg["train"].get("partner_sampling_strategy", "random"),
        pair_weight_kwargs=cfg["train"].get("pair_weight_kwargs"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=retrieval_collate_fn,
    )

    model = BridgeRetrievalModel(
        backbone_name=cfg["model"]["backbone_name"],
        dropout=cfg["model"]["dropout"],
        freeze_vision_backbone=cfg["model"]["freeze_vision_backbone"],
        use_text_anchors=cfg["model"]["use_text_anchors"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    save_dir = ensure_dir(cfg["train"]["save_dir"])
    epoch_logs = []
    best_checkpoint_path = save_dir / f"{cfg['experiment_name']}.pt"
    exp_log_path = save_dir / f"{cfg['experiment_name']}_train_log.json"
    latest_log_path = save_dir / "train_log.json"
    start_epoch = 0

    resume_path = cfg["train"].get("resume_from_checkpoint")
    if resume_path:
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        start_epoch = int(state.get("epoch", 0))
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        existing_logs = state.get("epoch_logs")
        if isinstance(existing_logs, list):
            epoch_logs.extend(existing_logs[:start_epoch])
        print({"resume_from": str(resume_path), "start_epoch": start_epoch})

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        model.train()
        running_loss = 0.0
        num_steps = len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(batch)

                loss_mode = cfg["train"].get("loss_mode", "weighted_pair")
                if loss_mode == "triplet":
                    damage_targets = encode_damage_labels(batch["damage_class"], cfg).to(device)
                    pair_loss = batch_triplet_loss(
                        outputs["image_embeds"],
                        labels=damage_targets,
                        margin=float(cfg["train"].get("triplet_margin", 0.2)),
                    )
                elif loss_mode == "supcon":
                    damage_targets = encode_damage_labels(batch["damage_class"], cfg).to(device)
                    pair_loss = batch_supervised_contrastive_loss(
                        outputs["image_embeds"],
                        labels=damage_targets,
                        temperature=float(cfg["train"]["temperature"]),
                    )
                elif loss_mode == "hard_negative_supcon":
                    anchor_targets = encode_damage_labels(batch["damage_class"], cfg).to(device)
                    partner_targets = encode_damage_labels(batch["partner_damage_class"], cfg).to(device)
                    pair_loss = paired_supervised_contrastive_loss(
                        outputs["image_embeds"],
                        anchor_labels=anchor_targets,
                        partner_embeddings=outputs["partner_embeds"],
                        partner_labels=partner_targets,
                        temperature=float(cfg["train"]["temperature"]),
                    )
                else:
                    pair_loss = weighted_pair_contrastive_loss(
                        outputs["image_embeds"],
                        outputs["partner_embeds"],
                        batch["pair_target_weight"],
                        temperature=float(cfg["train"]["temperature"]),
                    )
                severity_loss = regression_loss(outputs["severity_pred"], batch["severity_score"])

                if cfg["model"]["use_text_anchors"]:
                    text_embeds = model.encode_text(batch["text_prompt"], device=device)
                    alignment_loss = 1.0 - torch.mean(
                        torch.sum(outputs["image_embeds"] * text_embeds, dim=-1)
                    )
                else:
                    alignment_loss = torch.tensor(0.0, device=device)

                loss = (
                    float(cfg["train"]["engineering_loss_weight"]) * pair_loss
                    + float(cfg["train"]["severity_loss_weight"]) * severity_loss
                    + float(cfg["train"]["cls_loss_weight"]) * alignment_loss
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["train"]["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

            if step == 1 or step % 200 == 0 or step == num_steps:
                avg_so_far = running_loss / step
                print(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "num_steps": num_steps,
                        "loss": float(loss.item()),
                        "avg_loss": avg_so_far,
                    }
                )

        avg_loss = running_loss / max(1, len(train_loader))
        log_row = {"epoch": epoch + 1, "train_loss": avg_loss}
        epoch_logs.append(log_row)
        print(log_row)

        epoch_checkpoint_path = save_dir / f"{cfg['experiment_name']}_epoch{epoch + 1}.pt"
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
            "epoch_logs": epoch_logs,
        }
        torch.save(state, epoch_checkpoint_path)
        torch.save(state, best_checkpoint_path)
        payload = {"experiment_name": cfg["experiment_name"], "epochs": epoch_logs, "latest_checkpoint": str(epoch_checkpoint_path)}
        save_json(payload, exp_log_path)
        save_json(payload, latest_log_path)
        print(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

    payload = {"experiment_name": cfg["experiment_name"], "epochs": epoch_logs, "checkpoint": str(best_checkpoint_path)}
    save_json(payload, exp_log_path)
    save_json(payload, latest_log_path)
    print(f"Saved final checkpoint to {best_checkpoint_path}")


if __name__ == "__main__":
    main()
