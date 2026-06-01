"""CLIP-initialized retrieval model backed by the OpenAI clip package."""

from __future__ import annotations

from typing import Any

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_clip_name(backbone_name: str) -> str:
    normalized = backbone_name.replace("\\", "/").lower()
    if "vit-b/32" in normalized or "vit-base-patch32" in normalized or "clip-vit-base-patch32" in normalized:
        return "ViT-B/32"
    return backbone_name


class BridgeRetrievalModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        dropout: float = 0.1,
        freeze_vision_backbone: bool = False,
        use_text_anchors: bool = True,
    ) -> None:
        super().__init__()
        clip_name = _resolve_clip_name(backbone_name)
        self.clip, _ = clip.load(clip_name, device="cpu", jit=False)
        self.use_text_anchors = use_text_anchors

        projection_dim = int(self.clip.text_projection.shape[1])
        self.image_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
        )
        self.extent_head = nn.Linear(projection_dim, 1)

        if freeze_vision_backbone:
            visual = self.clip.visual
            for param in visual.parameters():
                param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = self.clip.encode_image(pixel_values)
        image_embeds = image_embeds.float()
        image_embeds = self.image_head(image_embeds)
        return F.normalize(image_embeds, dim=-1)

    def encode_text(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        tokens = clip.tokenize(prompts, truncate=True).to(device)
        text_embeds = self.clip.encode_text(tokens).float()
        return F.normalize(text_embeds, dim=-1)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        image_embeds = self.encode_image(batch["image"])
        outputs = {
            "image_embeds": image_embeds,
            "severity_pred": self.extent_head(image_embeds),
        }
        if "partner_image" in batch:
            partner_embeds = self.encode_image(batch["partner_image"])
            outputs["partner_embeds"] = partner_embeds
        return outputs
