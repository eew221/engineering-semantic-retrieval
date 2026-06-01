"""CLIP-initialized retrieval model with optional transformer-based backbones."""

from __future__ import annotations

import os
from typing import Any

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from transformers import AutoModel, AutoTokenizer


def _resolve_clip_name(backbone_name: str) -> str:
    normalized = backbone_name.replace("\\", "/").lower()
    if (
        "vit-b/32" in normalized
        or "vit-base-patch32" in normalized
        or "clip-vit-base-patch32" in normalized
        or "clip-vit-b-32" in normalized
    ):
        return "ViT-B/32"
    return backbone_name


def _resolve_backend(backbone_name: str) -> str:
    normalized = backbone_name.replace("\\", "/").lower()
    if "models--openai--clip-vit-base-patch32" in normalized or normalized.endswith("vit-b/32"):
        return "openai_clip"
    if "snapshots/" in normalized and "clip-vit-base-patch32" in normalized:
        return "openai_clip"
    return "transformers"


def _resolve_feature_dim(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Transformer backbone is missing config; cannot resolve feature dimension.")

    projection_dim = getattr(config, "projection_dim", None)
    if projection_dim:
        return int(projection_dim)

    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None):
        return int(text_config.hidden_size)

    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None and getattr(vision_config, "hidden_size", None):
        return int(vision_config.hidden_size)

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size:
        return int(hidden_size)

    raise ValueError("Unable to infer feature dimension from backbone config.")


def infer_checkpoint_backend(state_dict: dict[str, torch.Tensor]) -> str | None:
    keys = list(state_dict.keys())
    if any(
        key.startswith("clip.visual.")
        or key.startswith("clip.transformer.")
        or key in {"clip.positional_embedding", "clip.text_projection", "clip.token_embedding.weight", "clip.ln_final.weight"}
        for key in keys
    ):
        return "openai_clip"
    if any(
        key.startswith("clip.vision_model.")
        or key.startswith("clip.text_model.")
        or key.startswith("transformer.vision_model.")
        or key.startswith("transformer.text_model.")
        for key in keys
    ):
        return "transformers"
    return None


def _adapt_state_dict_for_backend(
    state_dict: dict[str, torch.Tensor],
    backend: str,
) -> dict[str, torch.Tensor]:
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("severity_head."):
            new_key = new_key.replace("severity_head.", "extent_head.", 1)
        if backend == "transformers" and new_key.startswith("clip."):
            new_key = new_key.replace("clip.", "transformer.", 1)
        adapted[new_key] = value
    return adapted


def build_model_for_state_dict(
    backbone_name: str,
    dropout: float,
    freeze_vision_backbone: bool,
    use_text_anchors: bool,
    state_dict: dict[str, torch.Tensor],
) -> tuple[BridgeRetrievalModel, str]:
    inferred = infer_checkpoint_backend(state_dict)
    candidates = [inferred] if inferred is not None else []
    candidates.extend([backend for backend in ("transformers", "openai_clip") if backend not in candidates])
    last_error: RuntimeError | None = None
    for backend in candidates:
        model = BridgeRetrievalModel(
            backbone_name=backbone_name,
            dropout=dropout,
            freeze_vision_backbone=freeze_vision_backbone,
            use_text_anchors=use_text_anchors,
            backend=backend,
        )
        try:
            model.load_state_dict(_adapt_state_dict_for_backend(state_dict, backend))
            return model, backend
        except RuntimeError as err:
            last_error = err
            continue
    assert last_error is not None
    raise last_error


class BridgeRetrievalModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        dropout: float = 0.1,
        freeze_vision_backbone: bool = False,
        use_text_anchors: bool = True,
        backend: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backend = backend or _resolve_backend(backbone_name)
        self.use_text_anchors = use_text_anchors

        if self.backend == "openai_clip":
            clip_name = _resolve_clip_name(backbone_name)
            self.clip, _ = clip.load(clip_name, device="cpu", jit=False)
            projection_dim = int(self.clip.text_projection.shape[1])
        else:
            self.transformer = AutoModel.from_pretrained(backbone_name)
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            projection_dim = _resolve_feature_dim(self.transformer)

        self.image_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
        )
        self.extent_head = nn.Linear(projection_dim, 1)

        if freeze_vision_backbone:
            if self.backend == "openai_clip":
                visual = self.clip.visual
                for param in visual.parameters():
                    param.requires_grad = False
            else:
                visual = getattr(self.transformer, "vision_model", None)
                if visual is not None:
                    for param in visual.parameters():
                        param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.backend == "openai_clip":
            image_embeds = self.clip.encode_image(pixel_values)
        else:
            image_embeds = self.transformer.get_image_features(pixel_values=pixel_values)
        image_embeds = image_embeds.float()
        image_embeds = self.image_head(image_embeds)
        return F.normalize(image_embeds, dim=-1)

    def encode_text(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        if self.backend == "openai_clip":
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_embeds = self.clip.encode_text(tokens).float()
        else:
            tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            text_embeds = self.transformer.get_text_features(**tokens).float()
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
