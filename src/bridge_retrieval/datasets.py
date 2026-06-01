"""Dataset and transform utilities for bridge defect retrieval."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from .engineering_semantics import SampleSemantics, pair_target_weight
from .labels import make_compositional_prompt


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def resolve_image_normalization(image_normalization: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    normalized = image_normalization.strip().lower()
    if normalized in {"siglip", "imagenet_standard", "imagenet-standard"}:
        return SIGLIP_MEAN, SIGLIP_STD
    return CLIP_MEAN, CLIP_STD


def build_image_transform(image_size: int, is_train: bool, image_normalization: str = "clip") -> transforms.Compose:
    mean, std = resolve_image_normalization(image_normalization)
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )
    return transforms.Compose(ops)


@dataclass
class RetrievalRow:
    sample_id: str
    image_path: str
    crop_path: str
    damage_class: str
    component_class: str
    severity_score: float
    split: str


class RetrievalDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_column: str = "crop_path",
        image_size: int = 224,
        is_train: bool = False,
        max_samples: int | None = None,
        use_full_image_fallback: bool = True,
        image_normalization: str = "clip",
        pair_weight_mode: str = "engineering",
        partner_sampling_strategy: str = "random",
        pair_weight_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].copy()
        self.df = self.df.reset_index(drop=True)
        self.image_column = image_column
        self.use_full_image_fallback = use_full_image_fallback
        self.transform = build_image_transform(
            image_size=image_size,
            is_train=is_train,
            image_normalization=image_normalization,
        )
        self.is_train = is_train
        self.pair_weight_mode = pair_weight_mode
        self.partner_sampling_strategy = partner_sampling_strategy
        self.pair_weight_kwargs = pair_weight_kwargs or {}

        self._indices_by_damage: dict[str, list[int]] = {}
        self._indices_by_component: dict[str, list[int]] = {}
        self._indices_by_damage_component: dict[tuple[str, str], list[int]] = {}
        for idx, row in self.df.iterrows():
            damage = str(row["damage_class"])
            component = str(row["component_class"])
            self._indices_by_damage.setdefault(damage, []).append(idx)
            self._indices_by_component.setdefault(component, []).append(idx)
            self._indices_by_damage_component.setdefault((damage, component), []).append(idx)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        preferred = Path(str(row[self.image_column]))
        if preferred.exists():
            return preferred
        if self.use_full_image_fallback and "image_path" in row.index:
            fallback = Path(str(row["image_path"]))
            if fallback.exists():
                return fallback
        raise FileNotFoundError(f"Missing image for sample {row.get('sample_id', 'unknown')}")

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def _load_row_image(self, row: pd.Series) -> torch.Tensor:
        try:
            return self._load_image(self._resolve_image_path(row))
        except Exception:
            if self.use_full_image_fallback and "image_path" in row.index:
                fallback = Path(str(row["image_path"]))
                if fallback.exists():
                    return self._load_image(fallback)
            raise

    def _sample_partner_index(self, index: int) -> int:
        row = self.df.iloc[index]
        damage = str(row["damage_class"])
        component = str(row["component_class"])
        same_damage = [idx for idx in self._indices_by_damage.get(damage, []) if idx != index]
        same_component = [idx for idx in self._indices_by_component.get(component, []) if idx != index]
        same_both = [idx for idx in self._indices_by_damage_component.get((damage, component), []) if idx != index]

        if self.partner_sampling_strategy == "engineering_hard":
            same_both_set = set(same_both)
            same_damage_set = set(same_damage)
            same_component_set = set(same_component)
            pos = same_both
            hard_neg_damage = [idx for idx in same_damage if idx not in same_both_set]
            hard_neg_component = [idx for idx in same_component if idx not in same_both_set]
            easy_neg = [
                idx
                for idx in range(len(self.df))
                if idx != index and idx not in same_damage_set and idx not in same_component_set
            ]

            bucket_order = []
            r = random.random()
            if r < 0.45:
                bucket_order = [pos, hard_neg_damage, hard_neg_component, easy_neg]
            elif r < 0.75:
                bucket_order = [hard_neg_damage, hard_neg_component, pos, easy_neg]
            else:
                bucket_order = [hard_neg_component, hard_neg_damage, pos, easy_neg]

            for bucket in bucket_order:
                if bucket:
                    return random.choice(bucket)

        candidates = list(set(same_damage) | set(same_component))
        if candidates:
            return random.choice(candidates)
        all_indices = list(range(len(self.df)))
        all_indices.remove(index)
        return random.choice(all_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image = self._load_row_image(row)

        sample = {
            "sample_id": str(row["sample_id"]),
            "image": image,
            "crop_path": str(row.get("crop_path", "")),
            "image_path": str(row.get("image_path", "")),
            "damage_class": str(row["damage_class"]),
            "component_class": str(row["component_class"]),
            "severity_score": torch.tensor(float(row["severity_score"]), dtype=torch.float32),
            "text_prompt": make_compositional_prompt(str(row["damage_class"]), str(row["component_class"])),
        }

        if not self.is_train:
            return sample

        partner_index = self._sample_partner_index(index)
        partner_row = self.df.iloc[partner_index]
        partner = {
            "partner_image": self._load_row_image(partner_row),
            "partner_damage_class": str(partner_row["damage_class"]),
            "partner_component_class": str(partner_row["component_class"]),
            "partner_severity_score": torch.tensor(float(partner_row["severity_score"]), dtype=torch.float32),
        }

        if self.pair_weight_mode == "defect_only_binary":
            weight = float(str(row["damage_class"]) == str(partner_row["damage_class"]))
        else:
            weight = pair_target_weight(
                SampleSemantics(
                    damage_class=str(row["damage_class"]),
                    component_class=str(row["component_class"]),
                    severity_score=float(row["severity_score"]),
                ),
                SampleSemantics(
                    damage_class=str(partner_row["damage_class"]),
                    component_class=str(partner_row["component_class"]),
                    severity_score=float(partner_row["severity_score"]),
                ),
                **self.pair_weight_kwargs,
            )
        sample.update(partner)
        sample["pair_target_weight"] = torch.tensor(weight, dtype=torch.float32)
        return sample


def retrieval_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tensor_keys = {
        "image",
        "partner_image",
        "severity_score",
        "partner_severity_score",
        "pair_target_weight",
    }
    for key in batch[0].keys():
        if key in tensor_keys:
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result
ImageFile.LOAD_TRUNCATED_IMAGES = True
