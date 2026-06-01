"""Prepare dacl10k retrieval metadata from official LabelMe-style annotations.

Expected dataset layout after extraction:

<data-root>/
  images/
    train/
    validation/
    test/
  annotations/
    train/
    validation/
    test/

Each label file is expected to be a LabelMe JSON with fields such as:
- imagePath
- imageHeight
- imageWidth
- shapes: [{label, points, shape_type}, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_retrieval.labels import DACL10K_COMPONENT_CLASSES, DACL10K_DAMAGE_CLASSES
from src.bridge_retrieval.utils import ensure_dir


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def shape_to_mask(width: int, height: int, shape: dict) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    points = shape.get("points", [])
    if len(points) < 2:
        return np.zeros((height, width), dtype=np.uint8)

    xy = [tuple(map(float, pt)) for pt in points]
    shape_type = str(shape.get("shape_type", "polygon")).lower()
    if shape_type == "rectangle" and len(xy) >= 2:
        draw.rectangle([xy[0], xy[1]], outline=1, fill=1)
    elif shape_type == "circle" and len(xy) >= 2:
        (x0, y0), (x1, y1) = xy[0], xy[1]
        radius = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        draw.ellipse([x0 - radius, y0 - radius, x0 + radius, y0 + radius], outline=1, fill=1)
    elif shape_type == "line" and len(xy) >= 2:
        draw.line(xy, fill=1, width=5)
    elif shape_type == "linestrip" and len(xy) >= 2:
        draw.line(xy, fill=1, width=5)
    else:
        draw.polygon(xy, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1), int(y1)


def overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    denom = max(1, int((mask_a > 0).sum()))
    return float(inter) / float(denom)


def infer_image_path(data_root: Path, split_name: str, label_json: Path, meta: dict) -> Path:
    candidates = []
    image_path = meta.get("imagePath")
    if image_path:
        candidates.append(data_root / str(image_path))
        candidates.append(data_root / "images" / split_name / Path(str(image_path)).name)
        candidates.append(label_json.parent / str(image_path))
    stem = label_json.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidates.append(data_root / "images" / split_name / f"{stem}{ext}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve image for label file: {label_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dacl10k"))
    parser.add_argument("--split-name", type=str, required=True, choices=["train", "validation", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_root = args.data_root / "annotations" / args.split_name
    if not labels_root.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_root}")

    output_dir = ensure_dir(args.output_dir)
    crops_dir = ensure_dir(output_dir / "crops" / args.split_name)

    damage_classes = {normalize_name(x) for x in DACL10K_DAMAGE_CLASSES}
    component_classes = {normalize_name(x) for x in DACL10K_COMPONENT_CLASSES if x != "bridge component unknown"}

    records = []
    for label_json in sorted(labels_root.rglob("*.json")):
        with open(label_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        width = int(meta["imageWidth"])
        height = int(meta["imageHeight"])
        image_path = infer_image_path(args.data_root, args.split_name, label_json, meta)
        image = Image.open(image_path).convert("RGB")

        parsed = []
        for idx, shape in enumerate(meta.get("shapes", [])):
            class_name = normalize_name(str(shape.get("label", "")))
            if not class_name:
                continue
            mask = shape_to_mask(width, height, shape)
            if mask.sum() == 0:
                continue
            parsed.append({"class_name": class_name, "mask": mask, "shape_id": idx})

        component_instances = [x for x in parsed if x["class_name"] in component_classes]
        damage_instances = [x for x in parsed if x["class_name"] in damage_classes]

        for damage in damage_instances:
            damage_mask = damage["mask"]
            component_name = "bridge component unknown"
            best_overlap = 0.0
            for comp in component_instances:
                score = overlap_ratio(damage_mask, comp["mask"])
                if score > best_overlap:
                    best_overlap = score
                    component_name = comp["class_name"]

            x0, y0, x1, y1 = bbox_from_mask(damage_mask)
            margin_x = max(8, int((x1 - x0) * 0.1))
            margin_y = max(8, int((y1 - y0) * 0.1))
            crop_box = (
                max(0, x0 - margin_x),
                max(0, y0 - margin_y),
                min(width, x1 + margin_x),
                min(height, y1 + margin_y),
            )
            crop = image.crop(crop_box)
            crop_name = f"{args.split_name}_{label_json.stem}_{damage['shape_id']}.png"
            crop_path = crops_dir / crop_name
            crop.save(crop_path)

            damage_area = float((damage_mask > 0).sum())
            severity_score = damage_area / float(width * height)
            records.append(
                {
                    "sample_id": f"{args.split_name}_{label_json.stem}_{damage['shape_id']}",
                    "image_path": str(image_path.resolve()),
                    "crop_path": str(crop_path.resolve()),
                    "damage_class": damage["class_name"],
                    "component_class": component_name,
                    "severity_score": severity_score,
                    "area_pixels": damage_area,
                    "split": args.split_name,
                }
            )

    df = pd.DataFrame.from_records(records)
    csv_path = output_dir / f"{args.split_name}_retrieval.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples to {csv_path}")


if __name__ == "__main__":
    main()
