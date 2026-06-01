from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def make_sample(path: Path, damage: str, component: str, severity: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), color=(235, 235, 235))
    draw = ImageDraw.Draw(image)

    component_color = {
        "beam": (180, 180, 210),
        "deck": (210, 180, 180),
        "pier": (180, 210, 180),
        "bridge component unknown": (200, 200, 200),
    }.get(component, (200, 200, 200))
    draw.rectangle([24, 64, 232, 192], fill=component_color)

    if damage == "crack":
        draw.line([48, 128, int(48 + 140 * max(0.2, severity * 10)), 128], fill=(30, 30, 30), width=5)
    elif damage == "spalling":
        radius = int(20 + severity * 800)
        draw.ellipse([96 - radius, 128 - radius, 96 + radius, 128 + radius], fill=(160, 110, 90))
    else:
        size = int(20 + severity * 800)
        draw.rectangle([120, 110, 120 + size, 110 + size], fill=(150, 120, 80))
    image.save(path)


def build_synthetic_dataset(base_dir: Path) -> None:
    rows = []
    splits = ["train", "val", "test"]
    specs = [
        ("crack", "beam", 0.03),
        ("crack", "beam", 0.05),
        ("crack", "deck", 0.04),
        ("spalling", "deck", 0.10),
        ("spalling", "deck", 0.14),
        ("rust", "pier", 0.08),
        ("rust", "pier", 0.12),
        ("crack", "pier", 0.06),
        ("spalling", "beam", 0.11),
    ]
    for split in splits:
        for idx, (damage, component, severity) in enumerate(specs):
            image_path = base_dir / "images" / split / f"{split}_{idx:03d}.png"
            make_sample(image_path, damage, component, severity)
            rows.append(
                {
                    "sample_id": f"{split}_{idx:03d}",
                    "image_path": str(image_path.resolve()),
                    "crop_path": str(image_path.resolve()),
                    "damage_class": damage,
                    "component_class": component,
                    "severity_score": severity,
                    "split": split,
                }
            )
    df = pd.DataFrame(rows)
    out_dir = base_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        df[df["split"] == split].to_csv(out_dir / f"dacl10k_retrieval_{split}.csv", index=False)


def main() -> None:
    work_dir = ROOT / "tmp_smoke"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    build_synthetic_dataset(work_dir)

    config_path = ROOT / "configs" / "bridge_retrieval.yaml"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("data/processed/dacl10k_retrieval_train.csv", str((work_dir / "processed" / "dacl10k_retrieval_train.csv").resolve()))
    text = text.replace("data/processed/dacl10k_retrieval_val.csv", str((work_dir / "processed" / "dacl10k_retrieval_val.csv").resolve()))
    text = text.replace("data/processed/dacl10k_retrieval_test.csv", str((work_dir / "processed" / "dacl10k_retrieval_test.csv").resolve()))
    text = text.replace("epochs: 5", "epochs: 1")
    text = text.replace("batch_size: 16", "batch_size: 4")
    text = text.replace("save_dir: outputs/checkpoints", f"save_dir: {str((work_dir / 'checkpoints').resolve()).replace(chr(92), '/')}")
    text = text.replace("metrics_dir: outputs/metrics", f"metrics_dir: {str((work_dir / 'metrics').resolve()).replace(chr(92), '/')}")
    temp_config = work_dir / "smoke_config.yaml"
    temp_config.write_text(text, encoding="utf-8")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "train_retrieval.py"), "--config", str(temp_config)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_retrieval.py"), "--config", str(temp_config)], check=True)


if __name__ == "__main__":
    main()
