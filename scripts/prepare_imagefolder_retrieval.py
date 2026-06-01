from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="test")
    parser.add_argument("--class-map", type=str, default="")
    parser.add_argument("--suffixes", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_map = {}
    if args.class_map:
        for item in args.class_map.split(","):
            src, dst = item.split("=", 1)
            class_map[src.strip()] = dst.strip()

    rows = []
    for class_dir in sorted([p for p in args.data_root.iterdir() if p.is_dir()]):
        class_name = class_map.get(class_dir.name, class_dir.name)
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in {x.lower() for x in args.suffixes}:
                continue
            rows.append(
                {
                    "sample_id": f"{args.split_name}_{class_dir.name}_{image_path.stem}",
                    "image_path": str(image_path.resolve()),
                    "crop_path": str(image_path.resolve()),
                    "damage_class": class_name,
                    "component_class": "bridge component unknown",
                    "severity_score": 0.0,
                    "split": args.split_name,
                }
            )

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows to {args.output_csv}")
    if len(df):
        print(df['damage_class'].value_counts().to_string())


if __name__ == "__main__":
    main()
