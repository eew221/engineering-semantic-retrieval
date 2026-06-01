from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CODEBRIM_MAP = {
    "Background": "background",
    "Crack": "crack",
    "Spallation": "spallation",
    "Efflorescence": "efflorescence",
    "ExposedBars": "exposed reinforcement bar",
    "CorrosionStain": "corrosion stain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--single-label-only", action="store_true")
    return parser.parse_args()


def parse_metadata(xml_path: Path) -> dict[str, list[str]]:
    root = ET.parse(xml_path).getroot()
    labels_by_name: dict[str, list[str]] = {}
    for defect in root.findall("Defect"):
        name = defect.attrib["name"]
        active = []
        for child in defect:
            if child.text and child.text.strip() == "1":
                mapped = CODEBRIM_MAP.get(child.tag)
                if mapped:
                    active.append(mapped)
        labels_by_name[name] = active
    return labels_by_name


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    metadata_root = data_root / "metadata"
    split_root = data_root / args.split

    labels_by_name = {}
    for xml_name in ["background.xml", "defects.xml"]:
        xml_path = metadata_root / xml_name
        if xml_path.exists():
            labels_by_name.update(parse_metadata(xml_path))

    rows = []
    for class_dir in sorted(split_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in class_dir.glob("*.png"):
            active = labels_by_name.get(image_path.name, [])
            if not active:
                continue
            non_background = [x for x in active if x != "background"]
            if args.single_label_only:
                if len(non_background) == 1:
                    damage_class = non_background[0]
                elif len(non_background) == 0 and active == ["background"]:
                    damage_class = "background"
                else:
                    continue
            else:
                damage_class = "+".join(sorted(non_background)) if non_background else "background"

            rows.append(
                {
                    "sample_id": f"codebrim_{args.split}_{image_path.stem}",
                    "image_path": str(image_path.resolve()),
                    "crop_path": str(image_path.resolve()),
                    "damage_class": damage_class,
                    "component_class": "bridge component unknown",
                    "severity_score": 0.0,
                    "split": args.split,
                    "label_count": len(non_background) if non_background else 1,
                }
            )

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows to {args.output_csv}")
    if len(df):
        print(df["damage_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
