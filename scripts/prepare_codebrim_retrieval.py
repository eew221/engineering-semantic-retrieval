"""Prepare a retrieval CSV from CODEBRIM labels.

This script assumes a simple metadata CSV with at least:
- image_path
- damage_class

Optional:
- component_class
- severity_score
- split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.bridge_retrieval.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/codebrim_retrieval.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    required = {"image_path", "damage_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "sample_id" not in df.columns:
        df["sample_id"] = [f"codebrim_{i:06d}" for i in range(len(df))]
    if "crop_path" not in df.columns:
        df["crop_path"] = df["image_path"]
    if "component_class" not in df.columns:
        df["component_class"] = "bridge component unknown"
    if "severity_score" not in df.columns:
        df["severity_score"] = 0.0
    if "split" not in df.columns:
        df["split"] = "train"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved normalized CODEBRIM retrieval CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
