from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def robust_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p95 = np.percentile(x, 95)
    if p95 <= eps:
        return np.zeros_like(x)
    return np.clip(x / p95, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    crop_w = []
    crop_h = []
    for p in df["crop_path"].tolist():
        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception:
            w, h = 1, 1
        crop_w.append(w)
        crop_h.append(h)

    df["crop_w"] = crop_w
    df["crop_h"] = crop_h
    df["crop_area"] = np.maximum(1, df["crop_w"] * df["crop_h"])
    df["severity_score_v1"] = df["severity_score"].astype(float)
    df["damage_density"] = df["area_pixels"].astype(float) / df["crop_area"].astype(float)
    df["elongation"] = np.maximum(df["crop_w"], df["crop_h"]) / np.maximum(1, np.minimum(df["crop_w"], df["crop_h"]))

    norm_area = robust_norm(df["severity_score_v1"].to_numpy(dtype=np.float32))
    norm_density = robust_norm(df["damage_density"].to_numpy(dtype=np.float32))
    norm_elongation = robust_norm(df["elongation"].to_numpy(dtype=np.float32))

    crack_mask = df["damage_class"].astype(str).str.contains("crack", case=False, regex=False).to_numpy()
    shape_term = np.where(crack_mask, norm_elongation, norm_density)

    severity_v2 = 0.45 * norm_area + 0.35 * norm_density + 0.20 * shape_term
    df["severity_score"] = severity_v2.astype(np.float32)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows with severity v2 to {args.output_csv}")


if __name__ == "__main__":
    main()
