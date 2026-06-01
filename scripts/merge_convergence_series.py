from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-json", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.series_json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} does not contain a list of rows")
        rows.extend(payload)

    dedup = {}
    for row in rows:
        dedup[int(row["epoch"])] = row
    merged = [dedup[k] for k in sorted(dedup)]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    epochs = [row["epoch"] for row in merged]
    maps = [row["mAP"] for row in merged]
    r1 = [row["Recall@1"] for row in merged]

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(epochs, maps, marker="o", linewidth=2.2, label="mAP")
    plt.plot(epochs, r1, marker="s", linewidth=2.0, label="Recall@1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Convergence on dacl10k validation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=220)
    plt.close()
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
