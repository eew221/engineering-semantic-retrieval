from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-files", nargs="+", type=Path, required=True)
    parser.add_argument("--lambda-values", nargs="+", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.metric_files) != len(args.lambda_values):
        raise ValueError("metric-files and lambda-values must have the same length")

    rows = []
    for lam, metric_path in zip(args.lambda_values, args.metric_files):
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        base = payload["base"]
        rows.append(
            {
                "lambda_t": lam,
                "mAP": base["mAP"],
                "Recall@1": base["Recall@1"],
                "Recall@5": base["Recall@5"],
                "NDCG@5": base["NDCG@5"],
                "metric_file": str(metric_path),
            }
        )

    rows.sort(key=lambda x: x["lambda_t"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lambdas = [row["lambda_t"] for row in rows]
    maps = [row["mAP"] for row in rows]
    r1 = [row["Recall@1"] for row in rows]

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(lambdas, maps, marker="o", linewidth=2.2, label="mAP")
    plt.plot(lambdas, r1, marker="s", linewidth=2.0, label="Recall@1")
    plt.xlabel(r"$\lambda_t$")
    plt.ylabel("Metric")
    plt.title(r"Text-anchor weight sensitivity on dacl10k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=220)
    plt.close()
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
