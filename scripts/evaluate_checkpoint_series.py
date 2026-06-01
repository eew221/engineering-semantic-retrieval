from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_retrieval.utils import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--rerank", action="store_true")
    return parser.parse_args()


def epoch_from_name(path: Path) -> int:
    stem = path.stem
    if "_epoch" not in stem:
        return 10**9
    return int(stem.rsplit("_epoch", 1)[1])


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    metrics_dir = ROOT / cfg["output"]["metrics_dir"]
    checkpoints = sorted(args.checkpoint_dir.glob(args.pattern), key=epoch_from_name)
    rows = []

    for ckpt in checkpoints:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_retrieval.py"),
            "--config",
            str(args.config),
            "--checkpoint",
            str(ckpt),
        ]
        if args.rerank:
            cmd.append("--rerank")
        subprocess.run(cmd, check=True, cwd=ROOT)

        metrics_path = metrics_dir / f"{cfg['experiment_name']}_test_metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        base = payload["base"]
        rows.append(
            {
                "epoch": epoch_from_name(ckpt),
                "checkpoint": str(ckpt),
                "mAP": base["mAP"],
                "Recall@1": base["Recall@1"],
                "Recall@5": base["Recall@5"],
                "NDCG@5": base["NDCG@5"],
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    epochs = [row["epoch"] for row in rows]
    maps = [row["mAP"] for row in rows]
    r1 = [row["Recall@1"] for row in rows]

    ensure_dir(args.output_png.parent)
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
