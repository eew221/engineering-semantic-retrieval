from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zeroshot-damage", type=Path, required=True)
    parser.add_argument("--full-damage", type=Path, required=True)
    parser.add_argument("--zeroshot-component", type=Path, required=True)
    parser.add_argument("--full-component", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    panels = [
        ("(a) Zero-shot CLIP, colored by damage", load_rgb(args.zeroshot_damage)),
        ("(b) Proposed model, colored by damage", load_rgb(args.full_damage)),
        ("(c) Zero-shot CLIP, colored by component", load_rgb(args.zeroshot_component)),
        ("(d) Proposed model, colored by component", load_rgb(args.full_component)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
