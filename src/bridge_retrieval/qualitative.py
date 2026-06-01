"""Qualitative retrieval visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


def _open_or_blank(path: str, size: tuple[int, int]) -> Image.Image:
    try:
        return Image.open(path).convert("RGB").resize(size)
    except Exception:
        return Image.new("RGB", size, color=(240, 240, 240))


def save_retrieval_grid(
    query_path: str,
    before_paths: Sequence[str],
    after_paths: Sequence[str],
    out_path: str | Path,
    title: str,
    tile_size: tuple[int, int] = (180, 180),
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    margin = 16
    cols = 1 + max(len(before_paths), len(after_paths))
    width = cols * tile_size[0] + (cols + 1) * margin
    height = 3 * tile_size[1] + 5 * margin + 40
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(0, 0, 0))

    rows = [
        ("Query", [query_path]),
        ("Before rerank", list(before_paths)),
        ("After rerank", list(after_paths)),
    ]

    for row_idx, (label, paths) in enumerate(rows):
        y = margin * (row_idx + 2) + row_idx * tile_size[1]
        draw.text((margin, y - 18), label, fill=(0, 0, 0))
        if row_idx == 0:
            tile = _open_or_blank(query_path, tile_size)
            canvas.paste(tile, (margin, y))
        else:
            for i, path in enumerate(paths[: cols - 1], start=1):
                x = margin + i * (tile_size[0] + margin)
                tile = _open_or_blank(path, tile_size)
                canvas.paste(tile, (x, y))

    canvas.save(out_path)
