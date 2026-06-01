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


def _draw_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    tile_size: tuple[int, int],
    image_path: str,
    title: str,
    lines: Sequence[str],
) -> None:
    image_h = tile_size[1]
    text_h = 108
    tile = _open_or_blank(image_path, (tile_size[0], image_h))
    canvas.paste(tile, (x, y))
    draw.rectangle((x, y, x + tile_size[0], y + image_h), outline=(40, 40, 40), width=2)
    draw.text((x + 6, y + image_h + 4), title, fill=(0, 0, 0))
    for idx, line in enumerate(lines[:5]):
        draw.text((x + 6, y + image_h + 24 + idx * 16), line, fill=(60, 60, 60))


def save_retrieval_grid(
    query_path: str,
    before_paths: Sequence[str],
    after_paths: Sequence[str],
    out_path: str | Path,
    title: str,
    query_lines: Sequence[str] | None = None,
    before_titles: Sequence[str] | None = None,
    after_titles: Sequence[str] | None = None,
    before_lines: Sequence[Sequence[str]] | None = None,
    after_lines: Sequence[Sequence[str]] | None = None,
    tile_size: tuple[int, int] = (180, 180),
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    margin = 16
    cols = 1 + max(len(before_paths), len(after_paths))
    width = cols * tile_size[0] + (cols + 1) * margin
    tile_total_h = tile_size[1] + 108
    height = 3 * tile_total_h + 5 * margin + 40
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(0, 0, 0))

    query_y = margin * 2 + 22
    before_y = query_y + tile_total_h + margin * 2
    after_y = before_y + tile_total_h + margin * 2

    draw.text((margin, query_y - 18), "Query", fill=(0, 0, 0))
    _draw_tile(
        canvas,
        draw,
        margin,
        query_y,
        tile_size,
        query_path,
        "query",
        list(query_lines or []),
    )

    draw.text((margin, before_y - 18), "Before rerank", fill=(0, 0, 0))
    for i, path in enumerate(before_paths[: cols - 1], start=1):
        x = margin + i * (tile_size[0] + margin)
        _draw_tile(
            canvas,
            draw,
            x,
            before_y,
            tile_size,
            path,
            (before_titles[i - 1] if before_titles is not None and i - 1 < len(before_titles) else f"top-{i}"),
            list(before_lines[i - 1]) if before_lines is not None and i - 1 < len(before_lines) else [],
        )

    draw.text((margin, after_y - 18), "After rerank", fill=(0, 0, 0))
    for i, path in enumerate(after_paths[: cols - 1], start=1):
        x = margin + i * (tile_size[0] + margin)
        _draw_tile(
            canvas,
            draw,
            x,
            after_y,
            tile_size,
            path,
            (after_titles[i - 1] if after_titles is not None and i - 1 < len(after_titles) else f"top-{i}"),
            list(after_lines[i - 1]) if after_lines is not None and i - 1 < len(after_lines) else [],
        )

    canvas.save(out_path)
