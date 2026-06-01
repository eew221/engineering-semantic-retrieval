from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: list[str], fill: str) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=24, fill=fill, outline="#1f1f1f", width=4)
    title_font = load_font(42)
    body_font = load_font(28)
    draw.text((x0 + 24, y0 + 18), title, fill="#111111", font=title_font)
    y = y0 + 78
    for line in body:
        draw.text((x0 + 30, y), line, fill="#222222", font=body_font)
        y += 38


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#2b5aa6") -> None:
    draw.line([start, end], fill=color, width=8)
    ex, ey = end
    draw.polygon([(ex, ey), (ex - 26, ey - 12), (ex - 26, ey + 12)], fill=color)


def build_method_figure(out_path: Path) -> None:
    width, height = 4200, 1800
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(56)
    subtitle_font = load_font(30)
    draw.text((120, 70), "Engineering-Semantic Bridge Defect Retrieval Pipeline", fill="#0f172a", font=title_font)
    draw.text(
        (120, 145),
        "Public annotations -> retrieval tuple construction -> CLIP-based semantic learning -> in-domain and cross-dataset evaluation",
        fill="#334155",
        font=subtitle_font,
    )

    top_y = 320
    box_w = 850
    box_h = 960
    gap = 140
    xs = [120, 120 + box_w + gap, 120 + 2 * (box_w + gap), 120 + 3 * (box_w + gap)]

    draw_box(
        draw,
        (xs[0], top_y, xs[0] + box_w, top_y + box_h),
        "1. Public Data Inputs",
        [
            "dacl10k bridge images",
            "damage masks",
            "bridge component masks",
            "CODEBRIM defect crops",
            "no manual report corpus needed",
        ],
        "#e9f2ff",
    )
    draw_box(
        draw,
        (xs[1], top_y, xs[1] + box_w, top_y + box_h),
        "2. Retrieval Tuple Construction",
        [
            "instance-centered crop extraction",
            "component assignment by overlap",
            "severity proxy from mask area",
            "tuple = (crop, damage, component, severity)",
            "external CODEBRIM defect-only subset",
        ],
        "#edf7ed",
    )
    draw_box(
        draw,
        (xs[2], top_y, xs[2] + box_w, top_y + box_h),
        "3. CLIP-Based Training",
        [
            "image encoder: CLIP ViT-B/32",
            "text anchors: 'crack on beam'",
            "weighted pair supervision",
            "severity regression head",
            "engineering-semantic embedding space",
        ],
        "#fff4e5",
    )
    draw_box(
        draw,
        (xs[3], top_y, xs[3] + box_w, top_y + box_h),
        "4. Retrieval Evaluation",
        [
            "dacl10k in-domain retrieval",
            "CODEBRIM cross-dataset retrieval",
            "mAP, Recall@K, NDCG@K",
            "component / severity consistency",
            "qualitative ranking analysis",
        ],
        "#f7ecff",
    )

    centers = [(x + box_w, top_y + box_h // 2) for x in xs[:-1]]
    next_starts = [(x, top_y + box_h // 2) for x in xs[1:]]
    for start, end in zip(centers, next_starts):
        draw_arrow(draw, (start[0] + 10, start[1]), (end[0] - 30, end[1]))

    note_font = load_font(28)
    draw.rounded_rectangle((140, 1400, 4050, 1670), radius=24, fill="#f8fafc", outline="#cbd5e1", width=3)
    draw.text(
        (180, 1440),
        "Key design idea: retrieval relevance is not binary visual similarity. It is graded by defect agreement, "
        "component agreement, and severity proximity, then transferred into pair weights and evaluation relevance.",
        fill="#0f172a",
        font=note_font,
    )
    draw.text(
        (180, 1515),
        "Main empirical pattern: engineering-semantic adaptation improves in-domain mAP on dacl10k, "
        "but zero-shot CLIP remains stronger on external CODEBRIM retrieval.",
        fill="#0f172a",
        font=note_font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(600, 600))


def build_panel_figure(src_dir: Path, filenames: list[str], labels: list[str], out_path: Path, title: str) -> None:
    images = [Image.open(src_dir / name).convert("RGB") for name in filenames]
    panel_w = 1192
    panel_h = 660
    margin = 50
    title_h = 120
    label_h = 60
    cols = len(images)
    width = cols * panel_w + (cols + 1) * margin
    height = title_h + label_h + panel_h + 2 * margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(38)
    label_font = load_font(30)

    draw.text((margin, 28), title, fill="#0f172a", font=title_font)

    for idx, (img, label) in enumerate(zip(images, labels)):
        x = margin + idx * (panel_w + margin)
        y = title_h + label_h
        canvas.paste(ImageOps.contain(img, (panel_w, panel_h)), (x, y))
        draw.text((x, title_h), f"({chr(97 + idx)}) {label}", fill="#111111", font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(600, 600))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir

    build_method_figure(out_dir / "Fig1_method_overview.png")

    build_panel_figure(
        ROOT / "outputs" / "qualitative_dacl10k",
        filenames=["retrieval_query_0002.png", "retrieval_query_0022.png", "retrieval_query_0026.png"],
        labels=[
            "Graffiti-like query with limited reranking gain",
            "Small defect query with visually ambiguous neighbors",
            "Bearing-region query with stronger structural context",
        ],
        out_path=out_dir / "Fig2_dacl10k_qualitative.png",
        title="Qualitative retrieval examples on dacl10k",
    )

    build_panel_figure(
        ROOT / "outputs" / "qualitative_codebrim",
        filenames=["retrieval_query_0009.png", "retrieval_query_0027.png", "retrieval_query_0037.png"],
        labels=[
            "Texture-dominated concrete surface query",
            "Surface pattern shift under external imagery",
            "Cross-dataset mismatch with broad background textures",
        ],
        out_path=out_dir / "Fig3_codebrim_qualitative.png",
        title="Cross-dataset qualitative retrieval examples on CODEBRIM",
    )

    print(f"Saved submission figure assets to {out_dir}")


if __name__ == "__main__":
    main()
