"""Optional: render generated uppercase text using image glyphs indexed by english.csv.

The CSV supplied for this task indexes isolated handwritten letters, so it is
used here for visual rendering, not for language-model training.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageOps


def load_glyphs(csv_path: Path, image_root: Path) -> dict[str, list[Path]]:
    glyphs: dict[str, list[Path]] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            path = image_root / Path(row["image"]).name
            if path.exists():
                glyphs.setdefault(row["label"].upper(), []).append(path)
    return glyphs


def glyph_image(path: Path, height: int) -> Image.Image:
    image = Image.open(path).convert("L")
    image = ImageOps.invert(image) if image.getextrema()[0] > 120 else image
    image.thumbnail((height, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (image.width + 8, height), 255)
    canvas.paste(image, (4, (height - image.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", default="generated_text.txt")
    parser.add_argument("--csv", required=True, help="Path to english.csv")
    parser.add_argument("--image_root", required=True, help="Folder containing the images in the CSV")
    parser.add_argument("--output", default="handwritten_output.png")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--glyph_height", type=int, default=58)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)
    glyphs = load_glyphs(Path(args.csv), Path(args.image_root))
    if not glyphs:
        raise FileNotFoundError("No glyph images were found. Check --image_root and the CSV paths.")

    text = Path(args.text_file).read_text(encoding="utf-8").upper()
    margin, line_gap = 35, 18
    lines, current = [], []
    current_width = 0
    for char in text:
        if char == "\n":
            lines.append(current); current, current_width = [], 0; continue
        if char == " ":
            item, item_width = None, args.glyph_height // 2
        elif char in glyphs:
            item = glyph_image(random.choice(glyphs[char]), args.glyph_height)
            item_width = item.width
        else:
            continue
        if current and current_width + item_width > args.width - 2 * margin:
            lines.append(current); current, current_width = [], 0
        current.append(item); current_width += item_width
    if current:
        lines.append(current)

    canvas = Image.new("L", (args.width, margin * 2 + len(lines) * (args.glyph_height + line_gap)), 255)
    y = margin
    for line in lines:
        x = margin
        for item in line:
            if item is not None:
                canvas.paste(item, (x, y))
                x += item.width
            else:
                x += args.glyph_height // 2
        y += args.glyph_height + line_gap
    canvas.convert("RGB").save(args.output)
    print(f"Saved handwritten image to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
