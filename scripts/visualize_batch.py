"""
Visualize a single batch: show composite, mask, and overlay side-by-side.

Usage:
  python scripts/visualize_batch.py output/demo/batch_0000
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def visualize(batch_dir: Path, save: bool = True) -> Image.Image:
    composite = Image.open(batch_dir / "composite.png").convert("RGB")
    mask      = Image.open(batch_dir / "mask.png").convert("L")

    with open(batch_dir / "metadata.json") as f:
        meta = json.load(f)

    W, H = composite.size

    # Overlay: tint AI region red on the composite
    overlay = composite.copy()
    mask_arr = np.array(mask)
    over_arr = np.array(overlay).copy()
    # Tint AI region with semi-transparent red
    ai_pixels = mask_arr == 255
    over_arr[ai_pixels, 0] = np.clip(over_arr[ai_pixels, 0].astype(int) + 80, 0, 255)
    over_arr[ai_pixels, 1] = np.clip(over_arr[ai_pixels, 1].astype(int) - 40, 0, 255)
    over_arr[ai_pixels, 2] = np.clip(over_arr[ai_pixels, 2].astype(int) - 40, 0, 255)
    overlay = Image.fromarray(over_arr, "RGB")

    # Convert mask to RGB for display
    mask_rgb = Image.merge("RGB", [mask, Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)])

    # Compose 3-panel image
    padding = 10
    panel_w = W
    panel_h = H + 30  # label space
    total_w = panel_w * 3 + padding * 4
    total_h = panel_h + padding * 2

    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))

    labels = ["Composite", "Mask (red=AI)", "Overlay (AI tinted red)"]
    panels = [composite, mask_rgb, overlay]

    for idx, (panel, label) in enumerate(zip(panels, labels)):
        x = padding + idx * (panel_w + padding)
        y = padding
        canvas.paste(panel, (x, y + 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((x, y + 5), label, fill=(220, 220, 220))

    # Metadata text
    draw = ImageDraw.Draw(canvas)
    ai_role = "base" if meta["ai_is_base"] else "patch"
    info = (f"AI role: {ai_role}  |  alpha: {meta['alpha']:.2f}  |  "
            f"patch: {meta['patch_size']}  |  "
            f"real_src: {meta['real_source']}  |  ai_src: {meta['ai_source']}")
    draw.text((padding, total_h - 18), info, fill=(180, 180, 180))

    if save:
        out_path = batch_dir / "visualization.png"
        canvas.save(out_path)
        print(f"Saved visualization to: {out_path}")

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    visualize(args.batch_dir)


if __name__ == "__main__":
    main()
