"""
Convert Quick, Draw! .npy bitmap files to individual PNG images.

Usage:
  python scripts/quickdraw_to_png.py \
      --npy_dir data/real/quickdraw \
      --out_dir data/real/quickdraw_png \
      --max_per_category 2000
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert_npy_to_png(
    npy_dir: Path,
    out_dir: Path,
    max_per_category: int = 2000,
    image_size: int = 256,
) -> None:
    npy_files = list(npy_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {npy_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for npy_path in npy_files:
        category = npy_path.stem
        cat_dir  = out_dir / category
        cat_dir.mkdir(exist_ok=True)

        # Quick Draw bitmaps: shape (N, 784) uint8, values 0-255
        data = np.load(npy_path)
        n    = min(len(data), max_per_category)

        for i in range(n):
            bitmap = data[i].reshape(28, 28)
            # Invert: Quick Draw has white strokes on black bg → invert to
            # look like ink on paper (more natural for mixing with real images)
            bitmap = 255 - bitmap
            img = Image.fromarray(bitmap, "L").resize(
                (image_size, image_size), Image.NEAREST
            ).convert("RGB")
            img.save(cat_dir / f"{category}_{i:05d}.png")
            total += 1

        print(f"  {category}: {n} images → {cat_dir}")

    print(f"Total: {total} Quick Draw images converted.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir",  type=Path, required=True)
    parser.add_argument("--out_dir",  type=Path, required=True)
    parser.add_argument("--max_per_category", type=int, default=2000)
    parser.add_argument("--image_size",       type=int, default=256)
    args = parser.parse_args()

    convert_npy_to_png(
        npy_dir=args.npy_dir,
        out_dir=args.out_dir,
        max_per_category=args.max_per_category,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
