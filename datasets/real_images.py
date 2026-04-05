"""
Real image provider.

Priority order
--------------
1. Load from DATA_DIR/real/ (user has placed downloaded images there).
2. Fall back to procedural generation: photos-like textures using
   Perlin-style noise + natural colour palettes, plus sketch-style images.

This module is intentionally dataset-agnostic: drop any JPEG/PNG files into
data/real/ (COCO, Open Images, Quick Draw PNG bitmaps, etc.) and the loader
will pick them up automatically.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import REAL_DIR, OUTPUT_SIZE


# ── Disk loader ──────────────────────────────────────────────────────────────

_IGNORED_DIR_NAMES = {
    "gt", "gts", "mask", "masks", "label", "labels",
    "annotation", "annotations", "quickdraw_png",
}

def _collect_paths(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [
        p for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and not any(part.lower() in _IGNORED_DIR_NAMES for part in p.parts)
    ]


def _resize_to_canvas(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale-to-fill then center-crop to `size`. No padding."""
    from PIL import ImageOps
    if img.size[0] == 0 or img.size[1] == 0:
        return Image.new("RGB", size, (0, 0, 0))
    return ImageOps.fit(img, size, Image.LANCZOS)


def load_real_image_from_disk(rng: random.Random) -> Image.Image | None:
    """Return a random real image from REAL_DIR, or None if directory empty."""
    paths = _collect_paths(REAL_DIR)
    if not paths:
        return None
    path = rng.choice(paths)
    img = Image.open(path).convert("RGB")
    img = _resize_to_canvas(img, OUTPUT_SIZE)
    return img


def load_real_image_with_meta(rng: random.Random) -> tuple[Image.Image, str, Path | None] | None:
    """
    Return (resized_image, dataset_name, original_path) from REAL_DIR.
    dataset_name is the subdirectory name directly under REAL_DIR (e.g. 'coco', 'div2k').
    original_path is the raw file path before resizing (for copying as source.png).
    Returns None if REAL_DIR is empty.
    """
    paths = _collect_paths(REAL_DIR)
    if not paths:
        return None
    path = rng.choice(paths)
    img = Image.open(path).convert("RGB")
    resized = _resize_to_canvas(img, OUTPUT_SIZE)

    # Derive dataset name: first component of path relative to REAL_DIR
    try:
        rel = path.relative_to(REAL_DIR)
        dataset_name = rel.parts[0] if rel.parts else "unknown"
    except ValueError:
        dataset_name = "unknown"

    return resized, dataset_name, path


# ── Procedural photo-like generator ─────────────────────────────────────────

def _smooth_noise(w: int, h: int, scale: float, rng: random.Random) -> np.ndarray:
    """Simple value-noise texture (fast, no scipy needed)."""
    grid_w = max(2, int(w / scale))
    grid_h = max(2, int(h / scale))
    grid = np.array([[rng.random() for _ in range(grid_w + 1)]
                     for _ in range(grid_h + 1)], dtype=np.float32)

    xs = np.linspace(0, grid_w, w)
    ys = np.linspace(0, grid_h, h)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, grid_w)
    y1 = np.minimum(y0 + 1, grid_h)
    tx = xs - x0
    ty = (ys - y0)[:, None]
    tx = tx[None, :]

    # Smoothstep
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)

    n = (grid[y0][:, x0] * (1 - tx) * (1 - ty)
         + grid[y0][:, x1] * tx * (1 - ty)
         + grid[y1][:, x0] * (1 - tx) * ty
         + grid[y1][:, x1] * tx * ty)
    return n


_NATURE_PALETTES = [
    # (base_rgb, noise_rgb)  — landscape / nature photos
    ((34, 85, 34),   (10, 40, 10)),    # forest green
    ((70, 130, 180), (20, 50, 80)),    # sky blue
    ((180, 160, 120),(40, 30, 20)),    # sandy earth
    ((120, 80, 60),  (30, 20, 15)),    # dark soil
    ((200, 200, 215),(30, 30, 30)),    # overcast sky
    ((100, 140, 80), (20, 30, 15)),    # meadow
]

_URBAN_PALETTES = [
    ((160, 160, 165),(20, 20, 20)),    # concrete
    ((80, 70, 60),   (20, 15, 10)),    # asphalt
    ((200, 180, 150),(30, 25, 20)),    # brick
]


def _make_photo_like(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Generate a plausible natural-scene texture."""
    w, h = size
    palettes = _NATURE_PALETTES + _URBAN_PALETTES
    base_col, noise_col = rng.choice(palettes)

    # Combine multiple octaves of noise
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    for scale in [64, 32, 16, 8]:
        noise += amplitude * _smooth_noise(w, h, scale, rng)
        total += amplitude
        amplitude *= 0.5
    noise /= total  # [0, 1]

    r = np.clip(base_col[0] + noise_col[0] * noise, 0, 255).astype(np.uint8)
    g = np.clip(base_col[1] + noise_col[1] * noise, 0, 255).astype(np.uint8)
    b = np.clip(base_col[2] + noise_col[2] * noise, 0, 255).astype(np.uint8)

    arr = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(arr, "RGB")

    # Light directional gradient to simulate illumination
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(h):
        alpha = int(30 * math.sin(math.pi * i / h))
        draw.line([(0, i), (w, i)], fill=(255, 255, 200, max(0, alpha)))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Slight blur to simulate depth of field
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
    return img


def _make_sketch_like(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Generate a human-sketch-style image (simple line drawing)."""
    w, h = size
    bg_grey = rng.randint(230, 255)
    img = Image.new("RGB", (w, h), (bg_grey, bg_grey, bg_grey))
    draw = ImageDraw.Draw(img)

    stroke_col = (rng.randint(0, 60), rng.randint(0, 60), rng.randint(0, 60))

    # Draw 5–15 random organic strokes (Bezier-like via many small segments)
    n_strokes = rng.randint(5, 15)
    for _ in range(n_strokes):
        pts = [(rng.randint(0, w), rng.randint(0, h)) for _ in range(rng.randint(3, 8))]
        width = rng.randint(1, 5)
        draw.line(pts, fill=stroke_col, width=width, joint="curve")

    # Simple geometric shapes (triangle, circle, rectangle) as sketch elements
    n_shapes = rng.randint(2, 6)
    for _ in range(n_shapes):
        shape = rng.choice(["ellipse", "rectangle"])
        x0, y0 = rng.randint(0, w - 50), rng.randint(0, h - 50)
        x1, y1 = x0 + rng.randint(20, min(150, w - x0)), y0 + rng.randint(20, min(150, h - y0))
        lw = rng.randint(1, 3)
        if shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], outline=stroke_col, width=lw)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=stroke_col, width=lw)

    # Slight noise to mimic paper texture
    arr = np.array(img, dtype=np.float32)
    arr += rng.gauss(0, 3) * np.random.default_rng(rng.randint(0, 2**32)).standard_normal(arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ── Public API ────────────────────────────────────────────────────────────────

def get_real_image(rng: random.Random) -> tuple[Image.Image, str]:
    """
    Return (image, source_tag).
    source_tag is 'disk' if loaded from REAL_DIR, otherwise 'synthetic_photo'
    or 'synthetic_sketch'.
    """
    img = load_real_image_from_disk(rng)
    if img is not None:
        return img, "disk"

    # Offline fallback
    style = rng.choice(["photo", "photo", "sketch"])  # 2:1 ratio
    if style == "photo":
        return _make_photo_like(OUTPUT_SIZE, rng), "synthetic_photo"
    else:
        return _make_sketch_like(OUTPUT_SIZE, rng), "synthetic_sketch"


def real_image_stream(seed: int | None = None) -> Iterator[tuple[Image.Image, str]]:
    """Infinite iterator of real images."""
    rng = random.Random(seed)
    while True:
        yield get_real_image(rng)
