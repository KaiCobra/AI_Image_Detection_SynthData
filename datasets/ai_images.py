"""
AI-generated image provider — Synthbuster dataset with prompts.csv.

Reads prompts.csv to get (filename, caption) pairs, then loads the
corresponding AI image from a randomly chosen generator subdirectory.

Fallback: procedural generation if no dataset is available.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from config import AI_DIR, OUTPUT_SIZE


# ── Synthbuster loader ──────────────────────────────────────────────────────

SYNTHBUSTER_ROOT = AI_DIR / "synthbuster" / "synthbuster"
PROMPTS_CSV = SYNTHBUSTER_ROOT / "prompts.csv"

GENERATORS = [
    "dalle2", "dalle3", "firefly", "glide", "midjourney-v5",
    "stable-diffusion-1-3", "stable-diffusion-1-4",
    "stable-diffusion-2", "stable-diffusion-xl",
]

_prompt_rows: list[dict] | None = None


def _load_prompts() -> list[dict]:
    """Load prompts.csv once, return list of {filename, caption} dicts."""
    global _prompt_rows
    if _prompt_rows is not None:
        return _prompt_rows

    if not PROMPTS_CSV.exists():
        _prompt_rows = []
        return _prompt_rows

    rows = []
    with open(PROMPTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            rows.append({
                "filename": row[0].strip(),
                "caption": row[1].strip(),
            })
    _prompt_rows = rows
    return _prompt_rows


def get_synthbuster_image(
    rng: random.Random,
) -> tuple[Image.Image, str, str, str] | None:
    """
    Pick a random (image, generator, filename, caption) from Synthbuster.

    Returns None if prompts.csv is missing or dataset not downloaded.
    """
    rows = _load_prompts()
    if not rows:
        return None

    row = rng.choice(rows)
    filename = row["filename"]
    caption = row["caption"]

    # Pick a random generator
    available = [g for g in GENERATORS if (SYNTHBUSTER_ROOT / g / f"{filename}.png").exists()]
    if not available:
        return None

    generator = rng.choice(available)
    image_path = SYNTHBUSTER_ROOT / generator / f"{filename}.png"
    img = Image.open(image_path).convert("RGB")

    source_tag = f"synthbuster/{generator}"
    return img, source_tag, filename, caption


def get_synthbuster_image_by_index(
    index: int,
    rng: random.Random,
) -> tuple[Image.Image, str, str, str] | None:
    """
    Get the i-th row from prompts.csv (wraps around if index > len).

    Returns (image, source_tag, filename, caption) or None.
    """
    rows = _load_prompts()
    if not rows:
        return None

    row = rows[index % len(rows)]
    filename = row["filename"]
    caption = row["caption"]

    available = [g for g in GENERATORS if (SYNTHBUSTER_ROOT / g / f"{filename}.png").exists()]
    if not available:
        return None

    generator = rng.choice(available)
    image_path = SYNTHBUSTER_ROOT / generator / f"{filename}.png"
    img = Image.open(image_path).convert("RGB")

    source_tag = f"synthbuster/{generator}"
    return img, source_tag, filename, caption


def get_synthbuster_image_path_by_index(
    index: int,
    rng: random.Random,
) -> tuple[Path, str, str, str] | None:
    """
    Same as get_synthbuster_image_by_index but returns the file path
    instead of loading the image. Useful for passing to SAM3 server.
    """
    rows = _load_prompts()
    if not rows:
        return None

    row = rows[index % len(rows)]
    filename = row["filename"]
    caption = row["caption"]

    available = [g for g in GENERATORS if (SYNTHBUSTER_ROOT / g / f"{filename}.png").exists()]
    if not available:
        return None

    generator = rng.choice(available)
    image_path = SYNTHBUSTER_ROOT / generator / f"{filename}.png"
    source_tag = f"synthbuster/{generator}"
    return image_path, source_tag, filename, caption


def synthbuster_count() -> int:
    """Number of rows in prompts.csv."""
    return len(_load_prompts())


# ── Disk loader (legacy fallback) ───────────────────────────────────────────

_IGNORED_DIR_NAMES = {"gt", "gts", "mask", "masks", "label", "labels", "annotation", "annotations"}

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


def load_ai_image_from_disk(rng: random.Random) -> Image.Image | None:
    paths = _collect_paths(AI_DIR)
    if not paths:
        return None
    path = rng.choice(paths)
    img = Image.open(path).convert("RGB")
    img = _resize_to_canvas(img, OUTPUT_SIZE)
    return img


# ── Procedural AI-artefact generator ─────────────────────────────────────────

_AI_PALETTES = [
    ((255, 100, 150), (100, 80, 220), (50, 200, 255)),
    ((255, 180, 50),  (220, 50, 100), (80, 30, 180)),
    ((0, 200, 180),   (0, 80, 220),   (180, 0, 255)),
    ((255, 220, 200), (200, 150, 100),(120, 80, 60)),
    ((150, 220, 255), (80, 160, 200), (40, 80, 160)),
    ((200, 255, 200), (80, 200, 120), (20, 120, 80)),
]


def _radial_gradient(size, center, col_a, col_b):
    w, h = size
    cx, cy = center[0] * w, center[1] * h
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    dist /= (dist.max() + 1e-8)
    arr = np.stack([
        col_a[c] * (1 - dist) + col_b[c] * dist for c in range(3)
    ], axis=-1).astype(np.uint8)
    return arr


def _make_smooth_gradient(size, rng):
    w, h = size
    col_a, col_b, col_c = rng.choice(_AI_PALETTES)
    cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    arr1 = _radial_gradient(size, (cx, cy), col_a, col_b)
    arr2 = _radial_gradient(size, (1 - cx, 1 - cy), col_b, col_c)
    mix = rng.uniform(0.3, 0.7)
    arr = (arr1 * mix + arr2 * (1 - mix)).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2, 5)))
    img = ImageEnhance.Color(img).enhance(rng.uniform(1.5, 2.5))
    return img


def _make_mandala(size, rng):
    w, h = size
    col_a, col_b, col_c = rng.choice(_AI_PALETTES)
    bg = Image.new("RGB", (w, h), col_b)
    draw = ImageDraw.Draw(bg)
    cx, cy = w // 2, h // 2
    n_rings = rng.randint(4, 10)
    n_arms = rng.randint(6, 16)
    for ring in range(n_rings, 0, -1):
        r = int(min(w, h) * ring / (2 * n_rings))
        t = ring / n_rings
        col = tuple(int(col_a[c] * t + col_c[c] * (1 - t)) for c in range(3))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)
    for arm in range(n_arms):
        angle = 2 * math.pi * arm / n_arms
        for seg in range(1, 8):
            r0 = int(min(w, h) * (seg - 1) / 16)
            r1 = int(min(w, h) * seg / 16)
            x0 = cx + int(r0 * math.cos(angle))
            y0 = cy + int(r0 * math.sin(angle))
            x1 = cx + int(r1 * math.cos(angle))
            y1 = cy + int(r1 * math.sin(angle))
            t = seg / 8
            col = tuple(int(col_a[c] * t + col_b[c] * (1 - t)) for c in range(3))
            draw.line([x0, y0, x1, y1], fill=col, width=rng.randint(1, 4))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))
    bg = ImageEnhance.Color(bg).enhance(rng.uniform(1.3, 2.0))
    return bg


def _make_dreamscape(size, rng):
    w, h = size
    col_a, col_b, col_c = rng.choice(_AI_PALETTES)
    img = Image.new("RGB", (w, h), col_a)
    draw = ImageDraw.Draw(img)
    sky_h = int(h * rng.uniform(0.3, 0.6))
    draw.rectangle([0, 0, w, sky_h], fill=col_b)
    sun_r = rng.randint(30, 80)
    sx = rng.randint(sun_r, w - sun_r)
    sy = rng.randint(sun_r, sky_h)
    draw.ellipse([sx - sun_r, sy - sun_r, sx + sun_r, sy + sun_r], fill=col_c)
    horizon_pts = []
    prev_y = sky_h
    for x in range(0, w + 10, 10):
        prev_y += rng.randint(-8, 8)
        prev_y = max(sky_h - 40, min(sky_h + 40, prev_y))
        horizon_pts.append((x, prev_y))
    horizon_pts += [(w, h), (0, h)]
    draw.polygon(horizon_pts, fill=col_a)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3, 8)))
    img = ImageEnhance.Color(img).enhance(rng.uniform(1.4, 2.2))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(1.05, 1.25))
    return img


def _make_portrait_like(size, rng):
    w, h = size
    col_a, col_b, _ = rng.choice([
        ((240, 200, 170), (180, 140, 110), (100, 80, 60)),
        ((200, 180, 220), (140, 120, 180), (80, 60, 120)),
    ])
    bg_col = (col_b[0] // 2, col_b[1] // 2, col_b[2] // 2)
    img = Image.new("RGB", (w, h), bg_col)
    draw = ImageDraw.Draw(img)
    face_w = int(w * rng.uniform(0.35, 0.55))
    face_h = int(h * rng.uniform(0.45, 0.65))
    fx = (w - face_w) // 2
    fy = (h - face_h) // 2
    draw.ellipse([fx, fy, fx + face_w, fy + face_h], fill=col_a)
    eye_y = fy + int(face_h * 0.35)
    eye_r = int(face_w * 0.08)
    left_x = fx + int(face_w * 0.3)
    right_x = fx + int(face_w * 0.7)
    dark = (col_a[0] // 3, col_a[1] // 3, col_a[2] // 3)
    draw.ellipse([left_x - eye_r, eye_y - eye_r, left_x + eye_r, eye_y + eye_r], fill=dark)
    draw.ellipse([right_x - eye_r, eye_y - eye_r, right_x + eye_r, eye_y + eye_r], fill=dark)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2, 5)))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.3))
    return img


_GENERATORS_PROC = [
    _make_smooth_gradient,
    _make_mandala,
    _make_dreamscape,
    _make_portrait_like,
]


# ── Public API ────────────────────────────────────────────────────────────────

def get_ai_image(rng: random.Random) -> tuple[Image.Image, str]:
    """
    Return (image, source_tag). Legacy API — no caption.
    """
    img = load_ai_image_from_disk(rng)
    if img is not None:
        return img, "disk"
    gen = rng.choice(_GENERATORS_PROC)
    img = gen(OUTPUT_SIZE, rng)
    return img, f"synthetic_{gen.__name__}"


def ai_image_stream(seed: int | None = None) -> Iterator[tuple[Image.Image, str]]:
    """Infinite iterator of AI-generated images."""
    rng = random.Random(seed)
    while True:
        yield get_ai_image(rng)
