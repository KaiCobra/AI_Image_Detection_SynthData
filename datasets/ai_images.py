"""
AI-generated image provider.

Priority order
--------------
1. Load from DATA_DIR/ai/ (user has placed downloaded AI-generated images there).
2. Fall back to procedural generation of images that mimic common visual
   artefacts seen in AI-generated content:
     • Unnaturally smooth gradients / dreamlike colour washes
     • Perfect radial symmetry / mandala-like patterns
     • Over-saturated, hyper-sharp textures (GAN sharpness artefacts)
     • Dreamlike blended landscapes with soft, uniform lighting
     • Synthetic portrait-like ovals with smooth skin-tone gradients

This module is dataset-agnostic: drop JPEG/PNG files from CIFAKE (split=FAKE),
ArtiFact, or GenImage into data/ai/ and the loader picks them up automatically.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from config import AI_DIR, OUTPUT_SIZE


# ── Disk loader ──────────────────────────────────────────────────────────────

def _collect_paths(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in directory.rglob("*") if p.suffix.lower() in exts]


def load_ai_image_from_disk(rng: random.Random) -> Image.Image | None:
    paths = _collect_paths(AI_DIR)
    if not paths:
        return None
    path = rng.choice(paths)
    img = Image.open(path).convert("RGB")
    img = img.resize(OUTPUT_SIZE, Image.LANCZOS)
    return img


# ── Procedural AI-artefact generator ─────────────────────────────────────────

_AI_PALETTES = [
    # (color_a, color_b, color_c)  — dreamlike gradient trios
    ((255, 100, 150), (100, 80, 220), (50, 200, 255)),   # neon dream
    ((255, 180, 50),  (220, 50, 100), (80, 30, 180)),    # sunset surreal
    ((0, 200, 180),   (0, 80, 220),   (180, 0, 255)),    # teal-violet
    ((255, 220, 200), (200, 150, 100),(120, 80, 60)),     # warm portrait
    ((150, 220, 255), (80, 160, 200), (40, 80, 160)),    # overcast digital
    ((200, 255, 200), (80, 200, 120), (20, 120, 80)),    # hyper-nature
]


def _radial_gradient(size: tuple[int, int],
                     center: tuple[float, float],
                     col_a: tuple[int, int, int],
                     col_b: tuple[int, int, int]) -> np.ndarray:
    w, h = size
    cx, cy = center[0] * w, center[1] * h
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    dist /= (dist.max() + 1e-8)
    arr = np.stack([
        col_a[c] * (1 - dist) + col_b[c] * dist for c in range(3)
    ], axis=-1).astype(np.uint8)
    return arr


def _make_smooth_gradient(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Unnaturally smooth colour gradient — hallmark of AI diffusion outputs."""
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


def _make_mandala(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Radially symmetric pattern — AI often produces perfect symmetry."""
    w, h = size
    col_a, col_b, col_c = rng.choice(_AI_PALETTES)
    bg = Image.new("RGB", (w, h), col_b)
    draw = ImageDraw.Draw(bg)
    cx, cy = w // 2, h // 2
    n_rings = rng.randint(4, 10)
    n_arms  = rng.randint(6, 16)

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


def _make_dreamscape(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Blend of soft shapes — mimics AI landscape / abstract art outputs."""
    w, h = size
    col_a, col_b, col_c = rng.choice(_AI_PALETTES)
    img = Image.new("RGB", (w, h), col_a)
    draw = ImageDraw.Draw(img)

    # Sky band
    sky_h = int(h * rng.uniform(0.3, 0.6))
    draw.rectangle([0, 0, w, sky_h], fill=col_b)

    # Oversaturated soft sun/moon disc
    sun_r = rng.randint(30, 80)
    sx = rng.randint(sun_r, w - sun_r)
    sy = rng.randint(sun_r, sky_h)
    draw.ellipse([sx - sun_r, sy - sun_r, sx + sun_r, sy + sun_r], fill=col_c)

    # Silhouette of landscape (perfectly smooth horizon)
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


def _make_portrait_like(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Oval face-like gradient — mimics GAN/diffusion portrait artefacts."""
    w, h = size
    col_a, col_b, _ = rng.choice([
        ((240, 200, 170), (180, 140, 110), (100, 80, 60)),  # skin tones
        ((200, 180, 220), (140, 120, 180), (80, 60, 120)),  # violet fantasy
    ])
    bg_col = (col_b[0] // 2, col_b[1] // 2, col_b[2] // 2)
    img = Image.new("RGB", (w, h), bg_col)
    draw = ImageDraw.Draw(img)

    # Face oval — unnaturally perfect
    face_w = int(w * rng.uniform(0.35, 0.55))
    face_h = int(h * rng.uniform(0.45, 0.65))
    fx = (w - face_w) // 2
    fy = (h - face_h) // 2
    draw.ellipse([fx, fy, fx + face_w, fy + face_h], fill=col_a)

    # Eye-like circles — perfectly symmetric
    eye_y = fy + int(face_h * 0.35)
    eye_r = int(face_w * 0.08)
    left_x  = fx + int(face_w * 0.3)
    right_x = fx + int(face_w * 0.7)
    dark = (col_a[0] // 3, col_a[1] // 3, col_a[2] // 3)
    draw.ellipse([left_x - eye_r, eye_y - eye_r, left_x + eye_r, eye_y + eye_r], fill=dark)
    draw.ellipse([right_x - eye_r, eye_y - eye_r, right_x + eye_r, eye_y + eye_r], fill=dark)

    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2, 5)))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.3))
    return img


_GENERATORS = [
    _make_smooth_gradient,
    _make_mandala,
    _make_dreamscape,
    _make_portrait_like,
]


# ── Public API ────────────────────────────────────────────────────────────────

def get_ai_image(rng: random.Random) -> tuple[Image.Image, str]:
    """
    Return (image, source_tag).
    source_tag is 'disk' if loaded from AI_DIR, otherwise 'synthetic_*'.
    """
    img = load_ai_image_from_disk(rng)
    if img is not None:
        return img, "disk"

    gen = rng.choice(_GENERATORS)
    img = gen(OUTPUT_SIZE, rng)
    return img, f"synthetic_{gen.__name__}"


def ai_image_stream(seed: int | None = None) -> Iterator[tuple[Image.Image, str]]:
    """Infinite iterator of AI-generated images."""
    rng = random.Random(seed)
    while True:
        yield get_ai_image(rng)
