"""
Composite image + binary mask generator.

Two modes
---------
1. **SAM3 cutout overlay** (new default):
   AI object cutout (RGBA from SAM3) is pasted onto a real base image.
   The mask is derived from the cutout's alpha channel.

2. **Full-image overlay** (fallback when Qwen3 returns </nothing> or SAM3 fails):
   The entire AI image is resized as a patch and overlaid onto the real base.

In both modes the real image is always the base and AI is always the overlay.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from config import (
    OUTPUT_SIZE,
    SMALL_RATIO_MIN,
    SMALL_RATIO_MAX,
)


@dataclass
class BatchResult:
    """Result of a single composite generation."""
    composite: Image.Image
    mask: Image.Image
    metadata: dict = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resize_to_canvas(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale-to-fill then center-crop to `size`. No padding."""
    from PIL import ImageOps
    if img.size[0] == 0 or img.size[1] == 0:
        return Image.new("RGB", size, (0, 0, 0))
    return ImageOps.fit(img, size, Image.LANCZOS)


def _random_placement(canvas_w: int, canvas_h: int,
                      patch_w: int, patch_h: int,
                      rng: random.Random) -> tuple[int, int]:
    max_x = max(0, canvas_w - patch_w)
    max_y = max(0, canvas_h - patch_h)
    return rng.randint(0, max_x), rng.randint(0, max_y)


def _random_patch_size(canvas_w: int, canvas_h: int,
                       rng: random.Random) -> tuple[int, int]:
    ratio = rng.uniform(SMALL_RATIO_MIN, SMALL_RATIO_MAX)
    pw = max(32, int(canvas_w * ratio))
    ph = max(32, int(canvas_h * ratio))
    return pw, ph


# ── SAM3 cutout overlay ─────────────────────────────────────────────────────

def generate_batch_with_cutout(
    real_img: Image.Image,
    real_source: str,
    cutout: Image.Image,
    ai_source: str,
    ai_filename: str,
    ai_caption: str,
    ai_object: str,
    sam3_score: float,
    sam3_bbox: list,
    rng: random.Random,
) -> BatchResult:
    """
    Paste an RGBA cutout (from SAM3) onto a real base image.

    The cutout is resized to fit the canvas while preserving aspect ratio,
    then placed at a random position.
    """
    W, H = OUTPUT_SIZE

    # Prepare base
    base = _resize_to_canvas(real_img, (W, H))

    # Resize cutout to fit within canvas (preserve aspect ratio)
    cw, ch = cutout.size
    scale = min(W / cw, H / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    cutout_resized = cutout.resize((new_w, new_h), Image.LANCZOS)

    # Random placement
    px, py = _random_placement(W, H, new_w, new_h, rng)

    # Composite
    canvas = base.convert("RGBA")
    canvas.paste(cutout_resized, (px, py), mask=cutout_resized)
    composite_rgb = canvas.convert("RGB")

    # Build mask: AI pixels where cutout alpha > 0
    mask_arr = np.zeros((H, W), dtype=np.uint8)
    cutout_alpha = np.array(cutout_resized.split()[-1])  # alpha channel
    mask_arr[py:py + new_h, px:px + new_w] = (cutout_alpha > 0).astype(np.uint8) * 255

    mask_img = Image.fromarray(mask_arr, "L")

    # Bounding box of AI region
    ai_ys, ai_xs = np.where(mask_arr == 255)
    if len(ai_xs) > 0:
        ai_bbox = (int(ai_xs.min()), int(ai_ys.min()),
                   int(ai_xs.max()), int(ai_ys.max()))
    else:
        ai_bbox = (0, 0, 0, 0)

    metadata = {
        "mode": "sam3_cutout",
        "ai_source": ai_source,
        "ai_filename": ai_filename,
        "ai_caption": ai_caption,
        "ai_object": ai_object,
        "sam3_score": round(sam3_score, 4),
        "sam3_bbox": sam3_bbox,
        "real_source": real_source,
        "overlay_position": (px, py),
        "overlay_size": (new_w, new_h),
        "ai_bbox": ai_bbox,
        "canvas_size": list(OUTPUT_SIZE),
    }

    return BatchResult(composite=composite_rgb, mask=mask_img, metadata=metadata)


# ── Full-image fallback overlay ──────────────────────────────────────────────

def generate_batch_fullimage(
    real_img: Image.Image,
    real_source: str,
    ai_img: Image.Image,
    ai_source: str,
    ai_filename: str,
    ai_caption: str,
    ai_object: str,
    rng: random.Random,
) -> BatchResult:
    """
    Overlay the entire AI image as a patch onto a real base image.
    Used when Qwen3 returns </nothing> or SAM3 fails to segment.
    """
    W, H = OUTPUT_SIZE

    base = _resize_to_canvas(real_img, (W, H))

    # Resize AI image as a random-sized patch
    pw, ph = _random_patch_size(W, H, rng)
    patch = _resize_to_canvas(ai_img, (pw, ph)).convert("RGBA")
    px, py = _random_placement(W, H, pw, ph, rng)

    # Composite (full opacity)
    canvas = base.convert("RGBA")
    canvas.paste(patch, (px, py), mask=patch)
    composite_rgb = canvas.convert("RGB")

    # Mask: entire patch region is AI
    mask_arr = np.zeros((H, W), dtype=np.uint8)
    mask_arr[py:py + ph, px:px + pw] = 255
    mask_img = Image.fromarray(mask_arr, "L")

    ai_bbox = (px, py, px + pw, py + ph)

    metadata = {
        "mode": "fullimage_overlay",
        "ai_source": ai_source,
        "ai_filename": ai_filename,
        "ai_caption": ai_caption,
        "ai_object": ai_object,
        "sam3_score": None,
        "sam3_bbox": None,
        "real_source": real_source,
        "overlay_position": (px, py),
        "overlay_size": (pw, ph),
        "ai_bbox": ai_bbox,
        "canvas_size": list(OUTPUT_SIZE),
    }

    return BatchResult(composite=composite_rgb, mask=mask_img, metadata=metadata)


# ── Legacy API (kept for backward compat) ────────────────────────────────────

def generate_batch(
    real_img: Image.Image,
    real_source: str,
    ai_img: Image.Image,
    ai_source: str,
    rng: random.Random,
    soft_edge: bool = True,
) -> BatchResult:
    """Legacy compositor — random base/patch roles with alpha blending."""
    from config import ALPHA_MIN, ALPHA_MAX

    W, H = OUTPUT_SIZE
    ai_is_base = rng.random() < 0.5

    if ai_is_base:
        base_img = ai_img.copy().resize((W, H), Image.LANCZOS)
        patch_src = real_img
    else:
        base_img = real_img.copy().resize((W, H), Image.LANCZOS)
        patch_src = ai_img

    pw, ph = _random_patch_size(W, H, rng)
    patch = patch_src.resize((pw, ph), Image.LANCZOS).convert("RGBA")
    px, py = _random_placement(W, H, pw, ph, rng)
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX)

    if soft_edge:
        blur_r = max(4, min(pw, ph) // 10)
        m = Image.new("L", (pw, ph), 255)
        m = m.filter(ImageFilter.GaussianBlur(radius=blur_r))
        a_arr = np.array(m, dtype=np.float32) / 255.0 * alpha * 255
        a_arr = np.clip(a_arr, 0, 255).astype(np.uint8)
        patch_a = Image.fromarray(a_arr, "L")
    else:
        patch_a = Image.new("L", (pw, ph), int(alpha * 255))

    r, g, b, _ = patch.split()
    patch = Image.merge("RGBA", (r, g, b, patch_a))

    canvas = base_img.convert("RGBA")
    canvas.paste(patch, (px, py), mask=patch_a)
    composite_rgb = canvas.convert("RGB")

    mask_arr = np.zeros((H, W), dtype=np.uint8)
    if ai_is_base:
        mask_arr[:, :] = 255
        a_full = np.zeros((H, W), dtype=np.float32)
        a_full[py:py + ph, px:px + pw] = np.array(patch_a, dtype=np.float32) / 255.0
        mask_arr[a_full > 0.5] = 0
    else:
        region_mask = np.zeros((H, W), dtype=np.float32)
        region_mask[py:py + ph, px:px + pw] = np.array(patch_a, dtype=np.float32) / 255.0
        mask_arr[region_mask > 0.3] = 255

    mask_img = Image.fromarray(mask_arr, "L")
    ai_ys, ai_xs = np.where(mask_arr == 255)
    if len(ai_xs) > 0:
        ai_bbox = (int(ai_xs.min()), int(ai_ys.min()),
                   int(ai_xs.max()), int(ai_ys.max()))
    else:
        ai_bbox = (0, 0, 0, 0)

    metadata = {
        "mode": "legacy",
        "ai_is_base": ai_is_base,
        "ai_bbox": ai_bbox,
        "alpha": round(alpha, 4),
        "patch_xy": (px, py),
        "patch_size": (pw, ph),
        "real_source": real_source,
        "ai_source": ai_source,
    }

    return BatchResult(composite=composite_rgb, mask=mask_img, metadata=metadata)
