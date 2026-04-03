"""
Composite image + binary mask generator.

Logic
-----
Each batch consists of:
  • composite.png  — one large base image with a smaller patch overlaid.
                     Either the base or the patch is the AI image (random).
  • mask.png       — binary mask where WHITE = AI-involved region,
                     BLACK = real/human region.

Placement rules
---------------
- One image is designated "large" (fills the canvas), the other "small"
  (resized to SMALL_RATIO_MIN–SMALL_RATIO_MAX of canvas, placed at a random
  position that keeps it fully within bounds).
- Which one is AI vs real is randomised each batch.
- The small image is blended onto the large with alpha ∈ [ALPHA_MIN, ALPHA_MAX]
  so the overlay is always clearly visible.
- The mask is a binary PNG (0 = real, 255 = AI) at the same size as the composite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import (
    OUTPUT_SIZE,
    ALPHA_MIN,
    ALPHA_MAX,
    SMALL_RATIO_MIN,
    SMALL_RATIO_MAX,
)


@dataclass
class BatchResult:
    """Result of a single composite generation."""
    composite: Image.Image        # RGB composite image
    mask: Image.Image             # L-mode mask (0=real, 255=AI)
    metadata: dict = field(default_factory=dict)
    """
    metadata keys:
      ai_is_base   : bool  — True if AI image is the large base
      ai_bbox      : (x0, y0, x1, y1)  — bounding box of AI region on canvas
      alpha        : float — blend alpha used for the small overlay
      real_source  : str   — 'disk' or 'synthetic_*'
      ai_source    : str   — 'disk' or 'synthetic_*'
    """


def _random_placement(canvas_w: int, canvas_h: int,
                      patch_w: int, patch_h: int,
                      rng: random.Random) -> tuple[int, int]:
    """Return a random (x, y) top-left such that the patch stays fully inside."""
    max_x = max(0, canvas_w - patch_w)
    max_y = max(0, canvas_h - patch_h)
    return rng.randint(0, max_x), rng.randint(0, max_y)


def _random_patch_size(canvas_w: int, canvas_h: int,
                       rng: random.Random) -> tuple[int, int]:
    ratio = rng.uniform(SMALL_RATIO_MIN, SMALL_RATIO_MAX)
    pw = max(32, int(canvas_w * ratio))
    ph = max(32, int(canvas_h * ratio))
    return pw, ph


def _soft_edge_mask(patch_w: int, patch_h: int, blur_r: int) -> Image.Image:
    """
    Create a soft-edged rectangular alpha mask for the patch.
    The interior is fully opaque; edges fade to transparent with a Gaussian blur.
    This makes the blend more natural.
    """
    m = Image.new("L", (patch_w, patch_h), 255)
    m = m.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return m


def generate_batch(
    real_img: Image.Image,
    real_source: str,
    ai_img: Image.Image,
    ai_source: str,
    rng: random.Random,
    soft_edge: bool = True,
) -> BatchResult:
    """
    Produce one (composite, mask) pair.

    Parameters
    ----------
    real_img, ai_img   : PIL RGB images at OUTPUT_SIZE
    real_source, ai_source : provenance tags (for metadata)
    rng                : seeded Random instance
    soft_edge          : blend patch edges softly (more realistic)
    """
    W, H = OUTPUT_SIZE

    # Decide which is the large base image
    ai_is_base = rng.random() < 0.5

    if ai_is_base:
        base_img  = ai_img.copy().resize((W, H), Image.LANCZOS)
        patch_src = real_img
    else:
        base_img  = real_img.copy().resize((W, H), Image.LANCZOS)
        patch_src = ai_img

    # Size and place the patch
    pw, ph = _random_patch_size(W, H, rng)
    patch = patch_src.resize((pw, ph), Image.LANCZOS).convert("RGBA")

    px, py = _random_placement(W, H, pw, ph, rng)
    alpha  = rng.uniform(ALPHA_MIN, ALPHA_MAX)

    # Build alpha channel for patch
    if soft_edge:
        blur_r = max(4, min(pw, ph) // 10)
        edge_mask = _soft_edge_mask(pw, ph, blur_r)
        a_arr = np.array(edge_mask, dtype=np.float32) / 255.0 * alpha * 255
        a_arr = np.clip(a_arr, 0, 255).astype(np.uint8)
        patch_a = Image.fromarray(a_arr, "L")
    else:
        flat_a = int(alpha * 255)
        patch_a = Image.new("L", (pw, ph), flat_a)

    r, g, b, _ = patch.split()
    patch = Image.merge("RGBA", (r, g, b, patch_a))

    # Composite
    canvas = base_img.convert("RGBA")
    canvas.paste(patch, (px, py), mask=patch_a)
    composite_rgb = canvas.convert("RGB")

    # ── Mask construction ─────────────────────────────────────────────────────
    # WHITE (255) = AI-involved pixel, BLACK (0) = real/human pixel.
    mask_arr = np.zeros((H, W), dtype=np.uint8)

    if ai_is_base:
        # The entire canvas is AI base; the patch (real) overlays it.
        # The patch region becomes real — still AI underneath, but the
        # visible top layer there is real. We mark the blend region based
        # on the actual alpha: pixels with high alpha → mostly real, low → mostly AI.
        mask_arr[:, :] = 255  # start: all AI
        a_full = np.zeros((H, W), dtype=np.float32)
        patch_alpha_arr = np.array(patch_a, dtype=np.float32) / 255.0
        a_full[py:py + ph, px:px + pw] = patch_alpha_arr
        # Where real patch alpha > 0.5, mark as real (0)
        mask_arr[a_full > 0.5] = 0
    else:
        # The base is real; the patch (AI) overlays it.
        # Where the AI patch alpha > 0 → mark as AI (255).
        patch_alpha_arr = np.array(patch_a, dtype=np.float32) / 255.0
        region_mask = np.zeros((H, W), dtype=np.float32)
        region_mask[py:py + ph, px:px + pw] = patch_alpha_arr
        mask_arr[region_mask > 0.3] = 255  # threshold: ai region

    mask_img = Image.fromarray(mask_arr, "L")

    # Compute bounding box of AI region in mask
    ai_ys, ai_xs = np.where(mask_arr == 255)
    if len(ai_xs) > 0:
        ai_bbox = (int(ai_xs.min()), int(ai_ys.min()),
                   int(ai_xs.max()), int(ai_ys.max()))
    else:
        ai_bbox = (0, 0, 0, 0)

    metadata = {
        "ai_is_base": ai_is_base,
        "ai_bbox":    ai_bbox,
        "alpha":      round(alpha, 4),
        "patch_xy":   (px, py),
        "patch_size": (pw, ph),
        "real_source": real_source,
        "ai_source":   ai_source,
    }

    return BatchResult(composite=composite_rgb, mask=mask_img, metadata=metadata)
