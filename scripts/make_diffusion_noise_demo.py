"""
Create a simple forward-diffusion noising demo from a single image.

Usage:
  python scripts/make_diffusion_noise_demo.py path/to/image.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _annotate(image: Image.Image, text: str) -> Image.Image:
    label_h = 34
    canvas = Image.new("RGB", (image.width, image.height + label_h), (18, 18, 18))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, image.height + 9), text, fill=(235, 235, 235))
    return canvas


def _to_model_space(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return arr * 2.0 - 1.0


def _from_model_space(arr: np.ndarray) -> Image.Image:
    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _coarse_noise(shape: tuple[int, int, int], rng: np.random.Generator, grain_size: int) -> np.ndarray:
    height, width, channels = shape
    grain_size = max(1, grain_size)
    small_h = max(1, (height + grain_size - 1) // grain_size)
    small_w = max(1, (width + grain_size - 1) // grain_size)
    small_noise = rng.standard_normal((small_h, small_w, channels), dtype=np.float32)
    noise = np.repeat(np.repeat(small_noise, grain_size, axis=0), grain_size, axis=1)
    return noise[:height, :width, :]


def generate_noisy_images(
    image_path: Path,
    output_dir: Path,
    steps: int = 5,
    signal_start: float = 1.0,
    signal_end: float = 0.0,
    grain_size: int = 14,
    seed: int = 42,
) -> list[Path]:
    image = Image.open(image_path).convert("RGB")
    x0 = _to_model_space(image)
    rng = np.random.default_rng(seed)

    alpha_bars = np.linspace(signal_start, signal_end, steps, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for index, alpha_bar in enumerate(alpha_bars, start=1):
        noise = _coarse_noise(x0.shape, rng, grain_size)
        xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * noise
        noisy = _from_model_space(xt)
        labeled = _annotate(
            noisy,
            f"step {index}/{steps}  signal={alpha_bar:.3f}  noise={1.0 - alpha_bar:.3f}  grain={grain_size}",
        )
        out_path = output_dir / f"{image_path.stem}_noise_{index:02d}.png"
        labeled.save(out_path)
        saved_paths.append(out_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate five forward-diffusion style noisy images from one input image."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save outputs. Default: output/diffusion_demo/<image_stem>",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--signal-start", type=float, default=1.0)
    parser.add_argument("--signal-end", type=float, default=0.0)
    parser.add_argument("--grain-size", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("output/diffusion_demo") / args.image_path.stem
    saved_paths = generate_noisy_images(
        image_path=args.image_path,
        output_dir=output_dir,
        steps=args.steps,
        signal_start=args.signal_start,
        signal_end=args.signal_end,
        grain_size=args.grain_size,
        seed=args.seed,
    )
    print(f"Saved {len(saved_paths)} images to: {output_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()