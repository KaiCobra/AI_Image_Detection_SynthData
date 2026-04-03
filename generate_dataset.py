"""
Main pipeline: generate synthetic AI-detection dataset batches.

Usage
-----
# Generate N batches into output/batches/
python generate_dataset.py --batches 100 --output output/batches

# Generate the 5 demo batches (default)
python generate_dataset.py

Each batch folder contains:
  batch_NNNN/
    composite.png   — blended image (AI + real, one large one small)
    mask.png        — binary mask: 255=AI region, 0=real region
    metadata.json   — provenance, alpha, bounding box, etc.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

# Ensure project root is on path when run from a subdirectory
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, DEMO_BATCHES, RANDOM_SEED, DATA_DIR
from datasets.real_images import get_real_image
from datasets.ai_images import get_ai_image
from pipeline.compositor import generate_batch


def run_pipeline(
    n_batches: int,
    output_dir: Path,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> list[Path]:
    """
    Generate `n_batches` composite+mask pairs.

    Returns list of batch directory paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    batch_dirs: list[Path] = []

    for i in range(n_batches):
        batch_name = f"batch_{i:04d}"
        batch_dir  = output_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        # -- Sample one real and one AI image ---------------------------------
        real_img, real_src = get_real_image(rng)
        ai_img,   ai_src   = get_ai_image(rng)

        # -- Composite --------------------------------------------------------
        result = generate_batch(
            real_img=real_img,
            real_source=real_src,
            ai_img=ai_img,
            ai_source=ai_src,
            rng=rng,
            soft_edge=True,
        )

        # -- Save -------------------------------------------------------------
        composite_path = batch_dir / "composite.png"
        mask_path      = batch_dir / "mask.png"
        meta_path      = batch_dir / "metadata.json"

        result.composite.save(composite_path, "PNG")
        result.mask.save(mask_path, "PNG")

        with open(meta_path, "w") as f:
            json.dump(result.metadata, f, indent=2)

        if verbose:
            ai_role = "base (large)" if result.metadata["ai_is_base"] else "patch (small)"
            print(
                f"  [{i+1:>3}/{n_batches}] {batch_name}  "
                f"AI={ai_role}  alpha={result.metadata['alpha']:.2f}  "
                f"patch={result.metadata['patch_size']}  "
                f"real_src={real_src}  ai_src={ai_src}"
            )

        batch_dirs.append(batch_dir)

    return batch_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic AI-detection dataset batches."
    )
    parser.add_argument(
        "--batches", "-n", type=int, default=DEMO_BATCHES,
        help=f"Number of batches to generate (default: {DEMO_BATCHES})"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=OUTPUT_DIR / "demo",
        help="Output directory (default: output/demo)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()

    # Inform user about data directories
    real_count = sum(1 for _ in (DATA_DIR / "real").rglob("*")
                     if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}) \
                 if (DATA_DIR / "real").exists() else 0
    ai_count   = sum(1 for _ in (DATA_DIR / "ai").rglob("*")
                     if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}) \
                 if (DATA_DIR / "ai").exists() else 0

    print("=" * 60)
    print("AI Image Detection — Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  Real images in data/real/ : {real_count}")
    print(f"  AI   images in data/ai/   : {ai_count}")
    if real_count == 0 or ai_count == 0:
        print("  [!] Missing data — using procedural image generation.")
        print("      See config.py for dataset download instructions.")
    print(f"  Batches to generate       : {args.batches}")
    print(f"  Output directory          : {args.output}")
    print(f"  Random seed               : {args.seed}")
    print("-" * 60)

    dirs = run_pipeline(
        n_batches=args.batches,
        output_dir=args.output,
        seed=args.seed,
        verbose=True,
    )

    print("-" * 60)
    print(f"Done. {len(dirs)} batch(es) saved to: {args.output}")
    print("\nEach batch contains:")
    print("  composite.png  — blended image")
    print("  mask.png       — binary mask (255=AI region, 0=real region)")
    print("  metadata.json  — provenance and blend parameters")


if __name__ == "__main__":
    main()
