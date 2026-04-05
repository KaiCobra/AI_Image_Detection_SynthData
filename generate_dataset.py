"""
Main pipeline: generate synthetic AI-detection dataset batches.

Flow
----
1. Read (filename, caption) from Synthbuster prompts.csv
2. Send AI image to SAM3 /auto_segment → get RGBA cutout of the largest object
   (fallback to full-image overlay if SAM3 finds nothing)
3. Overlay cutout onto a random real image
4. Save to KAI_NNN_[real_dataset]_[ai_dataset]/ folder

Prerequisites
-------------
- SAM3 server running on port 5051:   bash sam3_server/start_server.sh

Usage
-----
python generate_dataset.py --batches 20 --output output/KAI --start-id 1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, DEMO_BATCHES, RANDOM_SEED, OUTPUT_SIZE
from datasets.real_images import load_real_image_with_meta
from datasets.ai_images import (
    get_synthbuster_image_path_by_index,
    synthbuster_count,
)
from pipeline.compositor import (
    generate_batch_with_cutout,
    generate_batch_fullimage,
)
from sam3_server.client import auto_segment_object, is_server_running as sam3_running

EDITOR = "Kai"
TOOL_NAME = "AI_detection_synthData"
EDIT_FAMILY = "AI_Real_synth"


def _sample_id(n: int) -> str:
    return f"KAI_{n:03d}"


def _folder_name(sample_id: str, real_dataset: str, ai_dataset: str) -> str:
    return f"{sample_id}_[{real_dataset}]_[{ai_dataset}]"


def _resolution_str() -> str:
    w, h = OUTPUT_SIZE
    return f"{w}x{h}"


def run_pipeline(
    n_batches: int,
    output_dir: Path,
    start_id: int = 1,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> list[Path]:
    """Generate `n_batches` samples using SAM3 auto-segmentation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if not sam3_running():
        print("[ERROR] SAM3 server not running on port 5051.")
        print("        Start it:  bash sam3_server/start_server.sh")
        sys.exit(1)

    total_prompts = synthbuster_count()
    if total_prompts == 0:
        print("[ERROR] No prompts found. Ensure data/ai/synthbuster/synthbuster/prompts.csv exists.")
        sys.exit(1)

    batch_dirs: list[Path] = []
    batch_count = 0   # number of successfully saved samples
    prompt_idx = 0    # advances on every attempt

    while batch_count < n_batches:
        sample_id = _sample_id(start_id + batch_count)

        # -- Get AI image + caption from Synthbuster --------------------------
        entry = get_synthbuster_image_path_by_index(prompt_idx, rng)
        prompt_idx += 1
        if entry is None:
            continue

        ai_path, ai_source, ai_filename, ai_caption = entry
        ai_dataset = ai_source.split("/")[-1]  # e.g. "dalle3"

        # -- Get real image + dataset info ------------------------------------
        real_entry = load_real_image_with_meta(rng)
        if real_entry is None:
            print("[ERROR] No real images found in data/real/")
            sys.exit(1)
        real_img, real_dataset, real_path = real_entry

        # -- SAM3 auto-segmentation -------------------------------------------
        sam3_result = auto_segment_object(image_path=str(ai_path))

        if sam3_result is not None:
            result = generate_batch_with_cutout(
                real_img=real_img,
                real_source=real_dataset,
                cutout=sam3_result["cutout"],
                ai_source=ai_source,
                ai_filename=ai_filename,
                ai_caption=ai_caption,
                ai_object="auto",
                sam3_score=sam3_result["score"],
                sam3_bbox=sam3_result["bbox"],
                rng=rng,
            )
            mode = "cutout"
        else:
            ai_img = Image.open(ai_path).convert("RGB")
            result = generate_batch_fullimage(
                real_img=real_img,
                real_source=real_dataset,
                ai_img=ai_img,
                ai_source=ai_source,
                ai_filename=ai_filename,
                ai_caption=ai_caption,
                ai_object="auto",
                rng=rng,
            )
            mode = "fullimg (sam3 fail)"

        # -- Build folder and save files --------------------------------------
        folder_name = _folder_name(sample_id, real_dataset, ai_dataset)
        batch_dir = output_dir / folder_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        # source.png — original real image (pre-resize copy)
        if real_path is not None:
            shutil.copy2(real_path, batch_dir / "source.png")
        else:
            real_img.save(batch_dir / "source.png", "PNG")

        # source_ai.png — original AI image
        shutil.copy2(ai_path, batch_dir / "source_ai.png")

        # edited.png — composite
        result.composite.save(batch_dir / "edited.png", "PNG")

        # mask_core.png — binary mask
        result.mask.save(batch_dir / "mask_core.png", "PNG")

        # meta.json
        meta = {
            "sample_id": sample_id,
            "editor": EDITOR,
            "edit_family": EDIT_FAMILY,
            "pie_type": None,
            "ai_image_description": ai_caption,
            "real_image_dataset": real_dataset,
            "ai_image_dataset": ai_dataset,
            "blended_words": [],
            "tool_name": TOOL_NAME,
            "resolution": _resolution_str(),
        }
        with open(batch_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if verbose:
            score_str = f"{result.metadata['sam3_score']:.2f}" if result.metadata.get("sam3_score") else "-"
            print(
                f"  [{batch_count+1:>3}/{n_batches}] {sample_id}  "
                f"mode={mode:<20s}  "
                f"score={score_str}  "
                f"real={real_dataset}  ai={ai_dataset}"
            )

        batch_dirs.append(batch_dir)
        batch_count += 1

    return batch_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic AI-detection dataset (KAI format)."
    )
    parser.add_argument(
        "--batches", "-n", type=int, default=DEMO_BATCHES,
        help=f"Number of samples to generate (default: {DEMO_BATCHES})"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=OUTPUT_DIR / "KAI",
        help="Output directory (default: output/KAI)"
    )
    parser.add_argument(
        "--start-id", type=int, default=1,
        help="Starting sample number for KAI_NNN (default: 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()

    total_prompts = synthbuster_count()

    print("=" * 70)
    print("AI Image Detection — Synthetic Dataset Generator (KAI format)")
    print("=" * 70)
    print(f"  Synthbuster prompts  : {total_prompts}")
    print(f"  SAM3  server (5051)  : {'OK' if sam3_running() else 'NOT RUNNING'}")
    print(f"  Samples to generate  : {args.batches}")
    print(f"  Sample ID range      : {_sample_id(args.start_id)} ~ {_sample_id(args.start_id + args.batches - 1)}")
    print(f"  Output directory     : {args.output}")
    print(f"  Random seed          : {args.seed}")
    print("-" * 70)

    dirs = run_pipeline(
        n_batches=args.batches,
        output_dir=args.output,
        start_id=args.start_id,
        seed=args.seed,
        verbose=True,
    )

    print("-" * 70)
    print(f"Done. {len(dirs)} sample(s) saved to: {args.output}")
    print("\nEach sample folder contains:")
    print("  source.png     — original real image")
    print("  source_ai.png  — original AI image")
    print("  edited.png     — composite result")
    print("  mask_core.png  — binary mask (255=AI, 0=real)")
    print("  meta.json      — sample metadata")


if __name__ == "__main__":
    main()
