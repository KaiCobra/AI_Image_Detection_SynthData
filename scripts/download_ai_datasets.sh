#!/usr/bin/env bash
# ============================================================
# Download AI-GENERATED image datasets into data/ai/
# Run from the project root directory.
# ============================================================
set -euo pipefail

AI_DIR="data/ai"
mkdir -p "$AI_DIR"

echo "=== Downloading AI-Generated Image Datasets ==="

# ── 1. CIFAKE (HuggingFace) ──────────────────────────────────────────────────
# 60 000 AI-generated images (FAKE split), 32×32 upscalable to any size.
# License: Research/educational use  |  https://huggingface.co/datasets/jlbaker61/CIFAKE
echo "[1/3] CIFAKE — 60 K AI-generated images (HuggingFace)..."
python3 - <<'PYEOF'
import os, sys
from pathlib import Path

out_dir = Path("data/ai/cifake")
out_dir.mkdir(parents=True, exist_ok=True)
done_flag = out_dir / ".done"
if done_flag.exists():
    print("    CIFAKE already downloaded.")
    sys.exit(0)

try:
    from datasets import load_dataset
    print("    Loading CIFAKE FAKE split (streaming)...")
    ds = load_dataset("jlbaker61/CIFAKE", split="train", streaming=False)
    # Keep only FAKE images (label=0 in this dataset means FAKE)
    fake_ds = ds.filter(lambda x: x["label"] == 0)
    saved = 0
    for i, sample in enumerate(fake_ds):
        img = sample["image"].resize((256, 256))  # upscale from 32x32
        img.save(out_dir / f"cifake_{i:05d}.png")
        saved += 1
        if i % 5000 == 0:
            print(f"    Saved {saved} images...")
    done_flag.touch()
    print(f"    CIFAKE done: {saved} images saved to {out_dir}")
except Exception as e:
    print(f"    [ERROR] {e}")
    print("    Make sure 'datasets' is installed: pip install datasets")
PYEOF

# ── 2. ArtiFact (HuggingFace, streaming sample) ──────────────────────────────
# 2.5 M images from 27 AI generators. We take a 5 000-image streaming sample.
# License: CC BY 4.0  |  https://huggingface.co/datasets/awsaf49/artifact
echo "[2/3] ArtiFact — 5 000 sample images (HuggingFace streaming)..."
python3 - <<'PYEOF'
import sys
from pathlib import Path

out_dir = Path("data/ai/artifact")
out_dir.mkdir(parents=True, exist_ok=True)
done_flag = out_dir / ".done"
if done_flag.exists():
    print("    ArtiFact sample already downloaded.")
    sys.exit(0)

MAX_SAMPLES = 5000
try:
    from datasets import load_dataset
    ds = load_dataset("awsaf49/artifact", split="train", streaming=True)
    saved = 0
    for sample in ds:
        if sample.get("label", 1) != 1:  # 1 = AI-generated in this dataset
            continue
        img = sample["image"].convert("RGB").resize((512, 512))
        img.save(out_dir / f"artifact_{saved:05d}.png")
        saved += 1
        if saved >= MAX_SAMPLES:
            break
        if saved % 1000 == 0:
            print(f"    Saved {saved}/{MAX_SAMPLES}...")
    done_flag.touch()
    print(f"    ArtiFact done: {saved} images saved to {out_dir}")
except Exception as e:
    print(f"    [ERROR] {e}")
PYEOF

# ── 3. GenImage (manual download instructions) ───────────────────────────────
# 1.3 M images from 8 generators (Midjourney, SD 1.4/1.5, DALL-E 2, etc.)
# License: Research use  |  https://github.com/GenImage-Dataset/GenImage
echo "[3/3] GenImage — manual download required."
echo "    Visit: https://github.com/GenImage-Dataset/GenImage"
echo "    Follow the Google Drive links in the repository README."
echo "    Place extracted images into: data/ai/genimage/"
echo ""

echo "=== AI dataset download complete (or instructions shown) ==="
echo "Images are in: $AI_DIR"
