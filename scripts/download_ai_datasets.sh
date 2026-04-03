#!/usr/bin/env bash
# ============================================================
# Download AI-GENERATED image datasets into data/ai/
# Run from the project root directory.
#
# Default downloads (~3 GB, no registration needed):
#   Synthbuster     — 2024, Zenodo, ~9K 圖, 720P–4K
#                     涵蓋 DALL-E 2/3, Firefly, MJ v5, SD XL 等 9 種生成器
#   MS COCOAI       — 2025, HuggingFace, ~48K AI 圖
#                     涵蓋 SD 3, SDXL, DALL-E 3, Midjourney v6（最新）
#   DiffusionDB 10K — CC BY 4.0, HuggingFace, 10K 圖, 512×512
# ============================================================
set -euo pipefail

AI_DIR="data/ai"
mkdir -p "$AI_DIR"

echo "=== Downloading AI-Generated Image Datasets ==="

# ── 1. Synthbuster (Zenodo, 2024, no registration) ───────────────────────────
# ~9 000 images at 720P–4K, handpicked for photorealism.
# Generators: DALL-E 2, DALL-E 3, Adobe Firefly, Midjourney v5,
#             SD 1.3, SD 2, SD XL, Glide
# Published in: IEEE Open Journal of Signal Processing 2024
echo "[1/3] Synthbuster — ~9K images, 720P-4K (Zenodo, no login)..."
mkdir -p "$AI_DIR/synthbuster"
if [ ! -f "$AI_DIR/synthbuster/.done" ]; then
    wget -q --show-progress \
        "https://zenodo.org/records/10066460/files/synthbuster.zip" \
        -O /tmp/synthbuster.zip
    unzip -q /tmp/synthbuster.zip -d "$AI_DIR/synthbuster/"
    rm /tmp/synthbuster.zip
    touch "$AI_DIR/synthbuster/.done"
    echo "    Synthbuster done."
else
    echo "    Synthbuster already downloaded."
fi

# ── 2. MS COCOAI / Defactify 2025 (HuggingFace, no login) ──────────────────
# 96K images: 50% real COCO, 50% AI-generated.
# We download only the AI split (~48K images).
# Generators: Stable Diffusion 3, SDXL, SD 2.1, DALL-E 3, Midjourney v6
echo "[2/3] MS COCOAI (Defactify 2025) — ~48K AI images (HuggingFace, no login)..."
python3 - <<'PYEOF'
import sys
from pathlib import Path

out_dir = Path("data/ai/ms_cocoai")
done    = out_dir / ".done"
out_dir.mkdir(parents=True, exist_ok=True)

if done.exists():
    print("    MS COCOAI already downloaded.")
    sys.exit(0)

try:
    from datasets import load_dataset

    print("    Loading Rajarshi-Roy-research/Defactify_Image_Dataset ...")
    ds = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        split="train",
        streaming=True,
    )

    saved = 0
    for i, sample in enumerate(ds):
        # Keep only AI-generated images (label != "real")
        label = str(sample.get("label", "")).lower()
        if label == "real" or label == "0":
            continue
        img = sample["image"].convert("RGB")
        img.save(out_dir / f"cocoai_{saved:05d}.png")
        saved += 1
        if saved % 5000 == 0:
            print(f"    Saved {saved} MS COCOAI images...")
        if saved >= 48000:
            break

    done.touch()
    print(f"    MS COCOAI done: {saved} images in {out_dir}")

except Exception as e:
    print(f"    [ERROR] {e}")
    print("    Dataset URL: https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset")
PYEOF

# ── 3. DiffusionDB 10K (HuggingFace, CC BY 4.0, no login) ───────────────────
# 10 000 Stable Diffusion images at native 512×512 PNG.
echo "[3/3] DiffusionDB — 10K images @ 512×512 (HuggingFace)..."
python3 - <<'PYEOF'
import sys
from pathlib import Path

out_dir = Path("data/ai/diffusiondb_10k")
done    = out_dir / ".done"
out_dir.mkdir(parents=True, exist_ok=True)

if done.exists():
    print("    DiffusionDB already downloaded.")
    sys.exit(0)

try:
    from datasets import load_dataset

    print("    Loading poloclub/diffusiondb large_random_10k ...")
    ds = load_dataset("poloclub/diffusiondb", "large_random_10k", split="train")

    saved = 0
    for i, sample in enumerate(ds):
        sample["image"].convert("RGB").save(out_dir / f"diffdb_{i:05d}.png")
        saved += 1
        if saved % 2000 == 0:
            print(f"    Saved {saved}/10000 DiffusionDB images...")

    done.touch()
    print(f"    DiffusionDB done: {saved} images in {out_dir}")

except Exception as e:
    print(f"    [ERROR] {e}")
    print("    Subsets: large_random_1k / large_random_5k / large_random_10k")
PYEOF

echo ""
echo "=== AI dataset download complete ==="
echo "Images are in: $AI_DIR"
echo ""
echo "Optional larger datasets:"
echo "  ArtiFact  (2.5M, Apache 2.0, ~120 GB, 25 generators):"
echo "    kaggle datasets download -d awsaf49/artifact-dataset"
echo ""
echo "  GenImage  (2.68M, CC BY-NC-SA 4.0, ~500 GB, 8 generators):"
echo "    git clone https://github.com/GenImage-Dataset/GenImage"
echo "    python download_genimage.py --destination ./data/ai/genimage"
echo ""
echo "Counts:"
find "$AI_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l
