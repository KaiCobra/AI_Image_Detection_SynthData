#!/usr/bin/env bash
# ============================================================
# Download REAL image datasets into data/real/
# Run from the project root directory.
#
# Default downloads (~2 GB, no registration needed):
#   DIV2K        — 1 000 高畫質真實照片 (≥ 2K 解析度)
#   COCO 2017 val — 5 000 自然場景照片 (~1 GB)
#   Quick Draw!  — 10 類別 × 2 000 真人手繪素描
# ============================================================
set -euo pipefail

REAL_DIR="data/real"
mkdir -p "$REAL_DIR"

echo "=== Downloading Real Image Datasets ==="

# ── 1. DIV2K (HuggingFace, open access, no login) ────────────────────────────
# 1 000 high-resolution real photos, at least one dimension ≥ 2K.
# Diverse subjects: people, objects, nature, architecture, macro.
# No AI involvement — original photographs.
echo "[1/3] DIV2K — 1 000 high-resolution real photos (HuggingFace)..."
python3 - <<'PYEOF'
import sys
from pathlib import Path

out_dir = Path("data/real/div2k")
done    = out_dir / ".done"
out_dir.mkdir(parents=True, exist_ok=True)

if done.exists():
    print("    DIV2K already downloaded.")
    sys.exit(0)

try:
    from datasets import load_dataset

    print("    Loading eugenesiow/Div2k (bicubic_x2 config, train split) ...")
    ds = load_dataset("eugenesiow/Div2k", "bicubic_x2", split="train")

    saved = 0
    for i, sample in enumerate(ds):
        # Use the high-resolution version (hr key)
        img = sample.get("hr") or sample.get("image")
        if img is None:
            continue
        img.convert("RGB").save(out_dir / f"div2k_{i:04d}.png")
        saved += 1

    done.touch()
    print(f"    DIV2K done: {saved} images in {out_dir}")

except Exception as e:
    print(f"    [ERROR] {e}")
    print("    Alternative: download directly from https://data.vision.ee.ethz.ch/cvl/DIV2K/")
PYEOF

# ── 2. COCO 2017 Validation (~1 GB, CC BY 4.0) ──────────────────────────────
echo "[2/3] COCO 2017 val (5 000 natural scene images)..."
mkdir -p "$REAL_DIR/coco"
if [ ! -f "$REAL_DIR/coco/.done" ]; then
    wget -q --show-progress \
        http://images.cocodataset.org/zips/val2017.zip \
        -O /tmp/coco_val2017.zip
    unzip -q /tmp/coco_val2017.zip -d "$REAL_DIR/coco/"
    mv "$REAL_DIR/coco/val2017/"* "$REAL_DIR/coco/"
    rmdir "$REAL_DIR/coco/val2017"
    rm /tmp/coco_val2017.zip
    touch "$REAL_DIR/coco/.done"
    echo "    COCO 2017 val done."
else
    echo "    COCO already downloaded."
fi

# ── 3. Quick, Draw! bitmap sketches (CC BY 4.0) ──────────────────────────────
# 10 categories × up to 2 000 sketches each.
# Requires: pip install quickdraw
echo "[3/3] Quick Draw! human sketches (10 categories)..."
python3 - <<'PYEOF'
import sys
from pathlib import Path

out_dir = Path("data/real/quickdraw_png")
done = out_dir / ".done"
if done.exists():
    print("    Quick Draw already downloaded.")
    sys.exit(0)

CATEGORIES = [
    "cat", "dog", "house", "tree", "bicycle",
    "car", "fish", "bird", "flower", "sun",
]
MAX_PER = 2000

try:
    from quickdraw import QuickDrawData
    qd = QuickDrawData(recognized=True)

    total = 0
    for cat in CATEGORIES:
        cat_dir = out_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        drawings = qd.get_drawings(cat, count=MAX_PER)
        for i, drawing in enumerate(drawings):
            img = drawing.get_image(stroke_width=2).resize((256, 256))
            img.save(cat_dir / f"{cat}_{i:05d}.png")
            total += 1
        print(f"    {cat}: {i+1} images saved.")

    done.touch()
    print(f"    Quick Draw done: {total} total sketches.")

except ImportError:
    print("    [SKIP] pip install quickdraw  then re-run this script.")
PYEOF

echo ""
echo "=== Real dataset download complete ==="
echo "Images are in: $REAL_DIR"
echo ""
echo "Counts:"
find "$REAL_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l
