#!/usr/bin/env bash
# ============================================================
# Download REAL image datasets into data/real/
# Run from the project root directory.
#
# Recommended starter combination (~25 GB, fully open-licensed):
#   PASCAL VOC 2012  — CC BY 2.5   (22 500 photos, ~3 GB)
#   COCO 2017 val    — CC BY 4.0   (5 000 photos,  ~1 GB)
#   Quick Draw!      — CC BY 4.0   (10 categories × 2 000 sketches)
# ============================================================
set -euo pipefail

REAL_DIR="data/real"
mkdir -p "$REAL_DIR"

echo "=== Downloading Real Image Datasets ==="

# ── 1. PASCAL VOC 2012 (~3 GB, CC BY 2.5) ───────────────────────────────────
echo "[1/3] PASCAL VOC 2012 (22 500 natural photos)..."
mkdir -p "$REAL_DIR/voc2012"
if [ ! -f "$REAL_DIR/voc2012/.done" ]; then
    wget -q --show-progress \
        https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar \
        -O /tmp/voc2012.tar
    tar xf /tmp/voc2012.tar -C "$REAL_DIR/voc2012/" --strip-components=1
    rm /tmp/voc2012.tar
    # Flatten JPEGImages into voc2012/ for easy glob
    find "$REAL_DIR/voc2012" -name "*.jpg" -exec cp {} "$REAL_DIR/voc2012/" \;
    touch "$REAL_DIR/voc2012/.done"
    echo "    PASCAL VOC 2012 done."
else
    echo "    PASCAL VOC 2012 already downloaded."
fi

# ── 2. COCO 2017 Validation (~1 GB, CC BY 4.0) ──────────────────────────────
echo "[2/3] COCO 2017 val (5 000 images)..."
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
    print("    Alternatively use per-category .npy files:")
    print("      gsutil cp gs://quickdraw_dataset/full/numpy_bitmap/cat.npy data/real/quickdraw/")
    print("      python3 scripts/quickdraw_to_png.py --npy_dir data/real/quickdraw --out_dir data/real/quickdraw_png")
PYEOF

echo ""
echo "=== Real dataset download complete ==="
echo "Images are in: $REAL_DIR"
echo ""
echo "Counts:"
find "$REAL_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l
