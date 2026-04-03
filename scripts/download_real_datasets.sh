#!/usr/bin/env bash
# ============================================================
# Download REAL image datasets into data/real/
# Run from the project root directory.
# ============================================================
set -euo pipefail

REAL_DIR="data/real"
mkdir -p "$REAL_DIR"

echo "=== Downloading Real Image Datasets ==="

# ── 1. COCO 2017 Validation (5 000 images, ~1 GB) ────────────────────────────
# License: CC BY 4.0  |  https://cocodataset.org
echo "[1/3] COCO 2017 val (5 000 images)..."
mkdir -p "$REAL_DIR/coco"
if [ ! -f "$REAL_DIR/coco/.done" ]; then
    wget -q --show-progress \
        http://images.cocodataset.org/zips/val2017.zip \
        -O /tmp/coco_val2017.zip
    unzip -q /tmp/coco_val2017.zip -d "$REAL_DIR/coco/"
    rm /tmp/coco_val2017.zip
    touch "$REAL_DIR/coco/.done"
    echo "    COCO done."
else
    echo "    COCO already downloaded."
fi

# ── 2. Quick, Draw! PNG bitmaps (human sketches) ─────────────────────────────
# License: CC BY 4.0  |  https://github.com/googlecreativelab/quickdraw-dataset
# We download 5 categories as a small sample (~50 k sketches total).
echo "[2/3] Quick Draw! sketch bitmaps (5 categories)..."
mkdir -p "$REAL_DIR/quickdraw"
QD_BASE="https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
declare -a CATEGORIES=("cat" "dog" "house" "tree" "bicycle")
for cat in "${CATEGORIES[@]}"; do
    dest="$REAL_DIR/quickdraw/${cat}.npy"
    if [ ! -f "$dest" ]; then
        wget -q --show-progress "${QD_BASE}/${cat}.npy" -O "$dest"
        echo "    Downloaded ${cat}.npy"
    else
        echo "    ${cat}.npy already exists."
    fi
done
# Convert .npy sketches to PNG using the helper script
python3 scripts/quickdraw_to_png.py --npy_dir "$REAL_DIR/quickdraw" \
        --out_dir "$REAL_DIR/quickdraw_png" --max_per_category 2000
echo "    Quick Draw! done."

# ── 3. Open Images V7 (validation subset, via fiftyone) ──────────────────────
# License: CC BY 4.0  |  https://storage.googleapis.com/openimages/web/index.html
echo "[3/3] Open Images V7 validation (2 500 images via fiftyone)..."
python3 - <<'PYEOF'
try:
    import fiftyone.zoo as foz
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=2500,
        dataset_dir="data/real/open_images",
        label_types=[],
    )
    print(f"    Open Images: {len(ds)} images downloaded.")
except ImportError:
    print("    [SKIP] fiftyone not installed. Run: pip install fiftyone")
    print("           Then re-run this script.")
PYEOF

echo ""
echo "=== Real dataset download complete ==="
echo "Images are in: $REAL_DIR"
