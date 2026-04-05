#!/usr/bin/env bash
# ============================================================
# Download REAL image datasets into data/real/
# Run from the project root directory.
#
# Default downloads (~3 GB, no registration needed):
#   DIV2K        — 1 000 高畫質真實照片 (≥ 2K 解析度)
#   COCO 2017 val — 5 000 自然場景照片 (~1 GB)
#   Quick Draw!  — 10 類別 × 2 000 真人手繪素描
#   Movie-Poster — 1 500 張海報圖，含藝術字標題
# ============================================================
set -euo pipefail

REAL_DIR="data/real"
mkdir -p "$REAL_DIR"

echo "=== Downloading Real Image Datasets ==="

# ── 1. DIV2K (official download, open access, no login) ─────────────────────
# 900 high-resolution real photos, at least one dimension ≥ 2K.
# Diverse subjects: people, objects, nature, architecture, macro.
# No AI involvement — original photographs.
echo "[1/4] DIV2K — 900 high-resolution real photos (official download)..."
mkdir -p "$REAL_DIR/div2k"
if [ ! -f "$REAL_DIR/div2k/.done" ]; then
    if [ ! -f /tmp/DIV2K_train_HR.zip ]; then
        wget -q --show-progress \
            "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" \
            -O /tmp/DIV2K_train_HR.zip
    fi
    if [ ! -f /tmp/DIV2K_valid_HR.zip ]; then
        wget -q --show-progress \
            "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" \
            -O /tmp/DIV2K_valid_HR.zip
    fi

    unzip -oq /tmp/DIV2K_train_HR.zip -d "$REAL_DIR/div2k/"
    unzip -oq /tmp/DIV2K_valid_HR.zip -d "$REAL_DIR/div2k/"
    rm -f /tmp/DIV2K_train_HR.zip /tmp/DIV2K_valid_HR.zip
    touch "$REAL_DIR/div2k/.done"
    echo "    DIV2K done."
else
    echo "    DIV2K already downloaded."
fi

# ── 2. COCO 2017 Validation (~1 GB, CC BY 4.0) ──────────────────────────────
echo "[2/4] COCO 2017 val (5 000 natural scene images)..."
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
echo "[3/4] Quick Draw! human sketches (10 categories)..."
QUICKDRAW_RAW_DIR="$REAL_DIR/quickdraw"
QUICKDRAW_PNG_DIR="$REAL_DIR/quickdraw_png"
if [ ! -f "$QUICKDRAW_PNG_DIR/.done" ]; then
    mkdir -p "$QUICKDRAW_RAW_DIR"
    CATEGORIES=(cat dog house tree bicycle car fish bird flower sun)
    for category in "${CATEGORIES[@]}"; do
        if [ ! -f "$QUICKDRAW_RAW_DIR/${category}.npy" ]; then
            wget -q --show-progress \
                "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/${category}.npy" \
                -O "$QUICKDRAW_RAW_DIR/${category}.npy"
        fi
    done

    python3 scripts/quickdraw_to_png.py \
        --npy_dir "$QUICKDRAW_RAW_DIR" \
        --out_dir "$QUICKDRAW_PNG_DIR" \
        --max_per_category 2000 \
        --image_size 256

    touch "$QUICKDRAW_PNG_DIR/.done"
    echo "    Quick Draw done."
else
    echo "    Quick Draw already downloaded."
fi

# ── 4. Movie-Poster artistic text dataset (Google Drive) ───────────────────
echo "[4/4] Movie-Poster — 1 500 poster images with artistic text..."
bash scripts/download_movie_poster_dataset.sh

echo ""
echo "=== Real dataset download complete ==="
echo "Images are in: $REAL_DIR"
echo ""
echo "Counts:"
find "$REAL_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l
