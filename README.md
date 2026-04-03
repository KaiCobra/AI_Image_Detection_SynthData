# AI Image Detection — Synthetic Dataset Pipeline

> Generate composite (real + AI) images with pixel-level binary masks for training
> AI-detection models. Built for poster-competition fairness verification.

---

## Overview

Each **batch** in the dataset contains two files:

| File | Description |
|---|---|
| `composite.png` | One large base image + one smaller image overlaid on top. One of the two is AI-generated, the other is real. Overlap alpha is randomised in `[0.55, 0.95]`. |
| `mask.png` | Binary mask (8-bit grayscale). **White (255) = AI-involved region**, Black (0) = real/human region. |
| `metadata.json` | Provenance, alpha, patch bounding box, source tags. |

---

## Datasets Used

### Real images (Set A)

| Dataset | License | Type | Count | Disk | Access |
|---|---|---|---|---|---|
| **PASCAL VOC 2012** | CC BY 2.5 | Natural photos | 22 500 | ~3 GB | `torchvision` / direct wget |
| **COCO 2017 val** | CC BY 4.0 | Natural scene photos | 5 000 | ~1 GB | direct wget |
| **Open Images V7** | CC BY 4.0 | Diverse photos | 41 K (val) | ~1.2 GB | `gsutil` / `fiftyone` |
| **Quick, Draw!** | CC BY 4.0 | Human sketches/doodles | 50 M (use subsets) | per-category .npy | `pip install quickdraw` |

### AI-generated images (Set B)

| Dataset | License | Generators | Count | Disk | Access |
|---|---|---|---|---|---|
| **CIFAKE** | CC BY 4.0 | Stable Diffusion v1.4 | 60 K fake | ~1 GB | HuggingFace (no login) |
| **DiffusionDB** | CC BY 4.0 | Stable Diffusion | 14 M (use subsets) | 10 K ≈ few MB | HuggingFace (no login) |
| **ArtiFact** | Apache 2.0 | 25 generators (13 GANs + 7 diffusion) | 2.5 M | ~120 GB | Kaggle |
| **GenImage** | CC BY-NC-SA 4.0 | 8 generators (MJ, SD, DALL-E 2…) | 2.68 M | ~500 GB | download script |

> **Recommended starter** (no registration, <5 GB, fully open-licensed):
> real = PASCAL VOC 2012 + COCO val + Quick Draw sketches;
> AI = CIFAKE fake split + DiffusionDB `large_random_10k`

---

## Quick Start

### 1. Install dependencies

```bash
pip install Pillow numpy datasets huggingface_hub tqdm
```

### 2. (Optional) Download real datasets

```bash
bash scripts/download_real_datasets.sh
```

This populates `data/real/` with COCO val images, Quick Draw PNGs, and
Open Images samples.

### 3. (Optional) Download AI-generated datasets

```bash
bash scripts/download_ai_datasets.sh
```

This populates `data/ai/` with CIFAKE FAKE images and ArtiFact samples.

> **Without downloaded data** the pipeline falls back to procedurally generated
> images (noise-based nature textures / sketch-like drawings for real;
> gradient dreamscapes / mandala / portrait-like patterns for AI).
> The 5 demo batches in `output/demo/` were produced this way.

### 4. Generate batches

```bash
# Generate 5 demo batches (default)
python generate_dataset.py

# Generate 1 000 batches for training
python generate_dataset.py --batches 1000 --output output/train --seed 100
```

### 5. Visualize a batch

```bash
python scripts/visualize_batch.py output/demo/batch_0000
# → saves output/demo/batch_0000/visualization.png (3-panel: composite | mask | overlay)
```

---

## Project Structure

```
.
├── config.py                   # Paths, alpha range, size, dataset URLs
├── generate_dataset.py         # Main pipeline entry point
├── datasets/
│   ├── real_images.py          # Real image loader + procedural fallback
│   └── ai_images.py            # AI image loader + procedural fallback
├── pipeline/
│   └── compositor.py           # Composite + mask generation logic
├── scripts/
│   ├── download_real_datasets.sh
│   ├── download_ai_datasets.sh
│   ├── quickdraw_to_png.py     # Convert Quick Draw .npy → PNG
│   └── visualize_batch.py      # 3-panel visualization
├── data/
│   ├── real/                   # Place real images here
│   └── ai/                     # Place AI images here
└── output/
    └── demo/                   # 5 demo batches (procedurally generated)
        ├── batch_0000/
        │   ├── composite.png
        │   ├── mask.png
        │   ├── metadata.json
        │   └── visualization.png
        ├── batch_0001/ ...
        └── batch_0004/ ...
```

---

## Compositing Logic

```
randomly pick: AI-image is BASE (large) or PATCH (small)

if AI is BASE:
    canvas ← AI image (512×512)
    real patch ← resize real image to [30%–60%] of canvas
    place real patch at random position with alpha ∈ [0.55, 0.95]
    mask: WHITE everywhere EXCEPT where real patch alpha > 0.5

if AI is PATCH:
    canvas ← real image (512×512)
    AI patch ← resize AI image to [30%–60%] of canvas
    place AI patch at random position with alpha ∈ [0.55, 0.95]
    mask: BLACK everywhere EXCEPT where AI patch alpha > 0.3 → WHITE
```

Patch edges are blurred (Gaussian) for a natural soft blend.

---

## Demo Batch Results

| Batch | AI role | Alpha | AI coverage |
|---|---|---|---|
| batch_0000 | patch (small) | 0.72 | 12.1% |
| batch_0001 | patch (small) | 0.61 | 27.6% |
| batch_0002 | base (large)  | 0.83 | 87.4% |
| batch_0003 | base (large)  | 0.64 | 87.1% |
| batch_0004 | base (large)  | 0.74 | 73.8% |
