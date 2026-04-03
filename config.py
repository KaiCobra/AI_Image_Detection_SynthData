"""
Configuration for AI Image Detection Synthetic Dataset Pipeline.

Dataset Sources
---------------
REAL image datasets (publicly licensed, free to use):
  1. COCO 2017  — CC BY 4.0
       https://cocodataset.org/#download
       ~118 K training images, diverse everyday scenes
       Download: wget http://images.cocodataset.org/zips/val2017.zip

  2. Open Images Dataset V7  — CC BY 4.0
       https://storage.googleapis.com/openimages/web/index.html
       9 M images; pip install openimages  →  openimages download
       or: aws s3 --no-sign-request sync s3://open-images-dataset/validation .

  3. Google Quick, Draw!  — CC BY 4.0
       https://github.com/googlecreativelab/quickdraw-dataset
       50 M human sketches / doodles (PNG bitmaps available)
       Download: https://console.cloud.google.com/storage/browser/quickdraw_dataset

AI-GENERATED image datasets (publicly licensed / research use):
  1. CIFAKE  — CC0 / Research use
       https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
       60 K AI-generated images (CIFAR-10 size, 32×32 upscalable)
       Kaggle: kaggle datasets download birdy654/cifake-real-and-ai-generated-synthetic-images
       HuggingFace: datasets.load_dataset("jlbaker61/CIFAKE")  (split="FAKE")

  2. ArtiFact  — CC BY 4.0
       https://huggingface.co/datasets/awsaf49/artifact
       ~2.5 M images from 27 AI generators (SD, DALL-E, Midjourney, GAN…)
       HuggingFace: datasets.load_dataset("awsaf49/artifact", streaming=True)

  3. GenImage  — Research use
       https://github.com/GenImage-Dataset/GenImage
       1.3 M images from 8 generators (Midjourney, SD, DALL-E 2, etc.)
       Download via paper instructions or direct Google Drive links in the repo.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "data"
REAL_DIR      = DATA_DIR / "real"        # put real images here
AI_DIR        = DATA_DIR / "ai"          # put AI-generated images here
OUTPUT_DIR    = ROOT_DIR / "output"
DEMO_DIR      = OUTPUT_DIR / "demo"     # 5 demo batches (local generation)

# ── Composite parameters ────────────────────────────────────────────────────
OUTPUT_SIZE   = (512, 512)              # final composite image size (W, H)

# Overlay alpha range: AI patch will be blended with this opacity range.
# Range [0.55, 0.95] keeps the AI region clearly visible (not transparent).
ALPHA_MIN     = 0.55
ALPHA_MAX     = 0.95

# Size ratio of the smaller image relative to the larger one.
# e.g. 0.3 → the small patch occupies 30–60% of the canvas.
SMALL_RATIO_MIN = 0.30
SMALL_RATIO_MAX = 0.60

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED   = 42

# ── Demo (offline) generation ───────────────────────────────────────────────
# When real dataset directories are empty the pipeline falls back to
# procedurally generated images so the pipeline can be validated locally.
DEMO_BATCHES  = 5
