"""
Configuration for AI Image Detection Synthetic Dataset Pipeline.

Dataset Sources
---------------
REAL image datasets (publicly licensed, free to use):

  1. DIV2K  — Open access, no registration
       https://data.vision.ee.ethz.ch/cvl/DIV2K/
       900 high-resolution photos (train + validation HR), extremely clean and sharp
       Direct: DIV2K_train_HR.zip + DIV2K_valid_HR.zip

  2. COCO 2017 val  — CC BY 4.0
       https://cocodataset.org/#download
       5 000 natural scene images, ~1 GB
       Direct: wget http://images.cocodataset.org/zips/val2017.zip

  3. Open Images V7 (validation)  — CC BY 4.0
       https://storage.googleapis.com/openimages/web/index.html
       41 K images (validation subset only, ~1.2 GB)
       gsutil: gsutil -m rsync -r gs://open-images-dataset/validation .

  4. Google Quick, Draw!  — CC BY 4.0
       https://github.com/googlecreativelab/quickdraw-dataset
       50 M human sketches / doodles (28×28 bitmap .npy files per category)
       Direct: https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/

  5. Movie-Poster  — research dataset for artistic-style text detection
       https://github.com/AXNing/Artistic-style-text-detection
       1 500 movie posters with artistic-style titles
       Google Drive: https://drive.google.com/file/d/1anlWPsCX-6aYhUDqC33SXRufcpPpjLE2/view

AI-GENERATED image datasets (publicly licensed / research use):

  1. Synthbuster  — Open access, no registration  ★ 2024 新增
       https://zenodo.org/records/10066460
       ~9 000 張，720P–4K 高品質，手工挑選具欺騙性的圖片
       涵蓋：DALL-E 2, DALL-E 3, Adobe Firefly, Midjourney v5,
              Stable Diffusion 1.3 / 2 / XL, Glide
       Direct: wget https://zenodo.org/records/10066460/files/synthbuster.zip

  2. MS COCOAI (Defactify 2025)  — CC BY (繼承自 COCO), no registration  ★ 2025 新增
       https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset
       96 000 張（50% 真實 COCO，50% AI），balanced
       涵蓋：Stable Diffusion 3, SDXL, SD 2.1, DALL-E 3, Midjourney v6
       HuggingFace: datasets.load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
         → filter label != "real" for AI images only

  3. DiffusionDB  — CC BY 4.0
       https://huggingface.co/datasets/poloclub/diffusiondb
       14 M Stable Diffusion images; use modular subsets (1 K – 100 K):
         datasets.load_dataset("poloclub/diffusiondb", "large_random_10k")
       Native 512×512 PNG. No registration needed.

  4. ArtiFact  — Apache 2.0  (optional, large)
       https://www.kaggle.com/datasets/awsaf49/artifact-dataset
       2.5 M images from 25 generators. ~120 GB full.
       kaggle datasets download -d awsaf49/artifact-dataset

  5. GenImage  — CC BY-NC-SA 4.0 (non-commercial, optional, large)
       https://github.com/GenImage-Dataset/GenImage
       2.68 M images from 8 generators. ~500 GB full.
       python download_genimage.py  (resumable)
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
