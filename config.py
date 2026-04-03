"""
Configuration for AI Image Detection Synthetic Dataset Pipeline.

Dataset Sources
---------------
REAL image datasets (publicly licensed, free to use):

  1. PASCAL VOC 2012  — CC BY 2.5
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
       22 500 natural photos, ~3 GB, no registration required
       torchvision: VOCDetection(root="./data", year="2012", download=True)
       Direct: wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

  2. COCO 2017 val  — CC BY 4.0
       https://cocodataset.org/#download
       5 000 natural scene images, ~1 GB
       Direct: wget http://images.cocodataset.org/zips/val2017.zip
       fiftyone: foz.load_zoo_dataset("coco-2017", split="validation")

  3. Open Images V7 (validation)  — CC BY 4.0
       https://storage.googleapis.com/openimages/web/index.html
       41 K images (validation subset only, ~1.2 GB)
       gsutil: gsutil -m rsync -r gs://open-images-dataset/validation .
       fiftyone: foz.load_zoo_dataset("open-images-v7", split="validation", max_samples=5000)

  4. Google Quick, Draw!  — CC BY 4.0
       https://github.com/googlecreativelab/quickdraw-dataset
       50 M human sketches / doodles (28×28 bitmap .npy files per category)
       pip: pip install quickdraw
       Per-category: gsutil cp gs://quickdraw_dataset/full/numpy_bitmap/cat.npy .
       HuggingFace: datasets.load_dataset("google/quickdraw", "cat")

AI-GENERATED image datasets (publicly licensed / research use):

  1. CIFAKE  — CC BY 4.0
       https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset
       60 K AI-generated images (Stable Diffusion v1.4 at 32×32, upscalable)
       No registration needed (HuggingFace mirror):
         datasets.load_dataset("dragonintelligence/CIFAKE-image-dataset")
         → filter label==1 for FAKE images
       Kaggle: kaggle datasets download birdy654/cifake-real-and-ai-generated-synthetic-images

  2. DiffusionDB  — CC BY 4.0
       https://huggingface.co/datasets/poloclub/diffusiondb
       14 M Stable Diffusion images; use modular subsets (1 K – 100 K):
         datasets.load_dataset("poloclub/diffusiondb", "large_random_10k")
       Native 512×512 PNG. No registration needed.

  3. ArtiFact  — Apache 2.0
       https://www.kaggle.com/datasets/awsaf49/artifact-dataset
       2.5 M images from 25 generators (13 GANs + 7 diffusion models incl.
       SD, DALL-E, Midjourney). ~120 GB full.
       kaggle datasets download -d awsaf49/artifact-dataset

  4. GenImage  — CC BY-NC-SA 4.0 (non-commercial)
       https://github.com/GenImage-Dataset/GenImage
       2.68 M images from 8 generators (Midjourney, SD 1.4/1.5/XL,
       DALL-E 2, ADM, GLIDE, Wukong). ~500 GB full.
       python download_genimage.py  (resumable; download specific subfolders only)
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
