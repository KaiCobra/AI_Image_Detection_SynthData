#!/usr/bin/env bash
# One-liner to download all public datasets (real + AI-generated).
# Usage: bash download_datasets.sh
set -euo pipefail
pip install -q Pillow numpy datasets huggingface_hub tqdm quickdraw
bash scripts/download_real_datasets.sh
bash scripts/download_ai_datasets.sh
