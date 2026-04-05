#!/usr/bin/env bash
set -euo pipefail

POSTER_DIR="data/real/movie_poster"
DONE_FILE="$POSTER_DIR/.done"
DRIVE_URL="https://drive.google.com/file/d/1anlWPsCX-6aYhUDqC33SXRufcpPpjLE2/view?usp=drive_link"
CACHE_DIR="$POSTER_DIR/.cache"
ARCHIVE_PATH="$CACHE_DIR/movie_poster.zip"

mkdir -p "$POSTER_DIR"
mkdir -p "$CACHE_DIR"

if [ -f "$DONE_FILE" ]; then
    echo "    Movie-Poster already downloaded."
    exit 0
fi

if ! python3 -c "import gdown" >/dev/null 2>&1; then
    python3 -m pip install -q gdown
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

python3 -m gdown --fuzzy "$DRIVE_URL" -O "$ARCHIVE_PATH"

python3 - "$ARCHIVE_PATH" "$POSTER_DIR" <<'PYEOF'
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

archive_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
extract_dir = archive_path.parent / "extract"
extract_dir.mkdir(parents=True, exist_ok=True)


def extract_archive(path: Path, destination: Path) -> None:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(destination)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(destination)
        return
    raise SystemExit(f"Unsupported archive format: {path.name}")


def count_images(root: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ignored = {"gt", "gts", "mask", "masks", "label", "labels", "annotation", "annotations"}
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in exts
        and not any(part.lower() in ignored for part in path.parts)
    )


extract_archive(archive_path, extract_dir)

for child in output_dir.iterdir():
    if child.name in {".done", ".cache"}:
        continue
    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()

for child in extract_dir.iterdir():
    shutil.move(str(child), str(output_dir / child.name))

img_count = count_images(output_dir)
print(f"    Movie-Poster extracted: {img_count} images in {output_dir}")
PYEOF

touch "$DONE_FILE"