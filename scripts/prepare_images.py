"""Cache HAM10000 images at half resolution (600x450 -> 400x300) to speed up JPEG decoding during training.

Usage: .venv/bin/python -I scripts/prepare_images.py data/raw/ham10000/images data/processed/ham10000_400
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

SIZE = (400, 300)


def resize(args):
    src, dst = args
    if dst.exists():
        return
    with Image.open(src) as im:
        im.convert("RGB").resize(SIZE, Image.BICUBIC).save(dst, quality=95)


def main(src_dir, dst_dir):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, dst_dir / p.name) for p in sorted(src_dir.glob("*.jpg"))]
    with ProcessPoolExecutor() as ex:
        list(ex.map(resize, jobs, chunksize=64))
    print(f"{len(jobs)} images -> {dst_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
