"""
Add RGB ladder rungs to an existing dataset.

The luma-mode scrape only saves `luma.png` + `bppN.png` (grayscale).
For image-to-image RGB training we need `rgb_bppN.png` per sample.
This script walks every sample dir and adds those files using the
already-saved `rgb.png` as source.

Usage:
    python build_rgb_rungs.py --data ./data/cinematic_2k
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from posterize import posterize_rgb


def process_dir(sample_dir: Path) -> bool:
    rgb_path = sample_dir / "rgb.png"
    if not rgb_path.exists():
        return False
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for bpp in range(1, 9):
        out_path = sample_dir / f"rgb_bpp{bpp}.png"
        if out_path.exists():
            continue
        posterized = posterize_rgb(rgb, bpp)
        cv2.imwrite(str(out_path), cv2.cvtColor(posterized, cv2.COLOR_RGB2BGR))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()

    dirs = sorted(p for p in args.data.iterdir() if p.is_dir())
    print(f"processing {len(dirs)} samples in {args.data}")
    t0 = time.time()
    ok = 0
    skipped = 0
    for i, d in enumerate(dirs):
        if process_dir(d):
            ok += 1
        else:
            skipped += 1
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(dirs) - (i + 1)) / rate
            print(f"  {i+1:5d}/{len(dirs)}  ok={ok} skip={skipped}  {rate:5.1f}/s  eta {eta:5.0f}s")
    print(f"done. ok={ok} skipped={skipped}  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
