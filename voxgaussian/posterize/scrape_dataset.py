"""
Scrape images and build the bit-depth ladder dataset.

Sources:
  - Unsplash random API:  source.unsplash.com/random/{W}x{H}
  - URL list file (one URL per line)
  - Local directory (glob)

For each input image, writes:
  data/<id>/rgb.png         -- original
  data/<id>/luma.png        -- Rec.709 luma
  data/<id>/bpp1.png ...    -- posterized at each rung
  data/<id>/ladder.png      -- grid preview

Optionally also packs everything into one .npz with shape (N, 9, H, W)
for fast training-time loading.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from posterize import posterize_luma_ladder, ladder_to_grid


def fetch_random(width: int, height: int, timeout: float = 30.0) -> np.ndarray:
    """Random image from picsum.photos. Returns HxWx3 uint8 RGB."""
    url = f"https://picsum.photos/{width}/{height}"
    r = requests.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return np.asarray(img)


def fetch_and_save(
    index: int,
    width: int,
    height: int,
    out_dir: Path,
    no_grid: bool,
    rec709_exact: bool,
    retries: int = 3,
) -> tuple[int, str, str]:
    """Worker: fetch a random image and save its full ladder. Returns status row."""
    for attempt in range(retries):
        try:
            rgb = fetch_random(width, height)
            label = f"cine_{uuid.uuid4().hex[:10]}"
            ladder = posterize_luma_ladder(rgb, fast=not rec709_exact)
            save_ladder(out_dir / label, ladder, save_grid=not no_grid)
            return (index, label, "ok")
        except Exception as e:
            if attempt == retries - 1:
                return (index, "", f"fail: {type(e).__name__}: {e}")
            time.sleep(0.5 + attempt)
    return (index, "", "fail: exhausted")


def fetch_url(url: str, timeout: float = 15.0) -> np.ndarray:
    r = requests.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return np.asarray(img)


def load_local(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_to(rgb: np.ndarray, target: int | None) -> np.ndarray:
    if target is None:
        return rgb
    h, w = rgb.shape[:2]
    if max(h, w) <= target:
        return rgb
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_ladder(out_dir: Path, ladder: dict, save_grid: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(ladder["rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "luma.png"), ladder["luma"])
    for bpp in [k for k in ladder if isinstance(k, int)]:
        cv2.imwrite(str(out_dir / f"bpp{bpp}.png"), ladder[bpp]["display"])
    if save_grid:
        grid = ladder_to_grid(ladder)
        cv2.imwrite(str(out_dir / "ladder.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def pack_npz(samples: list[dict], out_path: Path) -> None:
    """
    Pack ladder samples to a single npz.
      rgb:   (N, H, W, 3) uint8
      luma:  (N, H, W)    uint8
      bpp:   (N, 8, H, W) uint8 -- rungs 1..8 stacked along axis=1
    """
    if not samples:
        return
    h, w = samples[0]["luma"].shape
    rgb = np.stack([s["rgb"] for s in samples])
    luma = np.stack([s["luma"] for s in samples])
    bpp_stack = np.stack(
        [np.stack([s[b]["display"] for b in range(1, 9)]) for s in samples]
    )
    np.savez_compressed(out_path, rgb=rgb, luma=luma, bpp=bpp_stack)
    print(f"packed {len(samples)} samples -> {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Scrape images + build bit-depth ladder pairs.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--count", type=int, help="random picsum.photos images to fetch")
    src.add_argument("--urls", type=Path, help="file with one URL per line")
    src.add_argument("--dir", type=Path, help="local directory of images")
    ap.add_argument("--width", type=int, default=1024, help="output width  (cinematic default 1024)")
    ap.add_argument("--height", type=int, default=448, help="output height (cinematic default 448 = ~2.29:1)")
    ap.add_argument("--size", type=int, default=None, help="square shorthand: sets both width=height=size")
    ap.add_argument("--out", type=Path, default=Path("./data/posterize_ladder"))
    ap.add_argument("--npz", action="store_true", help="also pack into a single .npz (only useful for small batches; memory-heavy)")
    ap.add_argument("--no-grid", action="store_true", help="skip ladder.png preview")
    ap.add_argument("--workers", type=int, default=8, help="parallel fetch workers (random mode only)")
    ap.add_argument("--rec709-exact", action="store_true",
                    help="use exact Rec.709 luma + index maps (slower); default uses fast Rec.601 + display-only")
    args = ap.parse_args()

    if args.size is not None:
        args.width = args.height = args.size

    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- Random-fetch fast path: ThreadPoolExecutor across workers ---
    if args.count is not None:
        n = args.count
        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    fetch_and_save, i, args.width, args.height,
                    args.out, args.no_grid, args.rec709_exact,
                )
                for i in range(n)
            ]
            for fut in as_completed(futures):
                i, label, status = fut.result()
                if status == "ok":
                    ok += 1
                    if ok % 25 == 0 or ok == n:
                        elapsed = time.time() - t_start
                        rate = ok / elapsed if elapsed > 0 else 0.0
                        eta = (n - ok) / rate if rate > 0 else float("inf")
                        print(f"[{ok:5d}/{n}] ok={ok} fail={fail}  {rate:5.1f} img/s  eta {eta:6.0f}s")
                else:
                    fail += 1
                    print(f"[{i}] {status}", file=sys.stderr)
        print(f"done. ok={ok} fail={fail}  {time.time() - t_start:.1f}s total  -> {args.out}")
        return

    # --- URL list / local dir: serial (typically small batches) ---
    sources: list = []
    if args.urls is not None:
        sources = [u.strip() for u in args.urls.read_text().splitlines() if u.strip()]
    elif args.dir is not None:
        sources = sorted(
            p for p in args.dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )

    samples = []
    for i, src in enumerate(sources):
        t0 = time.time()
        try:
            if isinstance(src, Path):
                rgb = load_local(src)
                label = src.stem
            else:
                rgb = fetch_url(src)
                label = f"url_{i:05d}_{uuid.uuid4().hex[:6]}"
        except Exception as e:
            print(f"[{i}] FAIL: {e}", file=sys.stderr)
            continue

        rgb = resize_to(rgb, max(args.width, args.height))
        ladder = posterize_luma_ladder(rgb, fast=not args.rec709_exact)
        save_ladder(args.out / label, ladder, save_grid=not args.no_grid)
        if args.npz:
            samples.append(ladder)
        print(f"[{i}] {label:40s} {rgb.shape[1]}x{rgb.shape[0]}  {time.time()-t0:.2f}s")

    if args.npz and samples:
        pack_npz(samples, args.out / "ladder.npz")


if __name__ == "__main__":
    main()
