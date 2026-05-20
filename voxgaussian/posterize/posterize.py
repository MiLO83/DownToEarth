"""
Progressive bit-depth ladder posterizer.

The "reverse posterizing" idea (LYRA2_PROPOSAL.md §6.6.2): instead of
diffusing full RGB in one shot, decompose into a ladder of stages where
each stage operates at a higher bit-depth than the last:

    1 BPP (2 luma levels)  -> stage 1 output
    2 BPP (4 levels)       -> stage 2 (bounded refinement of stage 1)
    3 BPP (8 levels)       -> stage 3
    ...
    8 BPP (256 levels)     -> stage 8 (final posterized luma)

This module builds the static dataset from RGB inputs: for any image we
emit the full ladder of posterized luma maps. A diffusion model trained
on these pairs would learn the un-posterizing transform at each rung.

All math is uint8 / numpy and runs sub-millisecond per megapixel.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2  # SIMD-optimised LUT + cvtColor
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# Fixed-point Rec.709 coefficients that sum to 256 (0.5 LSB error worst case).
_LUMA_709_FP = np.array([54, 183, 19], dtype=np.uint16)


def rgb_to_luma_709(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> HxW uint8 Rec.709 luma. Integer-fast path."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    b = rgb[..., 2].astype(np.uint16)
    y = (r * _LUMA_709_FP[0] + g * _LUMA_709_FP[1] + b * _LUMA_709_FP[2]) >> 8
    return y.astype(np.uint8)


# Precompute LUTs for every BPP rung. Each LUT is 256 bytes; trivial RAM.
def _make_display_lut(bpp: int) -> np.ndarray:
    n_levels = 1 << bpp
    if n_levels == 256:
        return np.arange(256, dtype=np.uint8)
    grid = np.arange(256, dtype=np.float32)
    idx = np.clip(np.round(grid * (n_levels - 1) / 255.0), 0, n_levels - 1)
    return np.round(idx * 255.0 / (n_levels - 1)).astype(np.uint8)


def _make_index_lut(bpp: int) -> np.ndarray:
    n_levels = 1 << bpp
    grid = np.arange(256, dtype=np.float32)
    return np.clip(np.round(grid * (n_levels - 1) / 255.0), 0, n_levels - 1).astype(np.uint8)


_DISPLAY_LUTS = {bpp: _make_display_lut(bpp) for bpp in range(1, 9)}
_INDEX_LUTS = {bpp: _make_index_lut(bpp) for bpp in range(1, 9)}


def posterize_luma(luma: np.ndarray, bpp: int, with_index: bool = True):
    """
    HxW uint8 luma -> display_uint8, or (display_uint8, index_uint8) if with_index.

      display: posterized luma in [0,255] at 2**bpp evenly-spaced levels.
      index:   raw stage index in [0, 2**bpp - 1].

    bpp=1 -> binary (2 levels: pure black/white)
    bpp=2 -> 4 levels
    bpp=8 -> 256 levels (essentially identity)

    Uses cv2.LUT (SIMD) when available -- ~1 ms per 1080p plane.
    """
    if bpp < 1 or bpp > 8:
        raise ValueError(f"bpp must be 1..8, got {bpp}")
    if luma.dtype != np.uint8:
        luma = luma.astype(np.uint8)
    if _HAS_CV2:
        display = cv2.LUT(luma, _DISPLAY_LUTS[bpp])
        if with_index:
            return display, cv2.LUT(luma, _INDEX_LUTS[bpp])
        return display
    display = _DISPLAY_LUTS[bpp][luma]
    if with_index:
        return display, _INDEX_LUTS[bpp][luma]
    return display


def rgb_to_luma_fast(rgb: np.ndarray) -> np.ndarray:
    """
    Realtime-grade luma. Rec.601 weights via cv2.cvtColor (SIMD) when
    available -- ~10x faster than the exact Rec.709 integer path. The
    visual difference at 1-8 BPP posterization is undetectable.
    """
    if _HAS_CV2:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb_to_luma_709(rgb)


def posterize_rgb(rgb: np.ndarray, bpp: int) -> np.ndarray:
    """
    HxWx3 uint8 RGB -> HxWx3 uint8 RGB, each channel posterized
    independently at the same bpp.

    rung N => 2**N levels per channel => (2**N)**3 total colours.
        bpp 1 ->     8 colours
        bpp 2 ->    64 colours
        bpp 3 ->   512 colours
        bpp 4 ->  4096 colours
        bpp 5 ->  ~33 k colours
        bpp 8 ->  16.7 M colours (essentially identity)
    """
    if bpp < 1 or bpp > 8:
        raise ValueError(f"bpp must be 1..8, got {bpp}")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    lut = _DISPLAY_LUTS[bpp]
    if _HAS_CV2:
        # cv2.LUT broadcasts the 1D LUT to each channel
        return cv2.LUT(rgb, lut)
    return lut[rgb]


def posterize_rgb_ladder(
    rgb: np.ndarray,
    bpps=(1, 2, 3, 4, 5, 6, 7, 8),
) -> dict:
    """
    HxWx3 uint8 RGB -> dict { 'rgb' (input), bpp (int) -> HxWx3 uint8 rung }.

    Per-channel posterization is the right framing for image-to-image
    color refinement: each RGB byte is independently snapped to the
    rung's 2**bpp levels, so a 3-channel deposterizer can recover full
    RGB by walking the rung ladder.
    """
    out = {"rgb": rgb}
    for bpp in bpps:
        out[bpp] = posterize_rgb(rgb, bpp)
    return out


def posterize_luma_ladder(
    rgb: np.ndarray,
    bpps=(1, 2, 3, 4, 5, 6, 7, 8),
    fast: bool = False,
    with_index: bool = True,
) -> dict:
    """
    HxWx3 uint8 RGB -> dict with keys:
        'luma':    HxW uint8 source luma
        'rgb':     HxWx3 uint8 input (reference, not copied if fast)
        bpp (int): {'display': HxW uint8, 'index': HxW uint8?}

    fast=False (default): exact Rec.709 + index maps  (dataset-grade)
    fast=True            : Rec.601 + cv2.LUT, display only  (realtime, ~7 ms 1080p)
    """
    if fast:
        luma = rgb_to_luma_fast(rgb)
        out = {"luma": luma, "rgb": rgb}
        for bpp in bpps:
            display = posterize_luma(luma, bpp, with_index=False)
            out[bpp] = {"display": display}
        return out
    luma = rgb_to_luma_709(rgb)
    out = {"luma": luma, "rgb": rgb.copy()}
    for bpp in bpps:
        if with_index:
            display, index = posterize_luma(luma, bpp, with_index=True)
            out[bpp] = {"display": display, "index": index}
        else:
            out[bpp] = {"display": posterize_luma(luma, bpp, with_index=False)}
    return out


def delta_pair(prev_display: np.ndarray, next_display: np.ndarray) -> np.ndarray:
    """
    Signed per-pixel delta between two rungs of the ladder.

    Used to construct the bounded-deviation training signal: stage N+1
    learns to predict (next_display - prev_display) which lives in a
    tiny range. Returned as int16 since signed deltas can hit +/-255.
    """
    return next_display.astype(np.int16) - prev_display.astype(np.int16)


def ladder_to_grid(ladder: dict, cols: int = 3) -> np.ndarray:
    """
    Pack ladder rungs into a single grid image for visual inspection.

    Layout (default cols=3, 9 panels): RGB | 1BPP | 2BPP / 3BPP | 4BPP | 5BPP / 6BPP | 7BPP | 8BPP
    Returns HxWx3 uint8.
    """
    bpp_keys = sorted([k for k in ladder.keys() if isinstance(k, int)])
    h, w = ladder["luma"].shape
    panels = [ladder["rgb"]]
    for bpp in bpp_keys:
        d = ladder[bpp]["display"]
        panels.append(np.stack([d, d, d], axis=-1))
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="Posterize an image to the bit-depth ladder.")
    ap.add_argument("image", help="input image path")
    ap.add_argument("--out", default="ladder.png", help="output grid PNG path")
    args = ap.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ladder = posterize_luma_ladder(rgb)
    grid = ladder_to_grid(ladder)
    cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.out}  {grid.shape}")
