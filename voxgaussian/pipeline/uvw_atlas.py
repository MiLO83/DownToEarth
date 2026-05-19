"""
uvw_atlas.py — Bidirectional voxel ↔ RGB atlas with zero-cost identity.

DESIGN
======

A 256³ voxel grid packs perfectly into a 4096² atlas because 256³ = 4096².
We lay it out as a 16×16 tile grid where each tile is a single w-slice of
size 256×256:

    atlas_x = (w &  15) * 256 + u          # tile column = low nibble of w
    atlas_y = (w >>  4) * 256 + v          # tile row    = high nibble of w

The bijection (u,v,w) ↔ (atlas_x, atlas_y) is purely arithmetic.
The *identity* mapping — "color of pixel = voxel coord of pixel" — therefore
costs ZERO bytes of storage; a 5-line shader computes it per-pixel:

    out vec4 fragColor;
    in  vec3 voxelCoord;        // (0..256, 0..256, 0..256)
    void main() {
        fragColor = vec4(voxelCoord / 256.0, 1.0);
    }

VRAM is spent only on *payload*. This module produces two payload atlases:

  histogram_atlas[c]   shape (n_classes, 4096, 4096), uint8
                       Ground-truth vote-count per (voxel, class).
                       Read-cold (only during refinement writes).

  summary_atlas        shape (4096, 4096, 4), uint8 RGBA
                       Pre-computed digest of the histogram.
                       Read-hot — every renderer / view-selector / ray-carver
                       gets everything they need in one texture fetch:

                         R = mode_class_id        (argmax over histogram)
                         G = mode_confidence      (top_count / total) * 255
                         B = total_obs_log        clamp(log2(total) * 16, 0, 255)
                         A = ambiguity_margin     (top1_frac - top2_frac) * 255

The B (log of observations) byte is the easy-to-miss one: high-confidence
voxels with 3 votes look identical to high-confidence voxels with 300
without it, and active-view-selection over-trusts undersampled cells.

The A (ambiguity margin) byte makes active-view-selection a single pass:
priority = (1 - A/255) * (1 - B/255)  →  reduce-sum inside camera frustum.

CONSUMING THE ATLAS
===================

Forward (voxel → summary):
    x, y = voxel_to_atlas(u, v, w)
    r, g, b, a = summary_atlas[y, x]

Backward (canonical-coord RGB → voxel):
    u, v, w = rgb  # literally, the bytes ARE the coord. no math.
    x, y = voxel_to_atlas(u, v, w)
    r, g, b, a = summary_atlas[y, x]

The "color is a coord" property means that any framebuffer rendered with the
identity shader becomes an instant lookup key for the summary atlas — no
inverse projection, no raycast through a sparse data structure.

LICENSE
=======
MIT. Original work by MiLO + Opie (2026-05-19).

Implements three long-standing graphics primitives:
  • G-buffer position encoding (Crassin et al. 2008, every deferred renderer)
  • Space-filling-curve volume packing (GigaVoxels et al.)
  • Derived-data cache atlas (every game engine since 2010)
recombined for voxel-occupancy data structures. No NVIDIA Lyra code,
no GPL deps. Ship freely.
"""
from __future__ import annotations

from typing import Iterable
import math
import numpy as np


# ─── Configuration ─────────────────────────────────────────────────────────

VOXEL_RES = 256                # voxels per axis (256³)
ATLAS_SIZE = 4096              # atlas dimension (4096² = 256³ = 16.7M)
TILE_SIZE = 256                # one w-slice per tile
TILES_PER_ROW = 16             # 16 × 16 = 256 tiles total

assert TILE_SIZE * TILES_PER_ROW == ATLAS_SIZE
assert VOXEL_RES == TILE_SIZE
assert TILES_PER_ROW * TILES_PER_ROW == VOXEL_RES


# ─── Bijection: voxel ↔ atlas position ─────────────────────────────────────

def voxel_to_atlas(u: int, v: int, w: int) -> tuple[int, int]:
    """(u, v, w) → (atlas_x, atlas_y). O(1), pure arithmetic.

    >>> voxel_to_atlas(0, 0, 0)
    (0, 0)
    >>> voxel_to_atlas(255, 255, 255)
    (4095, 4095)
    >>> voxel_to_atlas(128, 128, 0)
    (128, 128)
    >>> voxel_to_atlas(0, 0, 16)
    (0, 256)
    >>> voxel_to_atlas(0, 0, 17)
    (256, 256)
    """
    return ((w & 15) << 8) | u, ((w >> 4) << 8) | v


def atlas_to_voxel(atlas_x: int, atlas_y: int) -> tuple[int, int, int]:
    """(atlas_x, atlas_y) → (u, v, w). The inverse of voxel_to_atlas. O(1).

    >>> atlas_to_voxel(0, 0)
    (0, 0, 0)
    >>> atlas_to_voxel(4095, 4095)
    (255, 255, 255)
    >>> all(atlas_to_voxel(*voxel_to_atlas(u, v, w)) == (u, v, w)
    ...     for u in range(0, 256, 37)
    ...     for v in range(0, 256, 37)
    ...     for w in range(0, 256, 37))
    True
    """
    u = atlas_x & 255
    v = atlas_y & 255
    w = (atlas_y >> 8) * TILES_PER_ROW + (atlas_x >> 8)
    return u, v, w


def voxel_to_atlas_vec(uvw: np.ndarray) -> np.ndarray:
    """Vectorised voxel_to_atlas for an (N, 3) array of int coords.

    Returns an (N, 2) int array of (atlas_x, atlas_y).
    """
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    x = ((w & 15) << 8) | u
    y = ((w >> 4) << 8) | v
    return np.stack([x, y], axis=1)


# ─── Summary atlas: digest of the per-voxel class histogram ────────────────

def build_summary_atlas(
    histogram: dict[tuple[int, int, int], dict[int, int]],
    n_classes: int = 11,
) -> np.ndarray:
    """Build a 4096×4096 RGBA8 summary atlas from a sparse voxel histogram.

    Args:
        histogram: sparse {(u, v, w): {class_id: vote_count}} as produced by
                   voxgaussian's VoxelStore.
        n_classes: total number of classes in the taxonomy (default 11).

    Returns:
        ndarray shape (4096, 4096, 4), dtype uint8, channels = (R, G, B, A):
          R = mode_class_id
          G = mode_confidence × 255
          B = clamp(log2(total_votes) × 16, 0, 255)
          A = ambiguity_margin × 255
    """
    out = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.uint8)

    for (u, v, w), counts in histogram.items():
        if not counts:
            continue

        # Sort classes by count, descending. counts is small (n_classes ≤ 32).
        sorted_pairs = sorted(counts.items(), key=lambda kv: -kv[1])
        top_class, top_count = sorted_pairs[0]
        runner_count = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0
        total = sum(counts.values())

        if total == 0:
            continue

        confidence = top_count / total
        margin = (top_count - runner_count) / total
        obs_log = min(255, int(math.log2(total + 1) * 16))

        x, y = voxel_to_atlas(u, v, w)
        out[y, x, 0] = top_class & 0xFF
        out[y, x, 1] = int(confidence * 255)
        out[y, x, 2] = obs_log
        out[y, x, 3] = int(margin * 255)

    return out


def build_summary_atlas_dense(
    histogram_array: np.ndarray,
) -> np.ndarray:
    """Same as build_summary_atlas, but takes a dense (256, 256, 256, n_classes)
    int16 histogram instead of a sparse dict. Faster for filled grids.

    Args:
        histogram_array: (256, 256, 256, n_classes) int16 vote counts.
                         histogram_array[u, v, w, c] = votes for class c at voxel (u,v,w).

    Returns:
        (4096, 4096, 4) uint8 RGBA summary atlas.
    """
    if histogram_array.shape[:3] != (VOXEL_RES, VOXEL_RES, VOXEL_RES):
        raise ValueError(
            f"Expected histogram shape (256,256,256,n_classes), got {histogram_array.shape}"
        )

    total = histogram_array.sum(axis=-1)                          # (256,256,256)
    occupied_mask = total > 0
    sorted_idx = np.argsort(-histogram_array, axis=-1)            # (256,256,256,n_classes)
    top_class = sorted_idx[..., 0]                                # (256,256,256)
    top_count = np.take_along_axis(histogram_array, sorted_idx[..., :1], axis=-1).squeeze(-1)
    runner_count = (
        np.take_along_axis(histogram_array, sorted_idx[..., 1:2], axis=-1).squeeze(-1)
        if histogram_array.shape[-1] > 1
        else np.zeros_like(top_count)
    )

    safe_total = np.where(occupied_mask, total, 1)                # avoid div-by-zero
    confidence = (top_count / safe_total * 255).clip(0, 255).astype(np.uint8)
    margin = ((top_count - runner_count) / safe_total * 255).clip(0, 255).astype(np.uint8)
    obs_log = (np.log2(total.astype(np.float32) + 1) * 16).clip(0, 255).astype(np.uint8)
    top_class = (top_class * occupied_mask).astype(np.uint8)

    # Scatter the dense (256,256,256,4) into the tiled (4096,4096,4) layout.
    out = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.uint8)
    # Vectorised tile placement: for each w, copy a (256,256,4) slab.
    for w in range(VOXEL_RES):
        tile_x = (w & 15) * TILE_SIZE
        tile_y = (w >> 4) * TILE_SIZE
        out[tile_y:tile_y + TILE_SIZE, tile_x:tile_x + TILE_SIZE, 0] = top_class[:, :, w].T
        out[tile_y:tile_y + TILE_SIZE, tile_x:tile_x + TILE_SIZE, 1] = confidence[:, :, w].T
        out[tile_y:tile_y + TILE_SIZE, tile_x:tile_x + TILE_SIZE, 2] = obs_log[:, :, w].T
        out[tile_y:tile_y + TILE_SIZE, tile_x:tile_x + TILE_SIZE, 3] = margin[:, :, w].T

    return out


# ─── PNG export (for viewer consumption) ───────────────────────────────────

def export_atlas_png(atlas: np.ndarray, path: str) -> None:
    """Write an RGBA8 atlas to a PNG. Requires Pillow."""
    from PIL import Image
    if atlas.dtype != np.uint8:
        raise ValueError(f"Atlas must be uint8, got {atlas.dtype}")
    if atlas.ndim != 3 or atlas.shape[2] not in (3, 4):
        raise ValueError(f"Atlas must be HxWx3 or HxWx4, got {atlas.shape}")
    mode = "RGBA" if atlas.shape[2] == 4 else "RGB"
    Image.fromarray(atlas, mode=mode).save(path, format="PNG", optimize=False)


# ─── Active-view priority (one of the consumers the atlas serves) ──────────

def view_priority_from_summary(summary_atlas: np.ndarray) -> np.ndarray:
    """Compute per-voxel active-view priority from the summary atlas.

    High priority = voxels we should point a camera at next:
      • controversial classification (low margin)
      • undersampled (low total observations)
      • not yet converged (skip if confidence is already high & well-sampled)

    Returns:
        (256, 256, 256) float32 priority scores in [0, 1].
    """
    # Re-untile the summary atlas back to a (256, 256, 256, 4) cube view.
    cube = np.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES, 4), dtype=np.uint8)
    for w in range(VOXEL_RES):
        tile_x = (w & 15) * TILE_SIZE
        tile_y = (w >> 4) * TILE_SIZE
        cube[:, :, w, :] = summary_atlas[
            tile_y:tile_y + TILE_SIZE,
            tile_x:tile_x + TILE_SIZE,
            :,
        ].transpose(1, 0, 2)

    margin = cube[..., 3].astype(np.float32) / 255.0
    obs_log = cube[..., 2].astype(np.float32) / 255.0
    return (1.0 - margin) * (1.0 - obs_log)


# ─── Self-test ─────────────────────────────────────────────────────────────

def verify_bijection() -> bool:
    """Verify (atlas_to_voxel ∘ voxel_to_atlas) == identity, exhaustively.

    Returns True iff the bijection holds for all 256³ coordinates.
    """
    for u in range(VOXEL_RES):
        for v in range(VOXEL_RES):
            for w in range(VOXEL_RES):
                x, y = voxel_to_atlas(u, v, w)
                if (u, v, w) != atlas_to_voxel(x, y):
                    return False
                if not (0 <= x < ATLAS_SIZE and 0 <= y < ATLAS_SIZE):
                    return False
    return True


if __name__ == "__main__":
    import doctest
    failures, tests = doctest.testmod(verbose=False)
    print(f"doctest: {tests - failures}/{tests} passed")

    print("Exhaustive bijection check (256³ pairs)...", end=" ", flush=True)
    print("OK" if verify_bijection() else "FAILED")

    print("\nQuick synthetic build:")
    rng = np.random.default_rng(0)
    # 8 randomly-classed voxels with varying confidence
    hist = {
        (rng.integers(256), rng.integers(256), rng.integers(256)):
            {int(c): int(n) for c, n in zip(
                rng.choice(11, size=3, replace=False),
                rng.integers(1, 50, size=3),
            )}
        for _ in range(8)
    }
    atlas = build_summary_atlas(hist)
    print(f"  summary atlas: shape={atlas.shape} dtype={atlas.dtype}")
    print(f"  non-zero pixels: {(atlas[..., 1] > 0).sum()}")
