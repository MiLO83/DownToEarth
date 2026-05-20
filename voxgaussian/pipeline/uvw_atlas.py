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


# ─── Occupancy bitmap — same-grid 1-bit-per-voxel empty/full toggle ───────


class OccupancyBitmap:
    """Same-resolution 1-bit-per-voxel companion to the summary atlas.

    Stores one bit per voxel answering "is this slot populated?". Lets the
    shader skip the 4-byte main-atlas read for empty voxels via a single-bit
    test that's 24-32× cheaper in storage *and* bandwidth.

    Memory math (the headline numbers):

      Grid       Voxels      RGBA8 atlas    RGB only    1-bit mask    Ratio
      256³       16.7M       67 MB          50 MB       2.1 MB        24×
      512³       134M        537 MB         403 MB      16.8 MB       24×
      1024³      1.07B       4.3 GB         3.2 GB      134 MB        24×

    Layout: 8 voxels packed per byte along the X axis of the 2D atlas
    (preserves cache locality for horizontal-scan access patterns).
    Atlas dims must be divisible by 8 in the X direction.

    For res=256 (atlas 4096×4096): 4096/8 = 512 bytes per row, 4096 rows,
    total 2 MB. Fits trivially in any GPU's L2 cache.

    GLSL consumption (one texelFetch + shift + AND):
        uniform usampler2D occupancyBitmap;
        bool is_occupied(ivec2 atlas_xy) {
            uint byte_v = texelFetch(
                occupancyBitmap, ivec2(atlas_xy.x >> 3, atlas_xy.y), 0
            ).r;
            return ((byte_v >> (atlas_xy.x & 7)) & 1u) != 0u;
        }

    >>> bmap = OccupancyBitmap(atlas_w=64, atlas_h=64)
    >>> bmap.nbytes
    512
    >>> bmap.get(0, 0)
    False
    >>> bmap.set(0, 0)
    >>> bmap.get(0, 0)
    True
    >>> bmap.set(63, 63)
    >>> bmap.count_occupied()
    2
    >>> bmap.set(0, 0, False)
    >>> bmap.count_occupied()
    1
    """

    def __init__(self, atlas_w: int = ATLAS_SIZE, atlas_h: int = ATLAS_SIZE):
        if atlas_w % 8 != 0:
            raise ValueError(
                f"atlas_w must be divisible by 8 for X-axis bit-packing, got {atlas_w}"
            )
        self.atlas_w = atlas_w
        self.atlas_h = atlas_h
        # Shape (H, W/8) uint8 — each byte holds 8 horizontally-adjacent voxels.
        self.bits = np.zeros((atlas_h, atlas_w // 8), dtype=np.uint8)

    def __repr__(self) -> str:
        n = self.count_occupied()
        total = self.atlas_w * self.atlas_h
        pct = (100.0 * n / total) if total > 0 else 0.0
        return (f"OccupancyBitmap({self.atlas_w}×{self.atlas_h}, "
                f"{self.nbytes / 1024:.1f} KB, {n}/{total} occupied = {pct:.3f}%)")

    @property
    def nbytes(self) -> int:
        """Total bytes used by the bitmap. nvm of voxel count, just the array size."""
        return int(self.bits.nbytes)

    @property
    def shape(self) -> tuple[int, int]:
        """(atlas_h, atlas_w / 8) — the actual numpy shape of the packed array."""
        return self.bits.shape

    # ─── Single-voxel access ──────────────────────────────────────────────

    def set(self, atlas_x: int, atlas_y: int, occupied: bool = True) -> None:
        """Set / clear the bit at (atlas_x, atlas_y)."""
        byte_idx = atlas_x >> 3
        bit_idx = atlas_x & 7
        if occupied:
            self.bits[atlas_y, byte_idx] |= np.uint8(1 << bit_idx)
        else:
            self.bits[atlas_y, byte_idx] &= np.uint8(~(1 << bit_idx) & 0xFF)

    def get(self, atlas_x: int, atlas_y: int) -> bool:
        """Read the bit at (atlas_x, atlas_y)."""
        byte_idx = atlas_x >> 3
        bit_idx = atlas_x & 7
        return bool((self.bits[atlas_y, byte_idx] >> bit_idx) & 1)

    # ─── Voxel-coord access (uses the same bijection as the summary atlas) ──

    def set_voxel(self, u: int, v: int, w: int, occupied: bool = True) -> None:
        """Set occupancy at voxel (u, v, w). Uses voxel_to_atlas internally.

        >>> bmap = OccupancyBitmap()
        >>> bmap.set_voxel(73, 12, 200)
        >>> bmap.get_voxel(73, 12, 200)
        True
        >>> bmap.get_voxel(73, 12, 201)
        False
        """
        ax, ay = voxel_to_atlas(u, v, w)
        self.set(ax, ay, occupied)

    def get_voxel(self, u: int, v: int, w: int) -> bool:
        """Read occupancy at voxel (u, v, w)."""
        ax, ay = voxel_to_atlas(u, v, w)
        return self.get(ax, ay)

    # ─── Bulk operations ──────────────────────────────────────────────────

    def count_occupied(self) -> int:
        """Total number of bits set across the whole bitmap."""
        return int(np.unpackbits(self.bits).sum())

    def occupancy_fraction(self) -> float:
        """Fraction of voxels populated, in [0, 1]."""
        total = self.atlas_w * self.atlas_h
        return self.count_occupied() / total if total > 0 else 0.0

    def fill_from_voxel_iter(self, voxels) -> int:
        """Bulk-set occupancy from any iterable of (u, v, w) tuples.

        Returns the count of voxels written. Vectorised via voxel_to_atlas_vec
        for large lists. ~30× faster than calling set_voxel in a loop.
        """
        voxels = list(voxels)
        if not voxels:
            return 0
        coords = np.asarray(voxels, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected list of (u, v, w) triples; got shape {coords.shape}")
        atlas_xy = voxel_to_atlas_vec(coords)
        ax = atlas_xy[:, 0]
        ay = atlas_xy[:, 1]
        byte_idx = ax >> 3
        bit_mask = (1 << (ax & 7)).astype(np.uint8)
        # Use np.bitwise_or.at for unbuffered scatter (handles duplicates correctly)
        np.bitwise_or.at(self.bits, (ay, byte_idx), bit_mask)
        return len(voxels)

    def fill_from_dense_array(self, dense_occupied: np.ndarray) -> None:
        """Initialise from a dense (VOXEL_RES, VOXEL_RES, VOXEL_RES) bool array.

        For each voxel where dense_occupied[u, v, w] is True, set the bit.
        """
        if dense_occupied.shape != (VOXEL_RES, VOXEL_RES, VOXEL_RES):
            raise ValueError(
                f"Expected ({VOXEL_RES}, {VOXEL_RES}, {VOXEL_RES}); got {dense_occupied.shape}"
            )
        coords = np.argwhere(dense_occupied)
        if coords.size > 0:
            self.fill_from_voxel_iter([tuple(c) for c in coords])

    # ─── Serialise / deserialise ──────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Raw byte buffer in row-major order, atlas_h × (atlas_w / 8) bytes."""
        return self.bits.tobytes()

    @classmethod
    def from_bytes(
        cls, data: bytes, atlas_w: int = ATLAS_SIZE, atlas_h: int = ATLAS_SIZE
    ) -> "OccupancyBitmap":
        """Reconstruct from to_bytes() output. The atlas dims must match."""
        expected_bytes = atlas_h * (atlas_w // 8)
        if len(data) != expected_bytes:
            raise ValueError(
                f"Byte length {len(data)} does not match expected {expected_bytes} "
                f"for {atlas_w}×{atlas_h} atlas"
            )
        out = cls(atlas_w, atlas_h)
        out.bits = np.frombuffer(data, dtype=np.uint8).reshape(
            (atlas_h, atlas_w // 8)
        ).copy()
        return out

    @classmethod
    def from_voxel_store(
        cls, histogram: dict[tuple[int, int, int], dict[int, int]]
    ) -> "OccupancyBitmap":
        """Build a bitmap from voxgaussian's sparse histogram dict.

        Any voxel that appears in the dict (regardless of vote count) is
        considered populated. Use this to pair the bitmap with the existing
        summary atlas built by build_summary_atlas().
        """
        out = cls()
        out.fill_from_voxel_iter(histogram.keys())
        return out


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
        (int(rng.integers(256)), int(rng.integers(256)), int(rng.integers(256))):
            {int(c): int(n) for c, n in zip(
                rng.choice(11, size=3, replace=False),
                rng.integers(1, 50, size=3),
            )}
        for _ in range(8)
    }
    atlas = build_summary_atlas(hist)
    print(f"  summary atlas: shape={atlas.shape} dtype={atlas.dtype}")
    print(f"  non-zero pixels: {(atlas[..., 1] > 0).sum()}")

    print("\nOccupancy bitmap from the same histogram:")
    bmap = OccupancyBitmap.from_voxel_store(hist)
    print(f"  {bmap!r}")
    print(f"  expected occupied count: {len(hist)}")
    actual = bmap.count_occupied()
    print(f"  bitmap occupied count:   {actual}")
    print(f"  round-trip check: ", end="")
    # Verify every voxel from the histogram reads back as occupied
    ok = all(bmap.get_voxel(*idx) for idx in hist)
    # And a few random non-populated voxels read back as empty
    not_in_hist = [
        (i, i, i) for i in [0, 100, 200] if (i, i, i) not in hist
    ]
    ok = ok and not any(bmap.get_voxel(*idx) for idx in not_in_hist)
    print("OK" if ok else "FAILED")
    print(f"  total memory: {bmap.nbytes / 1024 / 1024:.2f} MB for a 4096² atlas")
    print(f"  vs RGBA8 summary atlas: {atlas.nbytes / 1024 / 1024:.1f} MB ({atlas.nbytes / bmap.nbytes:.0f}× ratio)")
