"""
propagate.py — Project inpainted (depth, semantic) view back into voxel votes.

Given a camera + an H×W depth map (meters per pixel) + H×W semantic map +
the voxel store, walk each pixel:
  1. Build the world-space point at that pixel's depth via inverse projection
  2. Identify the voxel containing that point
  3. Add a vote of `semantic[v, u]` to that voxel's histogram

Optionally, we can also "ray-carve" empty space along the camera ray up to
the depth: mark cells in front of the surface as `empty` votes. This kills
floating Gaussians/voxels left over from earlier passes.
"""
from __future__ import annotations
import math
import numpy as np
from .render_voxels import Camera
from .voxel_store import VoxelStore, EMPTY_CLASSES


def propagate(
    store: VoxelStore,
    camera: Camera,
    depth: np.ndarray,    # (H, W) meters; NaN = no information
    semantic: np.ndarray, # (H, W) class IDs (int8)
    rgb: np.ndarray | None = None,  # (H, W, 3) float 0..1; per-pixel inpaint color
    unknown_mask: np.ndarray | None = None,  # (H, W) bool; True where the pre-inpaint render saw nothing
    weight_at_camera: int = 3,
    ray_carve: bool = True,
    carve_step_m: float = 0.15,
    sample_stride: int = 2,
    carve_weight: float = 0.3,           # empty vote weight — much less than occupancy
    carve_respect_threshold: float = 0.65, # don't carve through cells already confidently non-empty
    gate_on_unknown_mask: bool = True,    # Lyra-2-style: only vote where the original render had a hole
) -> dict:
    """Project pixels into voxel votes.

    When `gate_on_unknown_mask=True` and `unknown_mask` is provided, votes are
    written ONLY for pixels that were originally unknown (i.e. the inpaint
    actually invented something there). Pixels the renderer already saw cleanly
    are not re-voted — the existing voxel histogram for those cells is sacred.
    This matches Lyra 2's "fill holes, don't re-imagine known geometry" loop
    and is what stops iterations from diluting the bootstrap into mush.

    Returns counters for debug: cells_voted, cells_carved.
    """
    H, W = depth.shape
    view, proj = camera.matrices()
    inv_view = np.linalg.inv(view)
    inv_proj = np.linalg.inv(proj)

    cam_pos = np.array(camera.position, dtype=np.float64)

    cells_voted = 0
    cells_carved = 0
    cells_skipped_already_known = 0
    apply_gate = gate_on_unknown_mask and unknown_mask is not None

    # Precompute the per-pixel ray direction (in world space) via inverse
    # projection of NDC corners → camera space → world space.
    # We use a vectorized approach for speed.
    us = (np.arange(W, dtype=np.float64) + 0.5) / W * 2.0 - 1.0
    vs = (np.arange(H, dtype=np.float64) + 0.5) / H * 2.0 - 1.0
    # Subsample
    us = us[::sample_stride]
    vs = vs[::sample_stride]
    UU, VV = np.meshgrid(us, vs)
    # Build NDC points at z=0 (near) and z=1 (far) to extract ray direction
    ones = np.ones_like(UU)
    ndc_near = np.stack([UU, VV, -1 * ones, ones], axis=-1)  # (h, w, 4)
    ndc_far  = np.stack([UU, VV,  1 * ones, ones], axis=-1)
    cam_near = ndc_near @ inv_proj.T
    cam_near = cam_near / cam_near[..., 3:4]
    cam_far  = ndc_far  @ inv_proj.T
    cam_far  = cam_far / cam_far[..., 3:4]
    world_near = cam_near @ inv_view.T
    world_far  = cam_far  @ inv_view.T
    dirs = world_far[..., :3] - world_near[..., :3]
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-9)

    # Iterate the downsampled grid
    sample_ys = np.arange(0, H, sample_stride)
    sample_xs = np.arange(0, W, sample_stride)
    for sy, py in enumerate(sample_ys):
        for sx, px in enumerate(sample_xs):
            # Lyra-2-style hole-only voting: skip pixels the renderer already saw.
            # The inpaint output at known pixels is at best a noisy reproduction
            # of existing voxels (depth ControlNet anchored it back to whatever
            # we showed it); re-voting those pixels is what dilutes the histogram.
            if apply_gate and not bool(unknown_mask[py, px]):
                cells_skipped_already_known += 1
                continue
            d = float(depth[py, px])
            cls = int(semantic[py, px])
            if cls == 1:    # sky — vote the ray's "far" voxel as sky but don't terminate
                continue
            if math.isnan(d) or d <= 0.0:
                continue

            ray_d = dirs[sy, sx]
            hit = cam_pos + ray_d * d
            idx = store.world_to_voxel(tuple(hit.tolist()))
            if store.in_bounds(idx):
                store.vote(idx, cls, weight=weight_at_camera)
                cells_voted += 1
                # Sub-cell surface offset (world-space displacement from
                # voxel center to the actual ray-hit point).
                center = store.voxel_to_world(idx)
                offset = (hit[0] - center[0], hit[1] - center[1], hit[2] - center[2])
                # Surface normal: direction from the hit point back toward
                # the camera. Light hit the surface from this direction →
                # surface normal points roughly toward the camera.
                normal_vec = cam_pos - hit
                nlen = np.linalg.norm(normal_vec)
                if nlen > 1e-6:
                    normal = (float(normal_vec[0] / nlen),
                              float(normal_vec[1] / nlen),
                              float(normal_vec[2] / nlen))
                else:
                    normal = None
                # Per-voxel color sample (if inpaint provided RGB) + offset + normal.
                if rgb is not None:
                    rcol = rgb[py, px]
                    store.vote_appearance(idx,
                                          (float(rcol[0]), float(rcol[1]), float(rcol[2])),
                                          offset, normal)
                else:
                    store.vote_appearance(idx, None, offset, normal)

            if ray_carve:
                steps = int(max(0, (d - store.cell_size) / carve_step_m))
                for s in range(1, steps):
                    t = s * carve_step_m
                    p = cam_pos + ray_d * t
                    cidx = store.world_to_voxel(tuple(p.tolist()))
                    if not store.in_bounds(cidx) or cidx == idx:
                        continue
                    existing_cls, existing_conf = store.mode(cidx)
                    if existing_conf >= carve_respect_threshold and existing_cls not in EMPTY_CLASSES:
                        # Already confidently classified as a structure surface —
                        # this inpaint disagrees, but trust history over a single
                        # discordant view. Stop carving past this cell too.
                        break
                    store.vote(cidx, 0, weight=carve_weight)
                    cells_carved += 1

    return {
        "cells_voted": cells_voted,
        "cells_carved": cells_carved,
        "cells_skipped_already_known": cells_skipped_already_known,
    }
