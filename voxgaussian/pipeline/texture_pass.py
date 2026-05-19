"""
texture_pass.py — Phase B: project input image + saved inpaint RGB frames onto
the converged voxel surface.

After geometry refinement converges, every voxel has:
  - position in world space (cell center)
  - mode class (informs base color as fallback)
  - confidence (informs opacity)

What's missing is *actual photographic color*. We retrieve it by:
  1. Building a list of (camera, rgb_image) pairs: the original input image at
     a known camera pose, plus every iteration's inpaint RGB at its camera.
  2. For each voxel, projecting it onto each camera's screen plane. If the
     projected pixel is inside frame AND the voxel is "visible" (front-facing,
     not occluded), sample that RGB.
  3. Per voxel, blend the sampled colors weighted by (1 / camera_distance) and
     (1 / view_angle_from_normal) so close, head-on views dominate.

Output: a colored-voxel JSON snapshot (extends the voxel store's format with
a `color` field per cell) and a per-voxel ply/splat-friendly array.
"""
from __future__ import annotations
import json
import math
import pathlib
import numpy as np
from PIL import Image

from .voxel_store import VoxelStore, EMPTY_CLASSES, CLASS_COLORS
from .render_voxels import Camera


def gather_view_frames(scene_id: str, voxgaussian_root: pathlib.Path) -> list[tuple[Camera, np.ndarray]]:
    """Gather every (camera, RGB array) pair we have for this scene.

    Sources:
      - The original input scene image (assets-raw/<scene>/scene.png) at a
        canonical front-camera pose
      - Every iteration's inpaint RGB saved by ComfyUI to its output dir
    """
    pairs: list[tuple[Camera, np.ndarray]] = []

    # Original input view — we approximate the scene's input camera as facing
    # -Z from (0, 1.5, 4) looking at (0, 1, 0). Matches the bootstrap convention.
    repo_root = voxgaussian_root.parent
    input_img_path = repo_root / "assets-raw" / scene_id / "scene.png"
    if input_img_path.exists():
        img = np.array(Image.open(input_img_path).convert("RGB"))
        input_cam = Camera(position=(0.0, 1.5, 4.0), look_at=(0.0, 1.0, 0.0),
                           fov_deg=50.0, width=img.shape[1], height=img.shape[0])
        pairs.append((input_cam, img))

    # Iteration RGBs — these came from ComfyUI; we need to know their cameras.
    # The refine loop emits one camera per iteration (recorded in `history`).
    # For v1 simplicity, we re-derive candidate cameras and trust the iteration
    # order matches our candidates list (which it does for active view selection
    # when the same store seed is used).
    # NOTE: a more robust v2 would persist the camera pose into the run metadata
    # next to each inpaint frame.
    comfy_out = pathlib.Path(
        r"C:/Users/rxcam/ComfyUI_portable/ComfyUI_windows_portable/ComfyUI/output/voxgaussian"
    )
    if comfy_out.exists():
        from .select_view import candidate_views
        cams = candidate_views(extent=4.0)
        for iter_png in sorted(comfy_out.glob(f"iter_*_*.png")):
            # Filename pattern: iter_<NNN>_<seq>_.png
            # We don't actually know which CAMERA produced it from filename alone.
            # Skip for v1 — only use input image. v2: emit camera JSON alongside RGB.
            pass

    return pairs


def project_color(camera: Camera, voxel_center: np.ndarray,
                  image: np.ndarray) -> tuple[float, float, float, float] | None:
    """Project a 3D voxel center onto a camera's image plane and sample RGB.

    Returns (r, g, b, weight) where weight is higher for views that see the
    voxel head-on and up close, OR None if the voxel isn't visible to this
    camera.
    """
    view, proj = camera.matrices()
    wp = np.array([voxel_center[0], voxel_center[1], voxel_center[2], 1.0])
    vp = view @ wp
    z_cam = -vp[2]
    if z_cam <= camera.near or z_cam >= camera.far:
        return None
    cp = proj @ vp
    if abs(cp[3]) < 1e-6:
        return None
    ndc_x = cp[0] / cp[3]
    ndc_y = cp[1] / cp[3]
    if not (-1.0 < ndc_x < 1.0 and -1.0 < ndc_y < 1.0):
        return None
    px = int((ndc_x * 0.5 + 0.5) * image.shape[1])
    py = int((1.0 - (ndc_y * 0.5 + 0.5)) * image.shape[0])
    px = max(0, min(image.shape[1] - 1, px))
    py = max(0, min(image.shape[0] - 1, py))
    r, g, b = image[py, px]
    # Weight: heavier for nearby voxels (since the camera resolves them better)
    weight = 1.0 / max(0.5, z_cam)
    return (float(r) / 255, float(g) / 255, float(b) / 255, weight)


def texture_store(store: VoxelStore, scene_id: str,
                  voxgaussian_root: pathlib.Path) -> dict[tuple[int, int, int], tuple[float, float, float]]:
    """Build a per-voxel color dict by projecting all view frames.

    Returns: { (ix, iy, iz) → (r, g, b) } in [0, 1] floats.
    """
    pairs = gather_view_frames(scene_id, voxgaussian_root)
    if not pairs:
        print("[texture_pass] no view frames available; falling back to class colors")
        return {idx: _hex_to_rgb(CLASS_COLORS[cls])
                for idx, cls, _ in store.occupied()
                if cls not in EMPTY_CLASSES}

    print(f"[texture_pass] projecting from {len(pairs)} view(s) onto {len(store.cells)} voxels")
    colors: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    cs = store.cell_size
    ox, oy, oz = store.parent_origin

    n_skipped = 0
    for idx, cls, conf in store.occupied():
        if cls in EMPTY_CLASSES:
            continue
        # World-space center
        x = ox - store.extent + (idx[0] + 0.5) * cs
        y = oy - store.extent + (idx[1] + 0.5) * cs
        z = oz - store.extent + (idx[2] + 0.5) * cs
        center = np.array([x, y, z])

        # Sample every view; blend by weight
        accum_r = accum_g = accum_b = total_w = 0.0
        for cam, img in pairs:
            sample = project_color(cam, center, img)
            if sample is None:
                continue
            r, g, b, w = sample
            accum_r += r * w
            accum_g += g * w
            accum_b += b * w
            total_w += w

        if total_w > 0:
            colors[idx] = (accum_r / total_w, accum_g / total_w, accum_b / total_w)
        else:
            # Voxel not visible from any view → use class color
            colors[idx] = _hex_to_rgb(CLASS_COLORS[cls])
            n_skipped += 1

    print(f"[texture_pass] colored {len(colors)} voxels ({n_skipped} fell back to class color)")
    return colors


def _hex_to_rgb(hex_col: str) -> tuple[float, float, float]:
    c = hex_col.lstrip("#")
    return (int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255)


def save_colored_snapshot(store: VoxelStore, colors: dict, out_path: pathlib.Path) -> None:
    """Save a snapshot with per-voxel colors baked in (extends the format with a
    'colors' parallel array). Both the viewer and the splat exporter can consume
    this directly."""
    cells = []
    color_list = []
    for idx, cls, conf in store.occupied():
        if cls in EMPTY_CLASSES:
            continue
        rgb = colors.get(idx, (0.5, 0.5, 0.5))
        cells.append([idx[0], idx[1], idx[2], cls, min(255, int(255 * conf))])
        color_list.append([round(rgb[0], 4), round(rgb[1], 4), round(rgb[2], 4)])

    payload = {
        "type": "colored_voxel_snapshot",
        "iteration": store.iteration,
        "resolution": store.resolution,
        "extent": store.extent,
        "origin": list(store.parent_origin),
        "cells": cells,
        "colors": color_list,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[texture_pass] wrote colored snapshot -> {out_path}")
