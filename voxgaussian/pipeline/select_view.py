"""
select_view.py — Active-view selection: 50/50 found ↔ unfound balance.

Pure uncertainty-maximization picks views where most of the frame is holes,
which gives the inpaint model no context to anchor against. Pure
known-maximization is pointless (no new info to gain). Sweet spot is a
view that has BOTH plenty of found voxels (strong inpaint context) AND
plenty of unfound pixels (real disocclusion holes to fill).

Scoring: cheap 32×32 render per candidate, then score = sqrt(found * unfound).
The geometric mean peaks when found ≈ unfound, naturally favouring views
that show "half scene, half hole" rather than "all scene" or "all sky."

Candidates: a small grid of poses surrounding the scene at varying
elevations + 2 axis-aligned map shots. ~30 candidates is plenty; rendering
each at 32×32 is ~30 ms, total view-selection cost <2 s per iteration even
at 100k+ voxels — negligible next to the inpaint pass.

After a view is selected, we record it in the `used_views` set so subsequent
picks favor unseen angles even if their balance score is similar.
"""
from __future__ import annotations
import math
import numpy as np
from .voxel_store import VoxelStore
from .render_voxels import Camera, render


def candidate_views(extent: float, num_orbital: int = 16, elevations_deg: list[float] | None = None) -> list[Camera]:
    """Generate a default set of candidate camera poses around a scene cube
    of half-width `extent`. Cameras orbit at radius ~ 1.8*extent and look at
    the scene center (height = floor + 1m)."""
    elevations_deg = elevations_deg or [-15.0, 5.0, 25.0, 45.0]
    radius = 1.8 * extent
    look_at = (0.0, 1.0, 0.0)
    cams: list[Camera] = []
    for el_deg in elevations_deg:
        el = math.radians(el_deg)
        for k in range(num_orbital):
            theta = 2 * math.pi * k / num_orbital
            x = radius * math.cos(el) * math.sin(theta)
            z = radius * math.cos(el) * math.cos(theta)
            y = radius * math.sin(el) + 1.5   # offset above floor
            cams.append(Camera(position=(x, y, z), look_at=look_at, fov_deg=50.0))
    # plus 2 high-angle "map view" shots
    cams.append(Camera(position=(0.0, 2.0 * extent, 0.01), look_at=(0, 0, 0), fov_deg=70.0))
    cams.append(Camera(position=(0.0, -0.2 * extent, 0.01), look_at=(0, 1.5, 0), fov_deg=70.0))
    return cams


def score_view(
    camera: Camera,
    store: VoxelStore,
    probe_res: int = 32,
) -> tuple[float, int, int]:
    """50/50 found-vs-unfound score for a candidate camera.

    Renders the candidate at a low probe resolution and counts:
      found    = pixels where a voxel was hit (existing geometry context)
      unfound  = pixels where the ray missed (disocclusion holes / inpaint canvas)

    Returns (score, found, unfound) where score = sqrt(found * unfound).
    The geometric mean peaks when found ≈ unfound, which is the 50/50 sweet
    spot: enough context for the inpaint to anchor, enough holes to learn from.

    Pure max-found or pure max-unfound views score 0 — same as a candidate
    that sees nothing at all. The pipeline naturally avoids "all sky" and
    "fully populated frame" candidates.
    """
    # Cheap probe render — same projection as the main render, just at
    # probe_res×probe_res so it costs ~30 ms even at 100k+ voxels.
    probe_cam = Camera(
        position=camera.position,
        look_at=camera.look_at,
        up=camera.up,
        fov_deg=camera.fov_deg,
        width=probe_res,
        height=probe_res,
        near=camera.near,
        far=camera.far,
    )
    result = render(store, probe_cam, splat_radius_px=1)
    unknown_mask = result["unknown_mask"]
    n_pixels = probe_res * probe_res
    n_unfound = int(unknown_mask.sum())
    n_found = n_pixels - n_unfound
    score = math.sqrt(max(0, n_found) * max(0, n_unfound))
    return score, n_found, n_unfound


def pick_next_view(store: VoxelStore, candidates: list[Camera],
                   used_views: set[int], penalty_for_used: float = 0.3) -> tuple[int, Camera]:
    """Score every candidate by the 50/50 found-vs-unfound balance and pick
    the highest. Already-used views get penalised so we keep exploring.
    """
    best_idx = 0
    best_score = -1.0
    best_found = 0
    best_unfound = 0
    for i, cam in enumerate(candidates):
        s, f, u = score_view(cam, store)
        if i in used_views:
            s *= penalty_for_used
        if s > best_score:
            best_score = s
            best_idx = i
            best_found = f
            best_unfound = u
    total = max(1, best_found + best_unfound)
    pct_found = 100.0 * best_found / total
    print(f"  view-select: cam #{best_idx}  "
          f"score={best_score:.0f}  found={best_found}  unfound={best_unfound}  "
          f"({pct_found:.0f}% found / {100.0 - pct_found:.0f}% unfound)")
    return best_idx, candidates[best_idx]
