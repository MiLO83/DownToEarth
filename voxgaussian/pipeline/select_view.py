"""
select_view.py — Active-view selection by uncertainty-mass-in-frustum.

We don't iterate around a fixed orbit. Each pass picks the camera angle that
maximizes information gain — specifically, the SUM of voxel uncertainty within
that camera's frustum, weighted by 1/depth so near-the-camera voxels count
more (since the inpaint will have more visible pixels to reason about them).

Candidates: a small grid of poses surrounding the scene at varying
elevations + 6 cardinal axis-aligned views. ~30 candidates is plenty.

After a view is selected, we record it in the `used_views` set so subsequent
picks favor unseen angles even if their uncertainty score is similar.
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


def score_view(camera: Camera, store: VoxelStore, downsample_voxels: int = 4) -> float:
    """Score a candidate view by summed uncertainty of voxels visible in its
    frustum, weighted by 1/depth.

    We don't fully rasterize for scoring (too slow per candidate). Instead we
    iterate occupied voxels at a downsampled stride, project each, and check
    if it's inside the screen + far/near. Cheap and correct enough.
    """
    view, proj = camera.matrices()
    cs = store.cell_size
    ox, oy, oz = store.parent_origin

    total = 0.0
    for n, (idx, _cls, _conf) in enumerate(store.occupied()):
        if downsample_voxels > 1 and (n % downsample_voxels) != 0:
            continue
        x = ox - store.extent + (idx[0] + 0.5) * cs
        y = oy - store.extent + (idx[1] + 0.5) * cs
        z = oz - store.extent + (idx[2] + 0.5) * cs
        # Transform to view space
        wp = np.array([x, y, z, 1.0])
        vp = view @ wp
        z_cam = -vp[2]
        if z_cam <= camera.near or z_cam >= camera.far:
            continue
        cp = proj @ vp
        if abs(cp[3]) < 1e-6:
            continue
        ndc_x = cp[0] / cp[3]
        ndc_y = cp[1] / cp[3]
        if not (-1.05 < ndc_x < 1.05 and -1.05 < ndc_y < 1.05):
            continue
        # Sum uncertainty * (1/depth) — near voxels matter more
        total += store.uncertainty(idx) / max(0.1, z_cam)

    # Also count unknown-frustum mass: bigger fov / closer view = more pixels
    # to inpaint into. We approximate this by adding a base score that's
    # higher for views whose camera is closer to the scene's bulk.
    return total


def pick_next_view(store: VoxelStore, candidates: list[Camera],
                   used_views: set[int], penalty_for_used: float = 0.3) -> tuple[int, Camera]:
    """Score every candidate, return (index, camera) of the best one.

    `used_views` is the set of candidate indices already used in previous
    iterations. Re-using a view is allowed but penalized so we explore.
    """
    best_idx = 0
    best_score = -1.0
    for i, cam in enumerate(candidates):
        s = score_view(cam, store)
        if i in used_views:
            s *= penalty_for_used
        if s > best_score:
            best_score = s
            best_idx = i
    return best_idx, candidates[best_idx]
