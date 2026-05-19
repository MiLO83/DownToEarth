"""
render_voxels.py — Rasterize occupied voxels to a camera-plane image.

Given a VoxelStore and a camera (position, look-at, fov), produce:
  - depth_map:    H×W float32, distance from camera to nearest occupied voxel
                  (or NaN where no voxel is hit)
  - semantic_map: H×W int8, class_id of the front-most voxel per pixel
  - confidence_map: H×W float32, mode confidence of the front-most voxel
  - canonical:    H×W×3 uint8, RGB = (u, v, w) voxel coord of the front-most
                  voxel per pixel; (0, 0, 0) where unknown. This is the
                  Lyra-2-style "geometry-locked" conditioning signal — fed to
                  a ControlNet alongside depth so the diffusion model treats
                  non-black pixels as anchored existing geometry and only
                  fills the genuine occlusion holes (black pixels).
  - unknown_mask: H×W bool, True where NO voxel was hit (regions inpaint must fill)

Implementation: project each occupied voxel center to screen space, depth-buffer
by z-distance, splat to a small kernel (1px for now). Pure numpy; runs in
~100-300 ms for 512×512 with ~100k voxels on a modern CPU. We're not chasing
photoreal here — the output feeds back into a ControlNet inpaint, so a chunky
rasterization is fine and may even regularize the inpaint.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .voxel_store import VoxelStore, EMPTY_CLASSES


@dataclass
class Camera:
    """Simple pinhole camera. All angles in radians, distances in meters."""
    position: tuple[float, float, float]
    look_at:  tuple[float, float, float] = (0.0, 1.0, 0.0)
    up:       tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg:  float = 50.0
    width:    int = 512
    height:   int = 512
    near:     float = 0.05
    far:      float = 50.0

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (view, projection) 4x4 matrices."""
        eye = np.array(self.position, dtype=np.float64)
        target = np.array(self.look_at, dtype=np.float64)
        up = np.array(self.up, dtype=np.float64)
        f = target - eye
        f /= np.linalg.norm(f) + 1e-9
        s = np.cross(f, up)
        s /= np.linalg.norm(s) + 1e-9
        u = np.cross(s, f)
        view = np.eye(4)
        view[0, :3] = s
        view[1, :3] = u
        view[2, :3] = -f
        view[:3, 3] = -view[:3, :3] @ eye

        aspect = self.width / self.height
        f_p = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
        proj = np.zeros((4, 4))
        proj[0, 0] = f_p / aspect
        proj[1, 1] = f_p
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1
        return view, proj


def render(store: VoxelStore, camera: Camera, splat_radius_px: int = 1) -> dict:
    """Rasterize occupied voxels into per-pixel maps from `camera`'s view."""
    W, H = camera.width, camera.height
    view, proj = camera.matrices()

    # Initialize buffers
    depth = np.full((H, W), np.nan, dtype=np.float32)
    semantic = np.zeros((H, W), dtype=np.int8)
    confidence = np.zeros((H, W), dtype=np.float32)
    # Canonical-coord image: each pixel's RGB = (u, v, w) of the front-most
    # voxel that hit it. Pixels with no voxel stay (0, 0, 0). For res ≤ 256
    # this is byte-perfect identity — the bytes literally ARE the coord.
    canonical = np.zeros((H, W, 3), dtype=np.uint8)

    # Collect occupied voxels as a batch
    indices = []
    classes = []
    confs = []
    for idx, cls, conf in store.occupied():
        if cls in EMPTY_CLASSES:
            continue
        indices.append(idx)
        classes.append(cls)
        confs.append(conf)
    if not indices:
        return _empty_result(camera)

    # Voxel centers in world space
    cs = store.cell_size
    ox, oy, oz = store.parent_origin
    idx_arr = np.array(indices, dtype=np.float32)
    centers = np.stack([
        ox - store.extent + (idx_arr[:, 0] + 0.5) * cs,
        oy - store.extent + (idx_arr[:, 1] + 0.5) * cs,
        oz - store.extent + (idx_arr[:, 2] + 0.5) * cs,
    ], axis=1)  # (N, 3)

    # Project to camera space: world → view → clip → screen
    homo = np.concatenate([centers, np.ones((len(centers), 1))], axis=1)  # (N, 4)
    view_pts = (view @ homo.T).T   # (N, 4)
    z_cam = -view_pts[:, 2]        # distance into the scene; +z is forward in our convention
    in_front = z_cam > camera.near
    if not np.any(in_front):
        return _empty_result(camera)

    clip = (proj @ view_pts.T).T   # (N, 4)
    w = clip[:, 3]
    valid = (np.abs(w) > 1e-6) & in_front
    ndc = clip[valid] / w[valid, None]   # (N', 4)
    sx = (ndc[:, 0] * 0.5 + 0.5) * W
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H
    px = np.round(sx).astype(np.int32)
    py = np.round(sy).astype(np.int32)
    z_valid = z_cam[valid]

    classes_arr = np.array(classes, dtype=np.int8)[valid]
    confs_arr = np.array(confs, dtype=np.float32)[valid]
    # Per-voxel (u, v, w) for the canonical-coord channel.
    uvw_arr = idx_arr[valid].astype(np.uint8)   # (N', 3), values in [0, res-1]

    # Splat each voxel as a small disk, keeping nearest-z per pixel
    R = splat_radius_px
    # Sort by distance (far→near) so near voxels naturally overwrite. Cheap depth-buffer.
    order = np.argsort(-z_valid)
    px = px[order]; py = py[order]
    z_valid = z_valid[order]
    classes_arr = classes_arr[order]
    confs_arr = confs_arr[order]
    uvw_arr = uvw_arr[order]

    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dx * dx + dy * dy > R * R: continue
            qx = px + dx
            qy = py + dy
            ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
            qx2 = qx[ok]; qy2 = qy[ok]
            zz = z_valid[ok]
            cc = classes_arr[ok]
            ff = confs_arr[ok]
            uu = uvw_arr[ok]
            # nearest-z wins
            cur = depth[qy2, qx2]
            replace = np.isnan(cur) | (zz < cur)
            depth[qy2[replace], qx2[replace]] = zz[replace]
            semantic[qy2[replace], qx2[replace]] = cc[replace]
            confidence[qy2[replace], qx2[replace]] = ff[replace]
            canonical[qy2[replace], qx2[replace]] = uu[replace]

    unknown_mask = np.isnan(depth)
    return {
        "depth": depth,            # (H,W) float32, NaN = unknown
        "semantic": semantic,      # (H,W) int8 class id
        "confidence": confidence,  # (H,W) float32 in [0,1]
        "canonical": canonical,    # (H,W,3) uint8, RGB=(u,v,w); (0,0,0)=unknown
        "unknown_mask": unknown_mask,  # (H,W) bool — True where inpaint should fill
        "camera": camera,
    }


def _empty_result(camera: Camera) -> dict:
    W, H = camera.width, camera.height
    return {
        "depth": np.full((H, W), np.nan, dtype=np.float32),
        "semantic": np.zeros((H, W), dtype=np.int8),
        "confidence": np.zeros((H, W), dtype=np.float32),
        "canonical": np.zeros((H, W, 3), dtype=np.uint8),
        "unknown_mask": np.ones((H, W), dtype=bool),
        "camera": camera,
    }


def write_debug_pngs(result: dict, out_prefix: str) -> None:
    """Dump depth + semantic + unknown maps to PNGs for visual debugging."""
    from PIL import Image
    from .voxel_store import CLASS_COLORS

    depth = result["depth"]
    sem = result["semantic"]
    unk = result["unknown_mask"]

    # Depth: normalize to 0-255, NaN → black
    d = depth.copy()
    finite = ~np.isnan(d)
    if finite.any():
        lo, hi = d[finite].min(), d[finite].max()
        d[finite] = 255.0 * (1.0 - (d[finite] - lo) / max(hi - lo, 1e-6))
    d[~finite] = 0
    Image.fromarray(d.astype(np.uint8)).save(f"{out_prefix}_depth.png")

    # Semantic: map class IDs to RGB
    rgb = np.zeros((sem.shape[0], sem.shape[1], 3), dtype=np.uint8)
    for cid, hexcol in CLASS_COLORS.items():
        c = hexcol.lstrip("#")
        rgb[sem == cid] = [int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)]
    rgb[unk] = (40, 40, 40)
    Image.fromarray(rgb).save(f"{out_prefix}_semantic.png")

    Image.fromarray((unk * 255).astype(np.uint8)).save(f"{out_prefix}_unknown.png")

    # Canonical: each pixel's RGB literally is its voxel coord. Saved raw —
    # this is what the ControlNet anchor reads.
    if "canonical" in result:
        Image.fromarray(result["canonical"], mode="RGB").save(f"{out_prefix}_canonical.png")
