"""
gaussian_fit.py - Multi-sub-gaussian per voxel with SD-sampled colors + per-
gauss normal for the dollhouse cutaway shader cull.

Strategy:
  - Each occupied voxel emits N sub-gaussians at jittered positions on the
    voxel's surface (defined by per-voxel mean offset + normal).
  - Each sub-gauss samples its color from the multi-view SD frames we
    persisted during refine (scene.png + each iteration's inpaint RGB).
  - Color blend weighting prefers views whose direction aligns with the
    voxel's surface normal AND that are close to the surface.
  - Output carries the per-voxel normal so the viewer can do shader-side
    cutaway culling (interior-only rendering).

Framerate-budgeted: target ~1M gaussians total. If the voxel count would
overflow, drop low-confidence voxels first; if it's underfull, increase
sub-gauss count per voxel.
"""
from __future__ import annotations
import json
import math
import pathlib
import numpy as np

from .voxel_store import VoxelStore, EMPTY_CLASSES, CLASS_COLORS
from .render_voxels import Camera


DEFAULT_BUDGET = 1_000_000
MAX_SUBGAUSS_PER_VOXEL = 8
MIN_SUBGAUSS_PER_VOXEL = 1


def _hex_to_rgb(hex_col: str) -> tuple[float, float, float]:
    c = hex_col.lstrip("#")
    return (int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255)


def _project_sample(camera: Camera, world_pt: np.ndarray, image: np.ndarray
                    ) -> tuple[float, float, float, float] | None:
    """Project a world point into the camera and sample RGB. Returns
    (r, g, b, distance_to_camera) or None if not visible."""
    view, proj = camera.matrices()
    wp = np.array([world_pt[0], world_pt[1], world_pt[2], 1.0])
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
    H, W = image.shape[:2]
    px = int((ndc_x * 0.5 + 0.5) * W)
    py = int((1.0 - (ndc_y * 0.5 + 0.5)) * H)
    px = max(0, min(W - 1, px))
    py = max(0, min(H - 1, py))
    pix = image[py, px]
    return (float(pix[0]) / 255, float(pix[1]) / 255, float(pix[2]) / 255, z_cam)


def _multi_view_color(world_pt: np.ndarray, normal: np.ndarray,
                      views: list[tuple[Camera, np.ndarray]],
                      fallback_rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Blend pixel samples from every view that sees this point. Weight =
    view alignment (dot of view-direction with surface normal) * 1/distance.
    """
    if not views:
        return fallback_rgb
    accum_r = accum_g = accum_b = total_w = 0.0
    for cam, img in views:
        s = _project_sample(cam, world_pt, img)
        if s is None:
            continue
        r, g, b, dist = s
        # View direction: from camera position toward the world point.
        # Surface is "facing" this camera if view_dir is roughly anti-parallel
        # to surface normal. Equivalently: alignment = dot(normal, cam_to_pt) < 0
        # means the surface faces the camera.
        cam_pos = np.array(cam.position)
        to_pt = world_pt - cam_pos
        to_pt_n = to_pt / (np.linalg.norm(to_pt) + 1e-9)
        alignment = -float(np.dot(normal, to_pt_n))   # 1 = head-on, 0 = grazing, <0 = behind
        if alignment <= 0.0:
            continue   # this view looks at the back of our surface — skip
        w = alignment / max(0.5, dist)
        accum_r += r * w; accum_g += g * w; accum_b += b * w; total_w += w
    if total_w <= 1e-9:
        return fallback_rgb
    return (accum_r / total_w, accum_g / total_w, accum_b / total_w)


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthogonal tangent vectors on the plane perpendicular to `normal`.
    Used to place sub-gaussians as a jittered patch on the voxel's surface."""
    n = normal / (np.linalg.norm(normal) + 1e-9)
    # Pick an arbitrary vector not parallel to n
    if abs(n[1]) < 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    else:
        helper = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1) + 1e-9
    t2 = np.cross(n, t1)
    return t1, t2


def _subgauss_positions(center: np.ndarray, normal: np.ndarray, count: int,
                        patch_size: float) -> list[np.ndarray]:
    """Generate `count` jittered positions on the surface plane around center."""
    if count <= 1:
        return [center]
    t1, t2 = _tangent_basis(normal)
    # Lay out in a roughly square grid scaled to patch_size
    side = int(math.ceil(math.sqrt(count)))
    step = patch_size / max(1, side - 1) if side > 1 else 0.0
    positions = []
    for i in range(count):
        gx = i % side
        gy = i // side
        u = (gx - (side - 1) / 2.0) * step
        v = (gy - (side - 1) / 2.0) * step
        positions.append(center + t1 * u + t2 * v)
    return positions[:count]


def load_persisted_views(scene_id: str, runs_root: pathlib.Path,
                         repo_root: pathlib.Path) -> list[tuple[Camera, np.ndarray]]:
    """Load every (camera, RGB) pair we saved during refine, plus the
    canonical scene.png front view."""
    from PIL import Image
    pairs: list[tuple[Camera, np.ndarray]] = []

    # Seed canonical view
    seed_png = repo_root / "assets-raw" / scene_id / "scene.png"
    if seed_png.exists():
        img = np.array(Image.open(seed_png).convert("RGB"))
        seed_cam = Camera(position=(0.0, 1.5, 4.0), look_at=(0.0, 1.0, 0.0),
                          fov_deg=50.0, width=img.shape[1], height=img.shape[0])
        pairs.append((seed_cam, img))

    # Iteration views
    views_dir = runs_root / scene_id / "views"
    if views_dir.exists():
        for d in sorted(views_dir.iterdir()):
            if not d.is_dir():
                continue
            rgb_path = d / "rgb.png"
            cam_path = d / "camera.json"
            if not (rgb_path.exists() and cam_path.exists()):
                continue
            img = np.array(Image.open(rgb_path).convert("RGB"))
            with open(cam_path, "r") as f:
                cm = json.load(f)
            cam = Camera(
                position=tuple(cm["position"]),
                look_at=tuple(cm["look_at"]),
                up=tuple(cm["up"]),
                fov_deg=cm["fov_deg"],
                width=cm["width"],
                height=cm["height"],
                near=cm["near"],
                far=cm["far"],
            )
            pairs.append((cam, img))

    print(f"[gaussian_fit] loaded {len(pairs)} view frames for multi-view blend")
    return pairs


def _voxel_normal(store: VoxelStore, idx: tuple[int, int, int]) -> np.ndarray:
    """Return the per-voxel mean normal (unit vector), or a sensible default
    if no samples (away from chunk origin)."""
    n = store.normals.get(idx)
    if n is not None and n[3] > 0:
        nx = n[0] / n[3]; ny = n[1] / n[3]; nz = n[2] / n[3]
        mag = math.sqrt(nx*nx + ny*ny + nz*nz)
        if mag > 1e-6:
            return np.array([nx / mag, ny / mag, nz / mag])
    # Fallback: outward radial from the chunk origin
    center = np.array(store.voxel_to_world(idx))
    out = center - np.array(store.parent_origin)
    mag = float(np.linalg.norm(out))
    if mag > 1e-6:
        return out / mag
    return np.array([0.0, 1.0, 0.0])


def fit_gaussians(store: VoxelStore,
                  views: list[tuple[Camera, np.ndarray]],
                  colors_fallback: dict[tuple[int, int, int], tuple[float, float, float]] | None = None,
                  budget: int = DEFAULT_BUDGET) -> tuple[list[dict], list[float]]:
    """Produce N sub-gaussians per occupied voxel, color-sampled from views.

    Returns (gaussians, centroid_xyz). The centroid is the mean position of
    surviving gaussians — used as the shader's cutaway-cull reference.
    """
    cs = store.cell_size

    # First pass: collect all voxels we'll emit gausses for, with sort-keys
    candidates: list[tuple[float, tuple[int, int, int], int, float]] = []
    for idx, cls, conf in store.occupied():
        if cls in EMPTY_CLASSES:
            continue
        candidates.append((conf, idx, cls, conf))   # sort by confidence desc
    candidates.sort(key=lambda x: -x[0])

    # Pick sub-gauss-per-voxel from budget
    n_voxels = len(candidates)
    if n_voxels == 0:
        return [], [0.0, 0.0, 0.0]
    subgauss = max(MIN_SUBGAUSS_PER_VOXEL,
                   min(MAX_SUBGAUSS_PER_VOXEL, budget // max(1, n_voxels)))
    # Drop low-confidence voxels if we'd still overflow at MIN sub-gauss
    if n_voxels * subgauss > budget:
        n_voxels = budget // subgauss
        candidates = candidates[:n_voxels]
    print(f"[gaussian_fit] {n_voxels} voxels x {subgauss} sub-gauss = "
          f"{n_voxels * subgauss} total gaussians (budget {budget})")

    patch_size = cs * 0.75   # spread sub-gausses across most of the cell face
    sub_scale = cs * 0.6 / math.sqrt(subgauss)   # smaller as density rises

    gaussians: list[dict] = []
    cent_sx = cent_sy = cent_sz = 0.0

    for _, idx, cls, conf in candidates:
        # Voxel surface position: cell center + mean offset
        center = np.array(store.voxel_to_world(idx))
        off = store.offsets.get(idx)
        if off is not None and off[3] > 0:
            center = center + np.array([off[0] / off[3], off[1] / off[3], off[2] / off[3]])

        normal = _voxel_normal(store, idx)

        positions = _subgauss_positions(center, normal, subgauss, patch_size)

        # Fallback color: per-voxel mean (from store.colors) → texture-pass color → class color
        fallback: tuple[float, float, float]
        fc = store.colors.get(idx)
        if fc is not None and fc[3] > 0:
            n = fc[3]
            fallback = (fc[0] / n, fc[1] / n, fc[2] / n)
        elif colors_fallback and idx in colors_fallback:
            fallback = colors_fallback[idx]
        else:
            fallback = _hex_to_rgb(CLASS_COLORS.get(cls, "#ffffff"))

        for pos in positions:
            rgb = _multi_view_color(pos, normal, views, fallback)
            gaussians.append({
                "p": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
                "c": [round(rgb[0], 4), round(rgb[1], 4), round(rgb[2], 4)],
                "a": round(conf, 4),
                "s": round(sub_scale, 4),
                "n": [round(float(normal[0]), 3), round(float(normal[1]), 3), round(float(normal[2]), 3)],
            })
            cent_sx += pos[0]; cent_sy += pos[1]; cent_sz += pos[2]

    cnt = max(1, len(gaussians))
    centroid = [cent_sx / cnt, cent_sy / cnt, cent_sz / cnt]
    return gaussians, centroid


def save_gaussians(gaussians: list[dict], centroid: list[float],
                   out_path: pathlib.Path, meta: dict | None = None) -> None:
    payload = {
        "type": "gaussian_cloud",
        "count": len(gaussians),
        "centroid": centroid,
        "gaussians": gaussians,
    }
    if meta:
        payload["meta"] = meta
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[gaussian_fit] wrote {len(gaussians)} gaussians -> {out_path}")


def fit_and_save(store: VoxelStore, colors: dict, out_path: pathlib.Path,
                 scene_id: str, budget: int = DEFAULT_BUDGET) -> int:
    """Top-level entry. Loads persisted views, fits multi-sub-gaussians,
    writes JSON. Returns total gauss count."""
    voxgaussian_root = pathlib.Path(__file__).resolve().parent.parent
    repo_root = voxgaussian_root.parent
    runs_root = voxgaussian_root / "runs"
    views = load_persisted_views(scene_id, runs_root, repo_root)
    gs, centroid = fit_gaussians(store, views, colors_fallback=colors, budget=budget)
    save_gaussians(gs, centroid, out_path, meta={
        "scene_id": scene_id,
        "iteration": store.iteration,
        "resolution": store.resolution,
        "extent": store.extent,
        "budget": budget,
    })
    return len(gs)
