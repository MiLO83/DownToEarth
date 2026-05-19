"""
bootstrap.py — Initialize the voxel grid from existing assets.

We don't start from "what is geometry?" — we have:
  1. A Hunyuan3D mesh of the scene (rough, single-object-flavored, but it's
     a reasonable initial topology)
  2. The original input image (the AI-generated scene PNG)
  3. Depth-Anything V2 (already installed)
  4. A semantic segmentation model — we'll use ADE20K via OneFormer or
     fall back to a simple color-band-based heuristic if that's not
     available

Pipeline:
  a) Voxelize the Hunyuan3D mesh → mark voxels along the mesh surface
  b) Run depth estimation + semantic segmentation on input image
  c) Project image pixels into world space using estimated depth, and
     vote their semantic class into the corresponding voxels
  d) Result: a partially-filled voxel histogram grid that already knows the
     topology and approximate class labels of everything the input image
     showed. The iterative refinement loop then fills the back/sides.
"""
from __future__ import annotations
import math
import pathlib
from typing import Optional
import numpy as np
from PIL import Image

from .voxel_store import VoxelStore, CLASSES


# Mapping from common ADE20K class names → our internal class IDs.
# Falls back to "prop" for anything we don't have an explicit bucket for.
ADE20K_TO_INTERNAL = {
    "sky": 1,
    "grass": 2, "field": 2, "ground": 2, "earth": 2, "dirt track": 2,
    "path": 3, "road": 3, "sidewalk": 3, "runway": 3, "stairway": 3, "stairs": 3,
    "stone": 3, "rock": 3,
    "water": 4, "sea": 4, "river": 4, "pool": 4, "lake": 4, "fountain": 4,
    "wall": 5, "fence": 5, "railing": 5, "column": 5, "pillar": 5,
    "building": 6, "house": 6, "hovel": 6, "tower": 6, "skyscraper": 6,
    "tree": 7, "plant": 7, "bush": 7, "flower": 7, "palm": 7,
    "person": 9, "human": 9,
}


def voxelize_mesh(mesh_path: pathlib.Path, store: VoxelStore,
                  default_class: int = 6) -> int:
    """Mark voxels intersected by a mesh's triangles.

    Cheap approximation: sample points densely on the mesh surface, vote each
    sample into its containing voxel. Surface-only — interior of solids stays
    empty. That's actually what we want for hollow Hunyuan3D outputs.

    `default_class` is what we vote until proper semantic projection refines
    things. Default to 'building' (6) since Hunyuan3D output for scenes tends
    to look building-shaped.
    """
    import trimesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if mesh is None or len(mesh.vertices) == 0:
        return 0
    # Sample ~10 points per voxel cell along the surface area
    target_samples = max(10000, int(mesh.area / (store.cell_size ** 2)) * 4)
    samples, face_ids = trimesh.sample.sample_surface(mesh, target_samples)
    face_normals = mesh.face_normals
    n_added = 0
    for p, fid in zip(samples, face_ids):
        idx = store.world_to_voxel((float(p[0]), float(p[1]), float(p[2])))
        if store.in_bounds(idx):
            store.vote(idx, default_class)
            n_added += 1
            # Surface offset = sample point - voxel center
            center = store.voxel_to_world(idx)
            offset = (float(p[0]) - center[0], float(p[1]) - center[1], float(p[2]) - center[2])
            # Mesh face normal — already a unit vector from trimesh
            fn = face_normals[fid]
            normal = (float(fn[0]), float(fn[1]), float(fn[2]))
            store.vote_appearance(idx, None, offset, normal)
    return n_added


def semantic_segment(image: Image.Image) -> tuple[np.ndarray, dict[int, str]]:
    """Try to run a real semantic segmenter. Fall back to a cheap heuristic
    if dependencies are missing. Returns (H,W) int array of class IDs in OUR
    internal taxonomy, plus a label dict for debug."""
    try:
        return _segment_oneformer(image)
    except Exception as e:
        print(f"  [bootstrap] OneFormer unavailable ({e}); falling back to heuristic")
        return _segment_heuristic(image)


def _segment_oneformer(image: Image.Image) -> tuple[np.ndarray, dict[int, str]]:
    """Use OneFormer (ADE20K trained) for proper semantic segmentation."""
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    import torch
    proc = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
    if torch.cuda.is_available():
        model = model.cuda()
    inputs = proc(images=image, task_inputs=["semantic"], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    seg = proc.post_process_semantic_segmentation(out, target_sizes=[image.size[::-1]])[0]
    seg = seg.cpu().numpy()

    id_to_label = model.config.id2label
    # Remap to our internal class IDs
    internal = np.zeros_like(seg, dtype=np.int8)
    for ade_id, label in id_to_label.items():
        label_lc = label.lower()
        for keyword, internal_id in ADE20K_TO_INTERNAL.items():
            if keyword in label_lc:
                internal[seg == ade_id] = internal_id
                break
        else:
            # No keyword matched → treat as 'prop'
            internal[seg == ade_id] = 8

    return internal, dict(id_to_label)


def _segment_heuristic(image: Image.Image) -> tuple[np.ndarray, dict[int, str]]:
    """Color-based fallback when OneFormer isn't available.

    Very rough: bright-blue top regions → sky, green → vegetation, brown/grey
    middle → buildings, dark-bottom → ground. Better than nothing for testing
    the pipeline plumbing.
    """
    arr = np.array(image.convert("RGB"))
    H, W, _ = arr.shape
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    out = np.full((H, W), 6, dtype=np.int8)  # default: building

    # Top third — if mostly blue, mark sky
    sky_mask = np.zeros_like(out, dtype=bool)
    sky_mask[:H // 3] = (b > 130) & (b > r) & (b > g - 20)
    out[sky_mask] = 1

    # Green-dominant regions → vegetation
    out[(g > r + 20) & (g > b + 10)] = 7

    # Bottom band, low brightness → ground
    bottom = np.zeros_like(out, dtype=bool)
    bottom[int(H * 0.6):] = True
    lum = (r.astype(int) + g + b) // 3
    out[bottom & (lum < 130)] = 2

    return out, CLASSES


def project_segmentation_to_voxels(
    image: Image.Image,
    semantic: np.ndarray,
    store: VoxelStore,
    camera_distance: float = 4.0,
    fov_deg: float = 50.0,
) -> int:
    """Use Depth-Anything depth to lift the segmentation into 3D, then vote
    into voxels. This is what gives the initial voxel grid its semantic
    labels (the Hunyuan3D mesh voxels are all 'building' until they get
    overridden by these votes).
    """
    import torch
    try:
        from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
        proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        model = DepthAnythingForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ).eval()
        if torch.cuda.is_available():
            model = model.cuda()
    except Exception as e:
        print(f"  [bootstrap] Depth-Anything unavailable ({e}); using flat depth")
        depth = np.full(image.size[::-1], camera_distance, dtype=np.float32)
    else:
        inp = proc(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            pred = model(**inp).predicted_depth.squeeze().cpu().numpy()
        # Resize to image dims, then normalize to a plausible meter range
        depth_img = Image.fromarray(pred).resize(image.size, Image.BILINEAR)
        depth = np.array(depth_img, dtype=np.float32)
        # Depth-Anything outputs relative depth — invert/normalize to [near, far] meters
        depth = depth.max() - depth
        depth = depth / (depth.max() + 1e-6)
        depth = 1.0 + depth * (camera_distance * 1.5)  # ~1m to ~6m range

    H, W = depth.shape
    aspect = W / H
    fy = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    fx = fy / aspect

    # Camera assumed at origin looking down -Z, the natural front-view convention.
    # Pixel (u, v) in [0,W) × [0,H) → normalized device → world ray
    us = (np.arange(W) + 0.5) / W * 2.0 - 1.0
    vs = (np.arange(H) + 0.5) / H * 2.0 - 1.0
    UU, VV = np.meshgrid(us, vs)
    dir_x = UU / fx
    dir_y = -VV / fy
    dir_z = -np.ones_like(UU)
    norm = np.sqrt(dir_x**2 + dir_y**2 + 1.0)
    dx, dy, dz = dir_x / norm, dir_y / norm, dir_z / norm

    cam_pos = np.array([0.0, 1.5, camera_distance], dtype=np.float32)
    n_added = 0
    # Sample seed image pixels for per-voxel color
    img_arr = np.asarray(image.resize((W, H)), dtype=np.float32) / 255.0
    if img_arr.ndim == 2:   # grayscale fallback
        img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)
    # Subsample for speed
    stride = max(1, W // 256)
    for v in range(0, H, stride):
        for u in range(0, W, stride):
            cls = int(semantic[v, u])
            if cls == 1:    # sky — don't project, would explode at far
                continue
            d = float(depth[v, u])
            wx = cam_pos[0] + dx[v, u] * d
            wy = cam_pos[1] + dy[v, u] * d
            wz = cam_pos[2] + dz[v, u] * d
            idx = store.world_to_voxel((float(wx), float(wy), float(wz)))
            if store.in_bounds(idx):
                store.vote(idx, cls)
                n_added += 1
                # Offset from voxel center to actual ray-hit
                center = store.voxel_to_world(idx)
                offset = (float(wx) - center[0], float(wy) - center[1], float(wz) - center[2])
                # Normal: from hit point back toward the seed camera (this view
                # 'lit' the surface, so the normal faces this camera).
                hit = np.array([wx, wy, wz], dtype=np.float32)
                n_vec = cam_pos - hit
                nlen = float(np.linalg.norm(n_vec))
                if nlen > 1e-6:
                    nrm = (float(n_vec[0] / nlen), float(n_vec[1] / nlen), float(n_vec[2] / nlen))
                else:
                    nrm = None
                # Color: the seed image pixel
                col = img_arr[v, u]
                rgb = (float(col[0]), float(col[1]), float(col[2]))
                store.vote_appearance(idx, rgb, offset, nrm)

    return n_added


def bootstrap_scene(
    scene_id: str,
    scene_image: pathlib.Path,
    mesh_path: pathlib.Path,
    extent: float = 4.0,
    resolution: int = 96,
) -> VoxelStore:
    """Full bootstrap: voxelize mesh + project segmented input image."""
    print(f"[bootstrap] {scene_id}: initializing voxel store ({resolution}³ @ ±{extent}m)")
    store = VoxelStore(extent=extent, resolution=resolution)

    print(f"[bootstrap] {scene_id}: voxelizing {mesh_path.name}")
    n = voxelize_mesh(mesh_path, store, default_class=6)
    print(f"[bootstrap] {scene_id}: {n} mesh-surface votes")

    print(f"[bootstrap] {scene_id}: running segmentation on {scene_image.name}")
    img = Image.open(scene_image).convert("RGB")
    semantic, _labels = semantic_segment(img)
    n2 = project_segmentation_to_voxels(img, semantic, store)
    print(f"[bootstrap] {scene_id}: {n2} segmented-pixel votes projected")

    print(f"[bootstrap] {scene_id}: active voxels = {len(store.cells)}, "
          f"converged = {sum(1 for i in store.cells if store.is_converged(i))}")
    return store
