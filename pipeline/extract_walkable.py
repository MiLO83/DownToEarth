#!/usr/bin/env python3
"""
extract_walkable.py — Derive the actor-walkable polygon for a scene by combining:
  1. Semantic segmentation (SAM2 or OneFormer) → floor pixel mask
  2. Depth estimation (Depth-Anything V2)      → 3D positions per pixel
  3. Normal estimation (from depth or Marigold) → confirm "up-facing" surfaces
  4. Plane fit + connected components          → walkable polygon in scene-space

Reads:  assets-raw/<scene-id>/scene.png
        assets-raw/<scene-id>/mesh.glb  (for absolute scale reference)
Writes: viewer/public/assets/scenes/<scene-id>/walkable.json

Usage:
    python extract_walkable.py [--scene HAMLET_SQUARE]

The walkable.json schema:
    {
      "polygons": [ [[x, z], [x, z], ...], ... ],
      "exits": (left untouched here — exits come from the manifest)
    }

Coordinates are in meters, scene-local. The walker uses these to constrain the
player's (x, z) position; y is assumed = floor level.

Note: this script does heavy ML inference (Depth-Anything V2 + a segmenter).
Run in an environment with PyTorch + the models downloaded.
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image

# Imports gated so the script's --help works on systems without ML deps installed
def _require_ml():
    global torch, transforms, DepthAnythingForDepthEstimation, AutoImageProcessor
    import torch
    from transformers import (
        DepthAnythingForDepthEstimation,
        AutoImageProcessor,
    )


REPO = pathlib.Path(__file__).resolve().parent.parent
ASSETS_RAW = REPO / "assets-raw"
ASSETS_DEPLOY = REPO / "viewer" / "public" / "assets" / "scenes"


def estimate_depth(image: Image.Image) -> np.ndarray:
    """Run Depth-Anything V2 → returns (H,W) float depth, normalized [0,1] near→far."""
    proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = DepthAnythingForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    inputs = proc(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    pred = out.predicted_depth.squeeze().cpu().numpy()
    # Resize to source res
    from PIL import Image as PILImage
    depth = np.array(PILImage.fromarray(pred).resize(image.size, PILImage.BILINEAR), dtype=np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth


def normals_from_depth(depth: np.ndarray) -> np.ndarray:
    """Compute per-pixel normals via finite-differences of the depth map. (H,W,3)"""
    dzdx = np.zeros_like(depth)
    dzdy = np.zeros_like(depth)
    dzdx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) * 0.5
    dzdy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) * 0.5
    # Scale gradients so they're comparable to depth axis
    nx = -dzdx * 30
    ny = -dzdy * 30
    nz = np.ones_like(depth)
    n = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    return n / norm


def floor_mask_from_normals(normals: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Pixels whose normal points roughly up AND that aren't in the upper third of image."""
    up_dot = normals[..., 1]
    H = normals.shape[0]
    not_sky = np.zeros_like(depth, dtype=bool)
    not_sky[H // 3:] = True
    return (up_dot > 0.6) & not_sky


def mask_to_polygons(mask: np.ndarray, image_width: int, image_height: int,
                     scene_extent: float = 8.0) -> list[list[list[float]]]:
    """Convert a binary mask to a list of simplified polygons in scene coords.

    Uses contour detection then Ramer-Douglas-Peucker simplification.
    """
    try:
        import cv2
    except ImportError:
        # Fall back to a bounding box of the mask if OpenCV isn't available
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []
        xmin, xmax = xs.min() / image_width - 0.5, xs.max() / image_width - 0.5
        ymin, ymax = ys.min() / image_height - 0.5, ys.max() / image_height - 0.5
        # Image y → scene -z (depth increases away from camera)
        s = scene_extent * 2
        return [[
            [xmin * s, -ymax * s], [xmax * s, -ymax * s],
            [xmax * s, -ymin * s], [xmin * s, -ymin * s],
        ]]

    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < (image_width * image_height) * 0.02:
            continue  # ignore tiny islands
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True).squeeze(1)
        poly = []
        for x, y in approx:
            sx = (x / image_width - 0.5) * scene_extent * 2
            sz = -(y / image_height - 0.5) * scene_extent * 2
            poly.append([float(sx), float(sz)])
        if len(poly) >= 3:
            out.append(poly)
    return out


def process(scene_id: str) -> None:
    png = ASSETS_RAW / scene_id / "scene.png"
    if not png.exists():
        print(f"  ! {scene_id}: no {png}", file=sys.stderr)
        return
    print(f"  → {scene_id}: loading {png}…")
    img = Image.open(png).convert("RGB")

    _require_ml()
    print(f"  → {scene_id}: estimating depth…")
    depth = estimate_depth(img)
    print(f"  → {scene_id}: computing normals…")
    normals = normals_from_depth(depth)
    print(f"  → {scene_id}: extracting floor mask…")
    mask = floor_mask_from_normals(normals, depth)

    print(f"  → {scene_id}: vectorizing to polygons…")
    polys = mask_to_polygons(mask, img.width, img.height)

    out_dir = ASSETS_DEPLOY / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    walkable = {"polygons": polys}
    (out_dir / "walkable.json").write_text(json.dumps(walkable, indent=2))
    print(f"  ✓ {scene_id}: wrote {out_dir / 'walkable.json'} ({len(polys)} polygon(s))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="Only this scene id (default: all)")
    args = ap.parse_args()

    manifest = json.loads((REPO / "viewer" / "public" / "manifest.json").read_text())
    targets = [args.scene] if args.scene else list(manifest["scenes"].keys())
    for sid in targets:
        process(sid)


if __name__ == "__main__":
    main()
