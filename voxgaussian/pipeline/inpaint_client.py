"""
inpaint_client.py — ComfyUI client for depth + semantic inpainting.

The pipeline's `render_voxels` step produces:
  - depth_render:    H×W float meters, NaN where unknown
  - semantic_render: H×W int8 class IDs, 0 where unknown
  - unknown_mask:    H×W bool — True where the view sees through to nothing

We want to fill the unknown regions with diffusion that's:
  - Conditioned on the KNOWN depth + semantic regions (so it doesn't drift)
  - Conditioned on the original input scene image for style consistency
  - Outputs new depth + new semantic for the unknown regions only

This module wraps a ComfyUI workflow that does that via:
  - ControlNet-depth (preserves known depth, hallucinates plausible depth for holes)
  - ControlNet-semantic-map (preserves known classes, hallucinates classes for holes)
  - SDXL or Flux inpaint backbone with the input image as IP-Adapter reference

For v1 the actual ComfyUI workflow JSON is shipped as `workflows/depth_semantic_inpaint.json`
and we patch the input images and masks before submitting. As with the
DownToEarth pipeline, the script falls back gracefully if ComfyUI isn't
configured exactly as expected.
"""
from __future__ import annotations
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import numpy as np
from PIL import Image
import websocket   # pip install websocket-client

from .voxel_store import CLASS_COLORS


COMFYUI_OUTPUT_DIR = pathlib.Path(
    r"C:/Users/rxcam/ComfyUI_portable/ComfyUI_windows_portable/ComfyUI/output"
)


# ─── Depth + semantic extraction (Python-side post-process) ─────────────
# Inpaint workflow returns one RGB. We re-derive depth & semantic locally
# using the same models the bootstrap uses. They're cached after first load.

_DEPTH_MODEL = None
_DEPTH_PROC = None

def _extract_depth(rgb: Image.Image, target_shape: tuple[int, int],
                   depth_range_m: float = 8.0) -> np.ndarray:
    """Run Depth-Anything V2 on an RGB image, return (H, W) float meters."""
    global _DEPTH_MODEL, _DEPTH_PROC
    try:
        import torch
        from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
        if _DEPTH_MODEL is None:
            _DEPTH_PROC = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            _DEPTH_MODEL = DepthAnythingForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf").eval()
            if torch.cuda.is_available():
                _DEPTH_MODEL = _DEPTH_MODEL.cuda()
        inp = _DEPTH_PROC(images=rgb, return_tensors="pt")
        if torch.cuda.is_available():
            inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            pred = _DEPTH_MODEL(**inp).predicted_depth.squeeze().cpu().numpy()
        depth_img = Image.fromarray(pred).resize(target_shape[::-1], Image.BILINEAR)
        d = np.array(depth_img, dtype=np.float32)
        # Invert relative depth (Depth-Anything: larger = closer) → meters
        d = d.max() - d
        d = d / (d.max() + 1e-6)
        return 0.8 + d * depth_range_m
    except Exception as e:
        print(f"[inpaint] depth extraction failed ({e}); returning constant 3m")
        return np.full(target_shape, 3.0, dtype=np.float32)


_SEM_MODEL = None
_SEM_PROC = None

def _extract_semantic(rgb: Image.Image, target_shape: tuple[int, int]) -> np.ndarray:
    """Run OneFormer on an RGB image, remap to our internal class IDs."""
    from .voxel_store import CLASSES
    from .bootstrap import ADE20K_TO_INTERNAL
    global _SEM_MODEL, _SEM_PROC
    try:
        import torch
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        if _SEM_MODEL is None:
            _SEM_PROC = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
            _SEM_MODEL = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_tiny")
            if torch.cuda.is_available():
                _SEM_MODEL = _SEM_MODEL.cuda()
        inputs = _SEM_PROC(images=rgb, task_inputs=["semantic"], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = _SEM_MODEL(**inputs)
        seg = _SEM_PROC.post_process_semantic_segmentation(out, target_sizes=[rgb.size[::-1]])[0]
        seg = seg.cpu().numpy()
        # Remap ADE20K IDs → our classes
        id2label = _SEM_MODEL.config.id2label
        internal = np.zeros(target_shape, dtype=np.int8)
        seg_resized = np.array(Image.fromarray(seg.astype(np.int32)).resize(
            target_shape[::-1], Image.NEAREST))
        for ade_id, label in id2label.items():
            label_lc = label.lower()
            iid = 8  # default to 'prop'
            for kw, cid in ADE20K_TO_INTERNAL.items():
                if kw in label_lc:
                    iid = cid
                    break
            internal[seg_resized == int(ade_id)] = iid
        return internal
    except Exception as e:
        print(f"[inpaint] semantic extraction failed ({e}); returning zeros")
        return np.zeros(target_shape, dtype=np.int8)


class InpaintClient:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188",
                 workflow_path: str | None = None):
        self.url = comfyui_url
        wf_path = workflow_path or (
            pathlib.Path(__file__).resolve().parent.parent / "workflows" / "depth_semantic_inpaint.json"
        )
        self.workflow_path = pathlib.Path(wf_path)
        self.client_id = str(uuid.uuid4())

    # ─── Image helpers ───────────────────────────────────────────────────

    def encode_depth_as_png(self, depth: np.ndarray, out_path: pathlib.Path,
                            depth_range_m: float = 8.0) -> None:
        """Save depth as a single-channel PNG normalized to [0, 255].
        NaN regions become 0 (which signals "unknown" to the inpaint mask
        side-channel — the actual mask is a separate file).
        """
        d = depth.copy()
        d = np.where(np.isnan(d), 0.0, d)
        d = np.clip(d / depth_range_m, 0.0, 1.0)
        Image.fromarray((d * 255).astype(np.uint8)).save(out_path)

    def encode_semantic_as_png(self, semantic: np.ndarray, out_path: pathlib.Path) -> None:
        """Save semantic map as RGB using CLASS_COLORS. ControlNet-segmentation
        consumes RGB images keyed to a palette."""
        rgb = np.zeros((semantic.shape[0], semantic.shape[1], 3), dtype=np.uint8)
        for cid, hex_col in CLASS_COLORS.items():
            c = hex_col.lstrip("#")
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            rgb[semantic == cid] = [r, g, b]
        Image.fromarray(rgb).save(out_path)

    def encode_mask_as_png(self, unknown_mask: np.ndarray, out_path: pathlib.Path) -> None:
        """White = inpaint here, black = keep."""
        Image.fromarray((unknown_mask * 255).astype(np.uint8)).save(out_path)

    # ─── ComfyUI plumbing ────────────────────────────────────────────────

    def upload(self, path: pathlib.Path) -> str:
        boundary = "----DTEBoundary"
        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{self.url}/upload/image", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["name"]

    def submit(self, workflow: dict) -> str:
        body = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        req = urllib.request.Request(
            f"{self.url}/prompt", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"\n!!! ComfyUI rejected the inpaint workflow ({e.code}):\n{err_body}\n",
                  file=sys.stderr)
            raise

    def wait(self, prompt_id: str, filename_prefix: str | None = None) -> list[bytes]:
        """Block on the websocket until the prompt completes; return PNG bytes
        for each image output in the saved order."""
        ws = websocket.create_connection(
            self.url.replace("http", "ws") + f"/ws?clientId={self.client_id}",
            timeout=600,
        )
        try:
            while True:
                msg = ws.recv()
                if isinstance(msg, bytes):
                    continue
                d = json.loads(msg)
                if d.get("type") == "executing":
                    dd = d.get("data", {})
                    if dd.get("node") is None and dd.get("prompt_id") == prompt_id:
                        break
                elif d.get("type") == "execution_error":
                    err = d.get("data", {})
                    raise RuntimeError(
                        f"Inpaint workflow failed in node {err.get('node_id')} "
                        f"({err.get('node_type')}): {err.get('exception_message', '?')}"
                    )
        finally:
            ws.close()

        with urllib.request.urlopen(f"{self.url}/history/{prompt_id}") as r:
            hist = json.loads(r.read())
        outputs = hist[prompt_id]["outputs"]
        results: list[bytes] = []
        for node_out in outputs.values():
            for img in node_out.get("images", []):
                qs = urllib.parse.urlencode({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
                with urllib.request.urlopen(f"{self.url}/view?{qs}") as r:
                    results.append(r.read())
        # Fallback: scan output dir
        if not results and filename_prefix:
            parts = filename_prefix.split("/")
            search_dir = COMFYUI_OUTPUT_DIR.joinpath(*parts[:-1])
            name_prefix = parts[-1]
            if search_dir.is_dir():
                for cand in sorted(search_dir.glob(f"{name_prefix}_*.png"),
                                   key=lambda p: p.stat().st_mtime, reverse=True)[:2]:
                    results.append(cand.read_bytes())
        return results

    # ─── High-level entry point ──────────────────────────────────────────

    def inpaint(self, depth: np.ndarray, semantic: np.ndarray, unknown_mask: np.ndarray,
                input_image_path: pathlib.Path, iteration: int) -> dict:
        """Run a full inpaint pass.

        ComfyUI workflow returns ONE inpainted RGB image. Python then runs
        Depth-Anything + a semantic segmenter on that RGB to produce the new
        depth + semantic maps, which get propagated back into voxel votes.

        Returns dict with `depth` (H×W float meters) and `semantic` (H×W int8)
        for the FULL frame (known regions preserved, unknowns filled).
        """
        if not self.workflow_path.exists():
            print(f"[inpaint] no workflow at {self.workflow_path}, passing through (no-op)")
            return {"depth": depth, "semantic": semantic, "rgb": None}

        # Stage inputs as PNGs and upload to ComfyUI's input/ directory
        scratch = pathlib.Path(self.workflow_path).resolve().parent / "_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        depth_png = scratch / f"depth_in_{iteration}.png"
        sem_png = scratch / f"semantic_in_{iteration}.png"
        mask_png = scratch / f"mask_{iteration}.png"
        self.encode_depth_as_png(depth, depth_png)
        self.encode_semantic_as_png(semantic, sem_png)
        self.encode_mask_as_png(unknown_mask, mask_png)

        depth_remote = self.upload(depth_png)
        sem_remote = self.upload(sem_png)
        mask_remote = self.upload(mask_png)
        ref_remote = self.upload(input_image_path)

        # Patch the workflow: titled LoadImage nodes get the remote filenames,
        # seed gets the iteration so each pass is deterministic-but-different.
        template = json.loads(self.workflow_path.read_text())
        wf = {k: v for k, v in template.items() if isinstance(v, dict)}
        for n in wf.values():
            cls = n.get("class_type", "")
            title = (n.get("_meta", {}).get("title", "") or "").lower()
            if cls == "LoadImage":
                if "reference" in title or "ref " in title:
                    n["inputs"]["image"] = ref_remote
                elif "depth" in title:
                    n["inputs"]["image"] = depth_remote
                elif "mask" in title:
                    n["inputs"]["image"] = mask_remote
                elif "semantic" in title:
                    n["inputs"]["image"] = sem_remote
            elif cls == "KSampler":
                n["inputs"]["seed"] = (iteration * 982451653) & 0xffffffff
            elif cls == "SaveImage":
                n["inputs"]["filename_prefix"] = f"voxgaussian/iter_{iteration:03d}"

        pid = self.submit(wf)
        png_blobs = self.wait(pid, filename_prefix=f"voxgaussian/iter_{iteration:03d}")
        if not png_blobs:
            raise RuntimeError(f"Inpaint returned no images for iteration {iteration}")

        # Decode RGB, then run Depth-Anything + semantic seg locally to derive
        # new depth + semantic. These reuse the same models bootstrap.py uses.
        import io
        rgb_img = Image.open(io.BytesIO(png_blobs[0])).convert("RGB")
        if rgb_img.size != (depth.shape[1], depth.shape[0]):
            rgb_img = rgb_img.resize((depth.shape[1], depth.shape[0]), Image.LANCZOS)

        new_depth = _extract_depth(rgb_img, target_shape=depth.shape)
        new_semantic = _extract_semantic(rgb_img, target_shape=semantic.shape)

        # Keep KNOWN regions from the original render; only adopt inpaint
        # within the unknown_mask. This stops the diffusion result from
        # subtly drifting the parts we already knew.
        known = ~unknown_mask
        out_depth = np.where(known, depth, new_depth)
        out_semantic = np.where(known, semantic, new_semantic).astype(np.int8)
        # Pass the full inpainted RGB through so propagate() can sample per-
        # voxel color at each hit pixel. Always full-frame (we trust the
        # inpaint's color across the whole image; KNOWN regions in depth are
        # preserved separately above).
        rgb_arr = np.asarray(rgb_img, dtype=np.float32) / 255.0   # H, W, 3 in 0..1
        return {"depth": out_depth, "semantic": out_semantic, "rgb": rgb_arr}

    def _decode_depth(self, blob: bytes, shape: tuple[int, int],
                      depth_range_m: float = 8.0) -> np.ndarray:
        import io
        img = Image.open(io.BytesIO(blob)).convert("L").resize(shape[::-1], Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        return (arr / 255.0) * depth_range_m

    def _decode_semantic(self, blob: bytes, shape: tuple[int, int]) -> np.ndarray:
        import io
        img = Image.open(io.BytesIO(blob)).convert("RGB").resize(shape[::-1], Image.NEAREST)
        rgb = np.array(img, dtype=np.uint8)
        out = np.zeros(shape, dtype=np.int8)
        for cid, hex_col in CLASS_COLORS.items():
            c = hex_col.lstrip("#")
            target = np.array([int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)], dtype=np.uint8)
            mask = np.all(rgb == target, axis=-1)
            out[mask] = cid
        return out
