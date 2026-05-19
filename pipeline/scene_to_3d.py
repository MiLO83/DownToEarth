#!/usr/bin/env python3
"""
scene_to_3d.py — Convert a generated scene PNG into a 3D mesh via Trellis or
Hunyuan3D 2 (running locally through ComfyUI).

Reads assets-raw/<scene-id>/scene.png and produces:
  - assets-raw/<scene-id>/mesh.glb              (the 3D scene)
  - viewer/public/assets/scenes/<scene-id>/mesh.glb  (deployed copy)

Usage:
    python scene_to_3d.py [--scene HAMLET_SQUARE] [--model trellis|hunyuan3d]
                          [--comfyui http://127.0.0.1:8188]

Notes:
  - Trellis ComfyUI nodes: https://github.com/jtydhr88/ComfyUI-Trellis
  - Hunyuan3D ComfyUI nodes: https://github.com/kijai/ComfyUI-Hunyuan3DWrapper
  - The workflow JSON under workflows/ assumes one of these is installed.
"""

from __future__ import annotations
import argparse
import base64
import json
import pathlib
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid
import websocket


REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO / "viewer" / "public" / "manifest.json"
WORKFLOWS = pathlib.Path(__file__).resolve().parent / "workflows"
ASSETS_RAW = REPO / "assets-raw"
ASSETS_DEPLOY = REPO / "viewer" / "public" / "assets" / "scenes"


def upload_image(comfy_url: str, image_path: pathlib.Path) -> str:
    """Upload PNG to ComfyUI's input directory; returns the filename to reference."""
    boundary = "----DownToEarthBoundary"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"image\"; filename=\"{image_path.name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{comfy_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["name"]


def post_workflow(comfy_url, workflow, client_id):
    body = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{comfy_url}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"\n!!! ComfyUI rejected the workflow ({e.code}):\n{err_body}\n", file=sys.stderr)
        debug_path = pathlib.Path(__file__).resolve().parent / "_last_workflow_attempt.json"
        debug_path.write_text(json.dumps(workflow, indent=2))
        print(f"!!! Workflow JSON dumped to: {debug_path}\n", file=sys.stderr)
        raise


COMFYUI_OUTPUT_DIR = pathlib.Path(
    r"C:/Users/rxcam/ComfyUI_portable/ComfyUI_windows_portable/ComfyUI/output"
)


def wait_for_mesh(comfy_url, prompt_id, client_id, filename_prefix=None) -> bytes:
    """Wait for the workflow to finish, then locate the produced .glb.

    Hy3DExportMesh (and most mesh-saving nodes) write to disk directly without
    registering anything in the prompt history's `outputs`. So we drop down to
    the filesystem: look in ComfyUI's output directory for the most recent
    .glb matching our filename_prefix.
    """
    ws = websocket.create_connection(comfy_url.replace("http", "ws") + f"/ws?clientId={client_id}",
                                     timeout=1800)
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "executing":
                d = data.get("data", {})
                if d.get("node") is None and d.get("prompt_id") == prompt_id:
                    break
            elif data.get("type") == "execution_error":
                err = data.get("data", {})
                raise RuntimeError(
                    f"ComfyUI execution error in node {err.get('node_id')} "
                    f"({err.get('node_type')}): {err.get('exception_message', '?')}"
                )
    finally:
        ws.close()

    # First try history (some nodes DO register outputs)
    with urllib.request.urlopen(f"{comfy_url}/history/{prompt_id}") as r:
        hist = json.loads(r.read())
    outputs = hist[prompt_id]["outputs"]
    for node_out in outputs.values():
        for key in ("meshes", "glb", "files"):
            for f in (node_out.get(key) or []):
                name = f.get("filename") or f.get("name")
                if not name:
                    continue
                qs = urllib.parse.urlencode({
                    "filename": name,
                    "subfolder": f.get("subfolder", ""),
                    "type": f.get("type", "output"),
                })
                with urllib.request.urlopen(f"{comfy_url}/view?{qs}") as r:
                    return r.read()

    # Fallback: scan ComfyUI's output directory for the newest matching .glb
    if filename_prefix:
        prefix_parts = filename_prefix.split("/")
        search_dir = COMFYUI_OUTPUT_DIR.joinpath(*prefix_parts[:-1])
        name_prefix = prefix_parts[-1]
        if search_dir.is_dir():
            candidates = sorted(
                search_dir.glob(f"{name_prefix}_*.glb"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0].read_bytes()

    raise RuntimeError(
        f"No mesh found for prompt {prompt_id}. Checked history and "
        f"{COMFYUI_OUTPUT_DIR}/{filename_prefix}*.glb. "
        f"Status: {hist[prompt_id].get('status', {})}"
    )


def build_workflow(template, scene_image_filename, scene_id, model_name):
    wf = json.loads(json.dumps(template))
    wf = {k: v for k, v in wf.items() if isinstance(v, dict)}
    for node_id, node in wf.items():
        cls = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if cls == "LoadImage":
            inputs["image"] = scene_image_filename
        elif cls in ("Hy3DExportMesh", "Hunyuan3DSave", "SaveGLB", "TrellisSaveMesh"):
            inputs["filename_prefix"] = f"downtoearth/{scene_id}_{model_name}"
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="Only this scene id")
    ap.add_argument("--model", default="trellis", choices=["trellis", "hunyuan3d"])
    ap.add_argument("--comfyui", default="http://127.0.0.1:8188")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    workflow_file = {
        "trellis": "trellis_scene.json",
        "hunyuan3d": "hunyuan3d_scene.json",
    }[args.model]
    template = json.loads((WORKFLOWS / workflow_file).read_text())

    manifest = json.loads(MANIFEST.read_text())
    targets = [args.scene] if args.scene else list(manifest["scenes"].keys())
    client_id = str(uuid.uuid4())

    for sid in targets:
        png = ASSETS_RAW / sid / "scene.png"
        if not png.exists():
            print(f"  ! {sid}: missing {png}, run gen_scene.py first", file=sys.stderr)
            continue
        out_glb = ASSETS_RAW / sid / "mesh.glb"
        if out_glb.exists() and not args.force:
            print(f"  ✓ {sid}: mesh.glb exists (use --force to redo)")
        else:
            print(f"  → {sid}: uploading scene image to ComfyUI…")
            fname = upload_image(args.comfyui, png)
            wf = build_workflow(template, fname, sid, args.model)
            print(f"  → {sid}: submitting {args.model} workflow…")
            pid = post_workflow(args.comfyui, wf, client_id)
            filename_prefix = f"downtoearth/{sid}_{args.model}"
            glb_bytes = wait_for_mesh(args.comfyui, pid, client_id, filename_prefix)
            out_glb.write_bytes(glb_bytes)
            print(f"  ✓ {sid}: wrote {out_glb} ({len(glb_bytes) // 1024} KB)")

        # Deploy to viewer
        deploy_dir = ASSETS_DEPLOY / sid
        deploy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_glb, deploy_dir / "mesh.glb")
        print(f"  ✓ {sid}: deployed to {deploy_dir / 'mesh.glb'}")


if __name__ == "__main__":
    main()
