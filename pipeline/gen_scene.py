#!/usr/bin/env python3
"""
gen_scene.py — Generate a JRPG scene image via ComfyUI API using Juggernaut XL.

Reads ../viewer/public/manifest.json, finds scenes that don't yet have a baked
scene image, and submits a workflow per scene to a running ComfyUI instance.
Writes the result to assets-raw/<scene-id>/scene.png so the next pipeline
stage (scene_to_3d.py) can consume it.

Usage:
    python gen_scene.py [--scene HAMLET_SQUARE] [--force] [--comfyui http://127.0.0.1:8188]

Requires ComfyUI running locally with Juggernaut XL checkpoint installed at the
path specified in workflows/juggernaut_scene.json.
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request
import uuid
import websocket  # pip install websocket-client


REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO / "viewer" / "public" / "manifest.json"
WORKFLOWS = pathlib.Path(__file__).resolve().parent / "workflows"
ASSETS_RAW = REPO / "assets-raw"


def post_workflow(comfy_url: str, workflow: dict, client_id: str) -> str:
    """Submit a workflow to ComfyUI's /prompt endpoint, return prompt_id."""
    body = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{comfy_url}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        # Surface ComfyUI's actual error so we can see what node it didn't like
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"\n!!! ComfyUI rejected the workflow ({e.code}):\n{err_body}\n", file=sys.stderr)
        # Also dump the workflow we sent so it can be inspected
        debug_path = pathlib.Path(__file__).resolve().parent / "_last_workflow_attempt.json"
        debug_path.write_text(json.dumps(workflow, indent=2))
        print(f"!!! Workflow JSON dumped to: {debug_path}\n", file=sys.stderr)
        raise


def wait_for_image(comfy_url: str, prompt_id: str, client_id: str) -> bytes:
    """Block on the websocket until the prompt completes, then fetch the image."""
    ws_url = comfy_url.replace("http", "ws") + f"/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url, timeout=600)
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "executing":
                d = data.get("data", {})
                if d.get("node") is None and d.get("prompt_id") == prompt_id:
                    break  # done
    finally:
        ws.close()

    # Fetch history → find the saved image
    with urllib.request.urlopen(f"{comfy_url}/history/{prompt_id}") as r:
        hist = json.loads(r.read())
    outputs = hist[prompt_id]["outputs"]
    for node_out in outputs.values():
        for img in node_out.get("images", []):
            qs = urllib.parse.urlencode({
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            })
            with urllib.request.urlopen(f"{comfy_url}/view?{qs}") as r:
                return r.read()
    raise RuntimeError(f"No image in prompt history for {prompt_id}")


def build_workflow(template: dict, prompt: str, neg_prompt: str, seed: int, checkpoint: str, out_prefix: str) -> dict:
    """Patch the workflow template with this scene's prompt/seed/checkpoint."""
    wf = json.loads(json.dumps(template))  # deep copy
    # Strip any non-node entries (e.g. _comment) so ComfyUI's validator doesn't choke
    wf = {k: v for k, v in wf.items() if isinstance(v, dict)}
    for node_id, node in wf.items():
        cls = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if cls == "CLIPTextEncode":
            title = node.get("_meta", {}).get("title", "").lower()
            if "negative" in title:
                inputs["text"] = neg_prompt
            else:
                inputs["text"] = prompt
        elif cls == "KSampler":
            inputs["seed"] = seed
        elif cls == "CheckpointLoaderSimple":
            inputs["ckpt_name"] = checkpoint
        elif cls == "SaveImage":
            inputs["filename_prefix"] = out_prefix
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="Only generate this scene id (default: all missing)")
    ap.add_argument("--force", action="store_true", help="Regenerate even if image exists")
    ap.add_argument("--comfyui", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    template = json.loads((WORKFLOWS / "juggernaut_scene.json").read_text())

    targets = [args.scene] if args.scene else list(manifest["scenes"].keys())
    client_id = str(uuid.uuid4())

    for sid in targets:
        sdef = manifest["scenes"].get(sid)
        if not sdef:
            print(f"  ! Unknown scene {sid}", file=sys.stderr)
            continue
        gen = sdef.get("generation")
        if not gen:
            print(f"  - {sid}: no generation block, skipping")
            continue
        out_dir = ASSETS_RAW / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / "scene.png"
        if out_png.exists() and not args.force:
            print(f"  ✓ {sid}: scene.png already exists (use --force to redo)")
            continue
        print(f"  → {sid}: submitting workflow…")

        wf = build_workflow(
            template,
            prompt=gen["prompt"],
            neg_prompt=gen.get("negativePrompt", ""),
            seed=gen.get("seed", int(time.time())),
            checkpoint=gen.get("checkpoint", "juggernautXL_v9.safetensors"),
            out_prefix=f"downtoearth/{sid}",
        )
        pid = post_workflow(args.comfyui, wf, client_id)
        img_bytes = wait_for_image(args.comfyui, pid, client_id)
        out_png.write_bytes(img_bytes)
        print(f"  ✓ {sid}: wrote {out_png} ({len(img_bytes) // 1024} KB)")


if __name__ == "__main__":
    main()
