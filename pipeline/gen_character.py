#!/usr/bin/env python3
"""
gen_character.py — Generate a consistent 3D character model.

Pipeline:
  1. Generate a clean front-facing reference image of the character via
     Juggernaut XL (or use existing characters/<id>/reference.png if present).
  2. Generate a back-facing image of the same character using IP-Adapter for
     visual consistency (so the back-view-hallucination in step 3 has real data).
  3. Run Hunyuan3D 2 multi-view → GLB mesh.
  4. Deploy to viewer/public/assets/characters/<id>.glb and a portrait PNG to
     viewer/public/assets/portraits/<id>.png.

Usage:
    python gen_character.py [--character hero] [--force]
                            [--comfyui http://127.0.0.1:8188]
"""

from __future__ import annotations
import argparse
import json
import pathlib
import shutil
import sys
import urllib.parse
import urllib.request
import uuid
import websocket


REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO / "viewer" / "public" / "manifest.json"
WORKFLOWS = pathlib.Path(__file__).resolve().parent / "workflows"
ASSETS_RAW = REPO / "assets-raw"
DEPLOY_CHARS = REPO / "viewer" / "public" / "assets" / "characters"
DEPLOY_PORTRAITS = REPO / "viewer" / "public" / "assets" / "portraits"


def post(comfy_url, workflow, client_id):
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


def wait(comfy_url, prompt_id, client_id, kinds=("images", "meshes", "glb"),
         filename_prefix=None):
    ws = websocket.create_connection(comfy_url.replace("http", "ws") + f"/ws?clientId={client_id}",
                                     timeout=1800)
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
                print(f"\n!!! ComfyUI execution error in node {err.get('node_id')} ({err.get('node_type')}):\n"
                      f"    {err.get('exception_message', '?')}\n"
                      f"    traceback:\n{''.join(err.get('traceback', []))}\n", file=sys.stderr)
                break
    finally:
        ws.close()
    with urllib.request.urlopen(f"{comfy_url}/history/{prompt_id}") as r:
        hist = json.loads(r.read())
    files = []
    for node_out in hist[prompt_id]["outputs"].values():
        for k in kinds:
            for f in node_out.get(k, []):
                qs = urllib.parse.urlencode({
                    "filename": f["filename"],
                    "subfolder": f.get("subfolder", ""),
                    "type": f.get("type", "output"),
                })
                with urllib.request.urlopen(f"{comfy_url}/view?{qs}") as r:
                    files.append((f["filename"], r.read()))
    # Fallback: scan ComfyUI's output dir for files matching filename_prefix.
    # Hy3DExportMesh saves to disk without registering the output in history.
    if not files and filename_prefix:
        parts = filename_prefix.split("/")
        search_dir = COMFYUI_OUTPUT_DIR.joinpath(*parts[:-1])
        name_prefix = parts[-1]
        if search_dir.is_dir():
            for ext in (".glb", ".obj", ".png"):
                candidates = sorted(
                    search_dir.glob(f"{name_prefix}_*{ext}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    files.append((candidates[0].name, candidates[0].read_bytes()))
                    break
    if not files:
        status = hist[prompt_id].get("status", {})
        print(f"\n!!! No output files for prompt {prompt_id}. Status: {json.dumps(status, indent=2)}\n",
              file=sys.stderr)
    return files


def upload_image(comfy_url: str, image_path: pathlib.Path) -> str:
    boundary = "----DownToEarthBoundary"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"image\"; filename=\"{image_path.name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{comfy_url}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["name"]


def build_reference_workflow(template, prompt, neg_prompt, seed):
    wf = json.loads(json.dumps(template))
    wf = {k: v for k, v in wf.items() if isinstance(v, dict)}
    for nid, n in wf.items():
        cls = n.get("class_type", "")
        ip = n.get("inputs", {})
        if cls == "CLIPTextEncode":
            title = n.get("_meta", {}).get("title", "").lower()
            ip["text"] = neg_prompt if "negative" in title else prompt
        elif cls == "KSampler":
            ip["seed"] = seed
    return wf


def build_backview_workflow(template, front_image_filename, prompt, seed):
    wf = json.loads(json.dumps(template))
    wf = {k: v for k, v in wf.items() if isinstance(v, dict)}
    for nid, n in wf.items():
        cls = n.get("class_type", "")
        ip = n.get("inputs", {})
        if cls == "LoadImage":
            ip["image"] = front_image_filename
        elif cls == "CLIPTextEncode":
            title = n.get("_meta", {}).get("title", "").lower()
            if "negative" not in title:
                ip["text"] = prompt + ", back view, character from behind"
        elif cls == "KSampler":
            ip["seed"] = seed + 1
    return wf


def build_3d_workflow(template, front_filename, back_filename, char_id):
    wf = json.loads(json.dumps(template))
    wf = {k: v for k, v in wf.items() if isinstance(v, dict)}
    # We rely on _meta.title to disambiguate front vs back LoadImage nodes
    for nid, n in wf.items():
        cls = n.get("class_type", "")
        ip = n.get("inputs", {})
        title = (n.get("_meta", {}).get("title", "") or "").lower()
        if cls == "LoadImage":
            if "back" in title:
                ip["image"] = back_filename
            else:
                ip["image"] = front_filename
        elif cls in ("Hy3DExportMesh", "Hunyuan3DSave", "SaveGLB"):
            ip["filename_prefix"] = f"downtoearth/character_{char_id}"
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", help="Only this character id")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--images-only", action="store_true",
                    help="Skip the Hunyuan3D mesh stage (useful while Hunyuan3D isn't installed yet)")
    ap.add_argument("--comfyui", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    targets = [args.character] if args.character else list(manifest["characters"].keys())
    client_id = str(uuid.uuid4())

    ref_template = json.loads((WORKFLOWS / "character_reference.json").read_text())
    backview_template = json.loads((WORKFLOWS / "character_backview.json").read_text())
    mesh_template = json.loads((WORKFLOWS / "hunyuan3d_character.json").read_text())

    for cid in targets:
        cdef = manifest["characters"].get(cid)
        if not cdef:
            print(f"  ! Unknown character {cid}", file=sys.stderr); continue
        appearance = cdef.get("appearance", {})
        char_raw = ASSETS_RAW / "characters" / cid
        char_raw.mkdir(parents=True, exist_ok=True)
        front_png = char_raw / "front.png"
        back_png = char_raw / "back.png"
        glb = char_raw / "model.glb"

        if not front_png.exists() or args.force:
            print(f"  → {cid}: generating front-view reference…")
            wf = build_reference_workflow(ref_template,
                appearance.get("prompt", ""),
                "blurry, low quality, deformed, multiple people",
                seed=hash(cid) & 0xffff)
            pid = post(args.comfyui, wf, client_id)
            files = wait(args.comfyui, pid, client_id, kinds=("images",))
            if not files: raise RuntimeError("no reference image returned")
            front_png.write_bytes(files[0][1])

        if not back_png.exists() or args.force:
            print(f"  → {cid}: generating back-view (IP-Adapter from front)…")
            front_uploaded = upload_image(args.comfyui, front_png)
            wf = build_backview_workflow(backview_template, front_uploaded,
                appearance.get("prompt", ""), seed=hash(cid) & 0xffff)
            pid = post(args.comfyui, wf, client_id)
            files = wait(args.comfyui, pid, client_id, kinds=("images",))
            if not files: raise RuntimeError("no back image returned")
            back_png.write_bytes(files[0][1])

        if args.images_only:
            DEPLOY_PORTRAITS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(front_png, DEPLOY_PORTRAITS / f"{cid}.png")
            print(f"  ✓ {cid}: 2D images done (mesh stage skipped — re-run without --images-only after Hunyuan3D is installed)")
            continue

        if not glb.exists() or args.force:
            print(f"  → {cid}: generating 3D mesh via Hunyuan3D 2…")
            f_up = upload_image(args.comfyui, front_png)
            b_up = upload_image(args.comfyui, back_png)
            wf = build_3d_workflow(mesh_template, f_up, b_up, cid)
            pid = post(args.comfyui, wf, client_id)
            files = wait(args.comfyui, pid, client_id, kinds=("meshes", "glb"),
                         filename_prefix=f"downtoearth/character_{cid}")
            if not files: raise RuntimeError("no mesh returned")
            glb.write_bytes(files[0][1])

        DEPLOY_CHARS.mkdir(parents=True, exist_ok=True)
        DEPLOY_PORTRAITS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(glb, DEPLOY_CHARS / f"{cid}.glb")
        shutil.copy2(front_png, DEPLOY_PORTRAITS / f"{cid}.png")
        print(f"  ✓ {cid}: deployed {DEPLOY_CHARS / f'{cid}.glb'}")


if __name__ == "__main__":
    main()
