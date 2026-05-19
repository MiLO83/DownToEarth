#!/usr/bin/env python3
"""
run_all.py — Full content generation pipeline for one or all scenes/characters.

Orchestrates:
  1. gen_scene.py             → assets-raw/<scene>/scene.png
  2. scene_to_3d.py            → viewer/public/assets/scenes/<scene>/mesh.glb
  3. extract_walkable.py       → viewer/public/assets/scenes/<scene>/walkable.json
  4. gen_character.py          → viewer/public/assets/characters/<char>.glb

Usage:
    python run_all.py                            # everything
    python run_all.py --scenes-only              # skip characters
    python run_all.py --characters-only          # skip scenes
    python run_all.py --scene hamlet-square      # one scene
    python run_all.py --character hero           # one character
"""
import argparse
import subprocess
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable


def run(script, extra=()):
    cmd = [PY, str(HERE / script), *extra]
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"!!! {script} failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-only", action="store_true")
    ap.add_argument("--characters-only", action="store_true")
    ap.add_argument("--scene", help="Single scene id")
    ap.add_argument("--character", help="Single character id")
    ap.add_argument("--model", default="hunyuan3d", choices=["trellis", "hunyuan3d"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--comfyui", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    common = ["--comfyui", args.comfyui] + (["--force"] if args.force else [])
    one_scene = ["--scene", args.scene] if args.scene else []
    one_char = ["--character", args.character] if args.character else []

    do_scenes = not args.characters_only
    do_chars = not args.scenes_only

    if do_scenes:
        run("gen_scene.py", common + one_scene)
        run("scene_to_3d.py", common + one_scene + ["--model", args.model])
        # extract_walkable doesn't talk to ComfyUI, doesn't take --comfyui or --force
        run("extract_walkable.py", one_scene)

    if do_chars:
        run("gen_character.py", common + one_char)

    print("\n=== Pipeline complete ===")
    print("Now run the viewer:  cd viewer && npm install && npm start")


if __name__ == "__main__":
    main()
