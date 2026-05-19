# Down To Earth

> 🎮 **Live UVW atlas demo:** https://downtoearth-9lq.pages.dev
>
> Interactive proof of the bidirectional voxel ↔ RGB atlas — drag to orbit,
> hover anywhere to see your screen pixel's color literally *being* a voxel
> coordinate. Side-by-side: class-colored vs canonical-RGB renderings of
> the same scene. See [`voxgaussian/README.md`](voxgaussian/README.md) and
> [`voxgaussian/pipeline/uvw_atlas.py`](voxgaussian/pipeline/uvw_atlas.py)
> for the design.

A JRPG-style WebXR walker. Backgrounds and characters are AI-generated locally (Juggernaut XL → Hunyuan3D / Trellis), converted to 3D scenes with walkable polygons (Depth-Anything V2 + segmentation), and traversed in-engine by a 3D character on Quest 3 (immersive VR or AR passthrough) or any modern desktop browser.

Inspired by Final Fantasy VII / IX — 3D characters on richly-rendered, fixed-camera-feeling scenes — but the scenes are actual 3D geometry so you can walk freely.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  pipeline/                                                  │
│    gen_scene.py         Juggernaut XL  → scene PNG          │
│    scene_to_3d.py       Trellis / Hunyuan3D  → mesh GLB     │
│    extract_walkable.py  Depth-Anything V2  → walkable poly  │
│    gen_character.py     IP-Adapter + Hunyuan3D MV → char GLB│
│    run_all.py           Orchestrates the whole pipeline     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼  deposits artifacts into
┌─────────────────────────────────────────────────────────────┐
│  viewer/public/                                             │
│    manifest.json                          (scene graph)     │
│    assets/scenes/<id>/mesh.glb            (3D environment)  │
│    assets/scenes/<id>/walkable.json       (where you can go)│
│    assets/characters/<id>.glb             (3D actors)       │
│    assets/portraits/<id>.png              (UI portraits)    │
│    index.html  +  js/*.js                 (Three.js + WebXR)│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼  served on http://localhost:5173
                              ▼  loadable in Meta Browser on Quest 3
```

## Quick start (placeholder content, no GPU work)

The viewer ships with stub walkable polygons and placeholder characters so you can run it immediately and verify the controls.

```sh
cd viewer
npm start
# open http://localhost:5173
```

Controls (flat-screen):
- **W A S D** or arrows — walk
- **Mouse drag** — orbit camera
- **Space / E / Enter** — interact with nearby NPC

Controls (Quest 3, click ENTER VR or ENTER PASSTHROUGH):
- **Left thumbstick** — walk
- **Right thumbstick** — turn
- **Trigger** — interact

Without baked content, you'll see placeholder ground planes and stick-figure capsule characters. Mostly proves the walking + scene-transitions + interaction loop works. The hamlet-square scene has two NPC capsules you can interact with via Space.

## Full pipeline (generate real content)

### Prerequisites

1. **ComfyUI** running locally — https://github.com/comfyanonymous/ComfyUI
   - Default expected at `http://127.0.0.1:8188`
2. **Custom nodes installed in ComfyUI**:
   - [ComfyUI-Trellis](https://github.com/jtydhr88/ComfyUI-Trellis) (or use the Hunyuan path)
   - [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
   - [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) (for character consistency)
3. **Checkpoints in ComfyUI `models/checkpoints/`**:
   - Juggernaut XL v9 (or later)
4. **Python deps for the pipeline orchestration**:
   ```sh
   cd pipeline
   pip install -r requirements.txt
   ```

### Run the pipeline

Generate everything:
```sh
cd pipeline
python run_all.py
```

Or step-by-step:
```sh
python gen_scene.py                   # Juggernaut XL → scene PNGs
python scene_to_3d.py                 # PNGs → 3D meshes (Hunyuan3D by default)
python extract_walkable.py            # Compute walkable polygons from depth + normals
python gen_character.py               # Front + back image + Hunyuan3D multi-view → character GLB
```

Just one scene / one character:
```sh
python run_all.py --scene hamlet-square
python run_all.py --character hero
```

Force regenerate:
```sh
python run_all.py --force
```

### After generation

Files land in:
```
viewer/public/assets/scenes/<scene-id>/mesh.glb
viewer/public/assets/scenes/<scene-id>/walkable.json
viewer/public/assets/characters/<char-id>.glb
viewer/public/assets/portraits/<char-id>.png
```

Refresh `http://localhost:5173` — the viewer will pick up the new assets automatically. Scenes that don't yet have a `mesh.glb` fall back to a placeholder ground plane so the rest of the game still loads.

## Editing the story

Everything is in `viewer/public/manifest.json`:

- **scenes** — each has a prompt, seed, camera framing, lighting setup, NPC list, and exits to other scenes
- **characters** — appearance prompt (used by `gen_character.py`), display name, portrait, model
- **startScene / startSpawn** — where the player begins

Edit prompts, re-run the pipeline for that scene, refresh.

## Quest 3 deployment

For development with auto-reload from your PC:
1. `npm start` on your dev machine, find its LAN IP (e.g., `192.168.1.42`)
2. On Quest 3, open Meta Browser, go to `http://192.168.1.42:5173`
3. Click **ENTER VR** for immersive mode or **ENTER PASSTHROUGH** for AR

For a "production" build, host the `viewer/public/` directory on any HTTPS static host (Cloudflare Pages, Netlify, GitHub Pages). WebXR requires HTTPS off-localhost.

## Notes & limitations

- **Scene-to-3D quality** varies by source image complexity. Best results: clear depth cues, single-vantage-point composition, not too much occlusion.
- **Walkable mask quality** depends on Depth-Anything's depth output. Reflective floors, glass, and water can confuse it. Manual touch-ups to `walkable.json` are sometimes worth doing.
- **Character consistency** is best when each character has a stable front-view reference image. The pipeline uses IP-Adapter for the back-view, then Hunyuan3D 2 multi-view for the mesh. For very stylized characters, training a small LoRA for them first will improve consistency further.
- **WebXR on Quest** uses Meta Browser. Hand tracking works but the controls in this build are thumbstick-only by default.

## Folder layout

```
DownToEarth/
├── README.md                    (this file)
├── pipeline/                    (Python content-gen scripts)
│   ├── gen_scene.py
│   ├── scene_to_3d.py
│   ├── extract_walkable.py
│   ├── gen_character.py
│   ├── run_all.py
│   ├── requirements.txt
│   └── workflows/               (ComfyUI workflow JSON templates)
├── assets-raw/                  (intermediate generated assets, not deployed)
└── viewer/
    ├── server.js                (tiny static HTTP server)
    ├── package.json
    └── public/
        ├── index.html
        ├── manifest.json        (game scene graph + character defs)
        ├── js/
        │   ├── app.js           (entry point)
        │   ├── scene-loader.js  (loads scene meshes + actors)
        │   ├── walker.js        (movement, walkable polygon, interaction)
        │   └── dialog.js        (dialog popup)
        └── assets/
            ├── scenes/<id>/     (mesh.glb + walkable.json per scene)
            ├── characters/      (per-character GLBs)
            └── portraits/       (per-character portrait PNGs)
```
