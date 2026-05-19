# voxgaussian

**Diffusion-guided iterative voxel-occupancy + Gaussian-splat refinement** for turning a single AI-generated image into a walkable 3D scene.

This is the pipeline we designed over several iterations of chat. Each design decision is documented in code comments at the relevant file.

## The architecture in one paragraph

A sparse voxel grid in scene-space holds **per-voxel class-id histograms** — each cell remembers every vote it's ever received about what semantic class lives there (`ground`, `wall`, `tree`, etc.). The pipeline bootstraps from an existing Hunyuan3D mesh + the input image's semantic segmentation. It then iteratively: picks the camera angle that maximizes uncertain-voxel coverage in its frustum (active view selection), renders the current voxel state's depth + semantic from that angle, sends the unknown regions through a depth+semantic ControlNet inpaint (no color yet — that's Phase B), projects the inpainted pixels back into voxel histograms (vote), and ray-carves empty space along the camera ray. The mode-class-with-confidence per voxel is the "true" content; below-confidence voxels stay open for revision in later passes. Convergence triggers when global voxel state change-rate falls below tolerance. A WebXR live viewer subscribes to a WebSocket stream and shows the voxel grid filling in iteration-by-iteration.

The Gaussian Splatting phase (Phase B — appearance) takes over after geometry converges: project the input image's pixels onto the now-frozen voxel/mesh via the original camera, inpaint any unobserved regions with the geometry as a strong ControlNet constraint, bake into UV-mapped texture or fit Gaussians anchored to occupied voxels.

## Files

```
voxgaussian/
├── pipeline/
│   ├── voxel_store.py        Sparse histogram-per-voxel data structure
│   ├── bootstrap.py          Init from Hunyuan3D mesh + scene image segmentation
│   ├── render_voxels.py      Voxel → depth+semantic image rasterizer
│   ├── select_view.py        Active view selection by uncertainty
│   ├── inpaint_client.py     ComfyUI client for depth+semantic ControlNet inpaint
│   ├── propagate.py          Image → voxel histogram update + ray-carving
│   ├── refine.py             Main loop orchestrator
│   └── live_server.py        WebSocket bridge for the live viewer
├── viewer/public/            Three.js + WebXR voxel viewer
│   ├── index.html
│   └── js/app.js
└── runs/<scene-id>/          Per-scene refinement output (voxel JSON + debug PNGs)
```

## How to run

Prerequisite: you've already run the DownToEarth scene generation pipeline so `assets-raw/<scene-id>/scene.png` and `assets-raw/<scene-id>/mesh.glb` exist.

**1. Test without the diffusion inpaint** (verifies plumbing, no GPU required beyond Depth-Anything for bootstrap):

```powershell
cd C:\Users\rxcam\Documents\DownToEarth\voxgaussian
C:\Users\rxcam\ComfyUI_portable\ComfyUI_windows_portable\python_embeded\python.exe -m pipeline.refine --scene hamlet-square --no-inpaint
```

You should see the voxel store bootstrap (Hunyuan3D mesh + segmented input image projected), then 12 iterations of view-select → render → carve. Open `http://localhost:5174` to watch live.

**2. With diffusion inpaint** (full pipeline, requires the inpaint workflow):

Write the ComfyUI inpaint workflow at `workflows/depth_semantic_inpaint.json`. It needs:
- `LoadImage` nodes titled "depth", "semantic", "mask", "reference" — the pipeline patches these by title
- A `KSampler` (seed gets patched per-iteration)
- A `SaveImage` outputting **depth-as-grayscale-png FIRST**, then **semantic-as-palette-png SECOND**

The simplest backbone: SDXL inpaint + ControlNet-depth + ControlNet-segmentation, with the reference image fed via IP-Adapter for style anchoring.

```powershell
C:\Users\rxcam\ComfyUI_portable\ComfyUI_windows_portable\python_embeded\python.exe -m pipeline.refine --scene hamlet-square
```

**3. Watch live**

Open `http://localhost:5174` while the pipeline runs. Voxels color by class (legend top-right), brightness = mode confidence. On Quest 3, point Meta Browser at `http://<your-pc-ip>:5174` and tap the `ENTER VR` button.

## What's done in this scaffold

- ✅ Sparse histogram-per-voxel store with mode, confidence, convergence detection, ray-carving support, JSON snapshots
- ✅ Multi-resolution coarse/fine hierarchy (`MultiResolutionVoxels`)
- ✅ Bootstrap from existing Hunyuan3D mesh + OneFormer semantic seg (with heuristic fallback) + Depth-Anything projection
- ✅ Voxel → image rasterizer producing depth, semantic, confidence, unknown-mask
- ✅ Active view selection by frustum uncertainty mass
- ✅ Vote propagation + ray-carving with class-aware weights
- ✅ ComfyUI inpaint client (workflow JSON pluggable)
- ✅ Refinement loop with per-iteration convergence checking + WebSocket streaming
- ✅ WebXR live viewer (Three.js + InstancedMesh + OrbitControls + VR)

## What's intentionally stubbed for v1

- **Phase B (texture pass)** — geometry converges first; the texture stage is sketched in `refine.py` but not yet implemented. The natural next step: project input image pixels onto the converged voxels from the original camera, then UV-bake and inpaint missed regions (essentially Hunyuan3D's existing texture pipeline, applied to our voxel-derived mesh).
- **Gaussian splat fitting tied to voxel occupancy** — the design's punchline (Gaussians constrained by voxel topology) is one optimization loop away. The voxel store already exposes everything needed (`is_solid`, `is_walkable`, per-voxel confidence); the missing piece is a gsplat-based renderer that masks updates against the voxel mask.
- **The inpaint workflow JSON** — sketch a working depth+semantic ControlNet flow in ComfyUI's UI, export as API format, drop in at `workflows/depth_semantic_inpaint.json`. Without that, `--no-inpaint` mode still runs end-to-end.

## Design provenance

This whole architecture was designed in a long chat conversation. Each design decision compounded:

| Decision | What it bought us |
|---|---|
| Voxel occupancy as topology gate on Gaussians | Eliminates floaters; gives a *physical* constraint diffusion alone can't enforce |
| Per-voxel class HISTOGRAM (not single class) | Automatic outlier rejection; per-voxel confidence; cheap rollback |
| Bootstrap from Hunyuan3D mesh | Skip "cold start" iterations; start with reasonable topology |
| Multi-resolution coarse-fine | Compute scales with information density, not grid resolution |
| Active view selection by uncertainty | No wasted iterations on already-converged regions |
| Depth-first, texture-later phases | Geometry convergence isolated from color drift; fewer iterations to stabilize |

The original time estimate for a research-quality prototype was 14 days. The final post-design estimate was sub-1-day, because good design eats complexity. This scaffold represents that work.

## Dependencies installed in ComfyUI's embedded Python

- `websockets` (live server)
- `websocket-client` (ComfyUI client)
- Already there: `numpy`, `pillow`, `trimesh`, `torch`, `transformers`

## Live viewer keyboard

- `mouse drag` — orbit
- `scroll` — zoom
- `right-drag` — pan
- `ENTER VR` button — Quest 3 / WebXR-capable headset
