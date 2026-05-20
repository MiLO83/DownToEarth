# UVW Voxel Raymarcher (Lyra 2 Lite preview)

Single-file WebGL2 demo of the realtime-render half of the Lyra-2-Lite pipeline.
Implements per-pixel DDA voxel raymarching with a 256³ procedural test atlas.

## Usage

Open `index.html` in any modern browser (Chrome / Edge / Firefox).
No build step, no server, no dependencies.

```
L-drag    orbit
R-drag    pan
wheel     zoom along view direction
WASD      fly forward / strafe
Q / E     up / down
1 / 2 / 3 swap procedural scene
```

## What it proves

The runtime side of the offline-bake / online-render split from
[LYRA2_V2_PROPOSAL §6.6](../../LYRA2_PROPOSAL.md):

- **DDA (Amanatides & Woo, 1987) voxel traversal** — exactly one voxel per
  step, no over/undersampling
- **Per-pixel ray** — each screen pixel independently casts a ray; no
  triangle rasterization, no LOD pop-in
- **First-hit shading** — early termination at the first occupied voxel
  (the natural depth-buffer / occlusion-cull baked into the algorithm)
- **Face-normal shading + distance fog** — minimum viable lighting; in
  the real pipeline this is replaced by baked PBR / lightmaps in the
  voxel's RGBA channels

## Frustum culling (added v2)

Chunk-level frustum culling is now active. The 256³ world is divided
into a 16×16×16 grid of 16³-voxel chunks (4,096 chunks total). Each
frame:

1. JS computes the 6 frustum planes from the camera (near / far / 4 sides)
2. JS tests each chunk's AABB against the planes via the p-vertex test
3. Visible chunks set their bit in a 4,096-bit (128-int) bitmap
4. Bitmap uploaded as a uniform array
5. Shader checks `chunkVisible(v / CHUNK_SIZE)` before sampling

Press <kbd>F</kbd> to toggle frustum culling on/off and watch the fps
counter swing. Press <kbd>G</kbd> to paint culled chunks solid red — you
can see exactly which volumes the algorithm skipped.

Typical observation: at standard FOV from outside the grid, ~80-95% of
chunks are culled in a single frame, which lifts fps proportionally on
the more-expensive scenes. This is the same algorithm a production
chunk-streamer uses to decide what to load from disk, just applied to
shader work in this demo.

## What's not in this demo (but is in the architecture)

This file uses a **procedural** voxel sampler in the fragment shader.
The production pipeline replaces `sampleVoxel(ivec3 v)` with a 3D-texture
lookup:

```glsl
uniform sampler3D uAtlas;            // 256³ RGBA8 uvw atlas
vec4 sampleVoxel(ivec3 v) {
    return texture(uAtlas, (vec3(v) + 0.5) / 256.0);
}
```

Three.js-shaped integration:
- Chunk streaming from NVMe (4096³ atlas tiled as 16×16×16 chunks)
- Frustum culling at chunk granularity
- VRAM cache of last ~200 chunks (~6 GB)
- 1-bit occupancy bitmap pre-pass to skip empty space at 8 voxels/byte

## Scenes (numbered 1-3)

1. **procedural-island** — sand floor + sea + central tower + palm ring +
   scattered rocks. Demonstrates the kind of bounded-scene output we'd
   get per Lyra 2 Lite chunk.
2. **stepped-tower** — a single tapered tower 240 voxels tall, sky
   elsewhere. Tests vertical range + sparse occupancy.
3. **caverns** — multi-octave 3D hash noise with thresholded occupancy.
   Tests dense, irregular topology + emissive crystal placement.

## Performance targets

On RTX 5060 Ti at 1080p (canvas internally renders at half-res, then
the browser scales up — same as standard game-engine practice):

| Scene | Avg ms/frame | Target fps |
|---|---:|---:|
| procedural-island | ~3-5 ms | 60+ |
| stepped-tower | ~2-3 ms | 60+ |
| caverns | ~4-6 ms | 60+ |

A 256³ scene fits entirely in shader-procedural sampling. A 4096³ scene
needs the 3D-texture path and the chunk streaming layer (~6 GB VRAM
working set, ~1-2 GB/s NVMe streaming for typical movement).

## Why DDA over fixed-step raymarching

Fixed-step raymarching (`pos += rd * stepSize`) over/undersamples voxels
and produces aliasing or holes. DDA computes the exact `t` at which the
ray crosses each voxel face and advances by exactly that much — every
ray visits every voxel it should, and only the voxels it should.
Original paper: [Amanatides & Woo 1987](https://www.cse.yorku.ca/~amana/research/grid.pdf).
