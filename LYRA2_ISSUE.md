<!--
  LYRA2_ISSUE.md — ready-to-paste body for filing on nv-tlabs/lyra.

  Suggested issue settings:
    Type:    Discussion (preferred — this is an RFC, not a bug) OR Issue
    Title:   "RFC: World-coord-keyed canonical encoding via bidirectional UVW↔RGB atlas (alternative to frame-keyed Sparse3DCache)"
    Labels:  enhancement, discussion, lyra-2 (apply whatever they use)

  Where to file:  https://github.com/nv-tlabs/lyra/discussions/new
                  or https://github.com/nv-tlabs/lyra/issues/new
-->

# RFC: World-coord-keyed canonical encoding for Lyra 2

Hi NVIDIA Toronto team — first, sincere thanks for releasing Lyra 1 + 2
under Apache 2.0 with this much detail. Studying the code has been a
treat.

This is a discussion-only proposal (not a PR — I can't retrain). Filing
in case the encoding shift below is interesting for a future training
run, or in case the team has already considered it and the reasons it
didn't make the cut would be useful to know.

**Full write-up with citations + porting sketch + trade-offs + a
before/after performance table:**
[github.com/MiLO83/DownToEarth/blob/main/LYRA2_PROPOSAL.md](https://github.com/MiLO83/DownToEarth/blob/main/LYRA2_PROPOSAL.md)

The short version below.

---

## In plain English

**Lyra 2 today:** keeps a photo album for every frame the camera ever
generated. To find "what's at this 3D point?" you flip through every
album asking "did *you* see this spot?"

**Proposed:** keep a single floor-plan map of the scene. Every spot on
the map has one unique address. No matter how many times the camera
re-observes that spot, it goes to one map cell. To find it, glance at
the map once.

Both approaches handle the same input + output. The map version trades
"unbounded but slow at long horizons" for "bounded but always O(1)."

---

## The architectural shift in two diagrams

**Current** ([`_build_canonical_spatial_coords`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L1525)
+ [`Sparse3DCache`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L2488)):

```
canonical-coord image pixel  →  (u_normalized, v_normalized, frame_slot)
                                          │
                                          ▼
                          attend over frame latents
                          (Sparse3DCache._world_points
                           keyed by frame_id; retrieve()
                           scores past frames by overlap)
```

**Proposed:**

```
canonical-coord image pixel  →  (world_x, world_y, world_z) quantized as RGB
                                          │
                                          ▼
                          attend over world-coord atlas
                          (single 2D atlas via 256³ ↔ 4096² bijection
                           OR sparse-tiled for unbounded scenes;
                           texture-sample lookup, no scoring loop)
```

The warp machinery in [`forward_warp_multiframes`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/datasets/forward_warp_utils_pytorch.py#L57)
transports either encoding correctly — it doesn't care whether the
input channels carry image-space or world-space coords.

The Lyra 2 codebase **already computes world coords per pixel** at
[`lyra2_model.py:2534`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L2534)
via `unproject_points` and stores them in `Sparse3DCache._world_points`.
The proposal is to use them as the canonical key, not just as cache
metadata.

---

## Wins (per the full write-up)

1. **One key per 3D point across all views** — geometric correspondence
   is structural, not learned. Multiple frames seeing the same point
   produce the same canonical RGB.
2. **Memory bounded by scene complexity, not trajectory length** —
   revisits reinforce existing atlas slots; they don't allocate new
   frame entries. For long-horizon revisiting trajectories this
   matters.
3. **O(1) retrieval per pixel** — texture sample replaces the
   visibility-scoring scan in
   [`Sparse3DCache.retrieve()`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L2737).
4. **Spatial locality preserved on GPU** — 16×16 tile layout puts 3D
   neighbours next to each other in the atlas, so texture cache helps
   for ray-marching access patterns.
5. **Conditioning becomes visually interpretable** — the RGB pixels are
   the coords; debugging is just looking at the image.

## Losses (also per the write-up)

1. **Quantization grid:** at 8-bit per axis, 1 cm cells over a 2.5 m
   world. At 16-bit, 65 535 levels per axis = 655 m world at 1 cm /
   65 m at 1 mm. At 32-bit, effectively unbounded. **RGBA16UI looks
   closest to Lyra 2's 90 m world spec.**
2. **Unbounded scenes need a hierarchical sparse-tiled variant** — a
   single bounded atlas saturates beyond `world_max`. Doable
   (octree-style page table) but non-trivial.
3. **Retraining required** — the conditioning head has learned the
   current `(u_norm, v_norm, frame_slot)` encoding. Adopting world
   coords requires fine-tune at minimum. This is the elephant-in-the-
   room and entirely your call.
4. **Loss of temporal info** — can be recovered by attaching a
   `last_updated` byte to each atlas slot.
5. **No benefit on single-frame use-cases** — wins all come from the
   re-observation loop, which is Lyra 2's actual workload, but worth
   being honest.

---

## Performance — headline numbers

Calibrated to your documented config (832 × 480, num_spatial_hist = 5,
Sparse3DCache downsample = 4):

| Dimension | Current (frame-keyed) | Proposed (world-coord atlas) |
|---|---|---|
| Canonical-coord image, per chunk | 5×3×480×832×fp16 = **11.4 MB** | RGBA8: **5.71 MB** / RGBA16UI: 11.4 MB |
| `_world_points` after 80 frames | **24 MB**, linear in trajectory length | 64 MB fixed (RGBA8 / 256³ atlas) |
| Memory growth on revisits | Linear (each frame = new slot) | **Flat** (slot reinforced in place) |
| `Sparse3DCache.retrieve()` cost | O(N candidate frames) | **O(1)** texture sample |
| Distinct keys per 3D point across views | One per past-frame-that-saw-it | **Exactly one** |
| Quantization error | None (fp32 internal) | 1 LSB per axis (1 cm at RGBA8, ≤ 0.1 mm at RGBA16) |

**Caveat:** none of these are measured. They're calculated from the
code-base and the paper. The numbers we can predict structurally are
solid; the open empirical question is whether the structural
simplifications translate to measurable quality wins (SSIM / LPIPS /
FID) without compute we don't have.

## Byte-length picker

For Lyra 2's actual 90 m world spec:

| Scene scale | Suggested format | Notes |
|---|---|---|
| ≤ 2.5 m | RGBA8 | Single-room |
| **2.5–655 m (Lyra 2 typical)** | **RGBA16UI** | Sub-cm precision over 655 m, byte-perfect coord identity, requires `EXT_color_buffer_integer` (already widely available) |
| 655 m – 167 km | RGBA32F | Mantissa-exact ≤ 2²⁴, lossy after |
| Unbounded | Hierarchical sparse + RGBA32UI | Octree of leaf-tile atlases, byte-identity within tiles |

---

## Open questions for the team

1. **Has the team explored world-coord-keyed conditioning during Lyra
   2's design?** If yes, what failed or made it not worth shipping?
2. The current `_world_points` storage is already world-coord per
   pixel — what is the cost (engineering + training) of restructuring
   the cross-attention to query them directly?
3. For the long-horizon 90 m+ unbounded case, **is a hierarchical
   sparse atlas competitive** in memory growth with the frame-keyed
   approach as trajectories get long?
4. **Which precision tier (RGBA8 / 16UI / 16F / 32F) would matter for
   Lyra 2's training data distribution?** RGBA16UI looks like the
   sweet spot from outside, but you'd know better.
5. **Is there interest in a co-authored experiment** if compute could
   be arranged? Happy to handle the encoding + atlas layer in any
   format that's useful.

---

## Background on the proposer

This grew out of building a Lyra-2-shaped local pipeline
([voxgaussian](https://github.com/MiLO83/DownToEarth/tree/main/voxgaussian))
that runs on consumer GPUs (Juggernaut XL inpaint backbone, 167k
Gaussian-splat output from a single 832×832 render in ~5 min,
[live demo](https://downtoearth-9lq.pages.dev)). I can't match Lyra 2's
quality — the gap is the 14 B-param Wan-2.1 backbone trained at scale
— but the UVW↔RGB bijection landed as a clean little data structure
with one property that seemed worth surfacing upstream: the bytes are
the coords, both ways, at zero VRAM for the identity mapping.

Happy to refine, narrow, or retract any of this. Even if the encoding
shift is a bad fit, "the team considered it and rejected because X"
would be useful documentation for anyone else looking at similar
architectures.

Thanks for reading. 🙏

cc: links to full doc + repos
- [Full proposal](https://github.com/MiLO83/DownToEarth/blob/main/LYRA2_PROPOSAL.md)
- [Bijection implementation, MIT](https://github.com/MiLO83/DownToEarth/blob/main/voxgaussian/pipeline/uvw_atlas.py)
- [Live demo](https://downtoearth-9lq.pages.dev)
