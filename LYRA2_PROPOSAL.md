# Proposal: World-coord-keyed canonical encoding via a bidirectional UVW↔RGB atlas

**Targeted at:** [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra), Lyra 2.0
**Author:** MiLO + Opie ([github.com/MiLO83/DownToEarth](https://github.com/MiLO83/DownToEarth))
**Status:** Architectural proposal — research / discussion, not a PR
**License of this document:** MIT. Code references in this doc are from the
Apache-2.0-licensed Lyra source tree.

---

## Summary

Lyra 2's canonical-coord conditioning is **image-space-and-frame-indexed**:
each pixel of the warped conditioning image encodes
`(u_normalized, v_normalized, frame_slot_idx)`. The cross-attention then
treats these as keys into a *frame-keyed* history of latents. This works,
but two consequences fall out:

1. **Multiple past frames seeing the same 3D point produce different
   canonical-coord values.** Geometric correspondence is implicit and has
   to be re-learned at every attention layer.
2. **History size grows with time, not scene complexity.** A frame that
   re-observes already-known geometry still occupies a full slot.

Lyra 2's `Sparse3DCache` already computes per-pixel world coordinates
(via `unproject_points` at
[`lyra2_model.py:2534`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L2534)),
so the world-coord information is present — it's simply not used as the
canonical-coord key.

This proposal: **encode the canonical-coord image as quantized world
coords (packed into RGB bytes via a bijective tile layout), and back
it with a world-coord-keyed atlas instead of a frame-keyed cache.** Same
warp machinery, different output encoding, different storage layout.
Below: the mechanics, the wins, the trade-offs, and an estimated
porting cost.

---

## The current Lyra 2 mechanism (with citations)

I'll keep this brief — the paper covers it well — but anchor each claim
to source.

### Canonical-coord construction

`_build_canonical_spatial_coords` ([`lyra2_model.py:1525-1545`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L1525)):

```python
xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
base_xy = torch.stack([xx, yy], dim=0)  # [2, H, W]
...
zs = torch.linspace(-1.0, 1.0, num_spatial_hist, device=device, dtype=dtype)
z = zs.view(num_spatial_hist, 1, 1, 1).expand(num_spatial_hist, 1, H, W)
coords = torch.cat([base_xy, z], dim=1)  # [N, 3, H, W]
```

So each pixel of slot `n` starts life as `(u_normalized, v_normalized, z_n)`
where `z_n` is the slot's position along `[-1, 1]`. These get
forward-warped via `forward_warp_multiframes`
([`forward_warp_utils_pytorch.py:57`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/datasets/forward_warp_utils_pytorch.py#L57))
through past camera poses + depths into the target view. The output of
the warp is the canonical-coord image fed to the DiT as conditioning.

### Storage and retrieval

`Sparse3DCache` ([`lyra2_model.py:2488`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L2488))
keeps:

```python
self._world_points: list[torch.Tensor] = []  # each: [B, H', W', 3]
self._latent_indices: list[int] = []
self._frame_ids: list[int] = []
self._depths: list[torch.Tensor] = []
self._w2cs: list[torch.Tensor] = []
self._Ks: list[torch.Tensor] = []
self._rgbs: dict[int, torch.Tensor] = {}
```

`add()` calls `unproject_points` to compute world coords per pixel
([line 2534](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L2534)),
then stores those alongside the depth/w2c/K and the RGB. **The world
coords are computed and stored.** Retrieval (`Sparse3DCache.retrieve`)
selects past frames by visibility overlap with the target view and
returns their latent indices for cross-attention.

So: **frame-keyed cache, image-space-plus-frame canonical-coords.**

---

## The proposed alternative

### Encoding change

Replace the `(u_norm, v_norm, frame_slot)` encoding with **quantized
world coords packed into RGB bytes**:

```python
# Quantize world coords to [0, 2^B - 1] per axis (B = bits per channel).
# For B = 8 (RGBA8): 256 levels per axis = 16.7M unique 3D positions in a
# bounded world. For B = 16 (RGBA16UI): 65,536 levels = 281T positions.
def world_to_canonical_rgb(world_xyz: Tensor, world_min: Tensor,
                           world_max: Tensor, bits: int = 8) -> Tensor:
    n = (1 << bits) - 1
    normed = (world_xyz - world_min) / (world_max - world_min)  # [0, 1]
    return (normed.clamp(0, 1) * n).to(torch.uint8 if bits == 8 else torch.uint16)
```

This is byte-perfect bijective: the bytes in the canonical image **are**
the world coords (modulo a known world_min/max transform). The current
warp machinery transports them correctly — `forward_warp_multiframes`
doesn't care whether `frame1` is RGB, normalized image coords, or
quantized world coords; it just resamples per the depth + camera transform.

### Storage change

Replace the frame-keyed `Sparse3DCache` with a **world-coord-keyed atlas**
addressed via a 2D tile layout (the bijection: 256³ → 4096² for a single-
texture-array fit, hierarchical sparse beyond that):

```python
# Pack (u, v, w) ∈ [0, 256)³ into a 2D atlas position via a 16×16 tile grid.
# Each tile is one full w-slice of size 256×256.
def voxel_to_atlas(u: int, v: int, w: int) -> tuple[int, int]:
    return ((w & 15) << 8) | u, ((w >> 4) << 8) | v
```

The atlas stores whatever Lyra 2 currently stores per past frame —
latent feature vectors, RGB, or both — but indexed by 3D position
instead of by `(frame_id, pixel_position)`. When the same 3D point
gets observed by multiple frames, the new write **strengthens or
replaces** the existing slot rather than adding a new frame entry.

This bijection is documented + verified in
[`voxgaussian/pipeline/uvw_atlas.py`](https://github.com/MiLO83/DownToEarth/blob/main/voxgaussian/pipeline/uvw_atlas.py)
with an exhaustive 16.7M-pair round-trip test. The identity mapping
(atlas-position ↔ canonical-pass RGB) is *mathematically inherited* and
costs zero VRAM — only the payload uses storage.

### What changes in the cross-attention

Currently cross-attention is over frame latents, keyed by the warped
`(u_norm, v_norm, frame_slot)`. With the proposal, cross-attention is
over the atlas, keyed by the warped `(world_x, world_y, world_z)`. The
attention math is unchanged; only the storage layout changes.

---

## Why this is interesting (the wins)

1. **Geometric correspondence becomes structural, not learned.** A 3D
   point seen by frames 3, 7, and 12 has *one* key — its world coord —
   instead of three different `(u, v, frame_slot)` triples that the
   model must learn to treat as equivalent. This should reduce the
   inductive load on the cross-attention.

2. **Memory scales with scene complexity, not time.** Re-observing
   known geometry reinforces existing atlas slots; it doesn't allocate
   new frame slots. For long camera trajectories that revisit already-
   explored regions (which Lyra 2's 90 m worlds do, by design), this
   bounds memory growth.

3. **O(1) "is this 3D point known?" lookup.** Currently
   `Sparse3DCache.retrieve` scores all candidate frames by visibility
   overlap (see
   [`lyra2_model.py:2737-2832`](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L2737)).
   With a world-coord atlas, the question becomes a texture sample at
   the warped coord — single instruction.

4. **Spatial locality on GPU.** A tile-layout (or Hilbert-curve)
   bijection preserves 3D-neighbour adjacency in the 2D atlas, so cache
   prefetch helps for the typical "raymarch-y" access pattern.

5. **Debuggable conditioning.** Visualizing the canonical-coord image
   shows you exactly what the model sees: pixel colors *are* coords. No
   inverse-projection step to interpret it.

---

## Trade-offs (the catches)

1. **Bounded vs. unbounded scenes.** 256³ fits in 4096² (16.7 M atlas
   slots, ≈ 2.5 m world at 1 cm cells). Lyra 2's 90 m+ worlds need a
   hierarchical sparse-tiled version — atlas grows octree-style as the
   camera moves into new regions. Doable but more engineering. RGBA16
   pushes the bounded reach to ≈ 655 m at 1 cm cells.

2. **Retraining required.** The model has learned to read
   `(u, v, frame_slot)`. Adopting world-coord encoding changes its
   inputs — at minimum the conditioning head needs fine-tuning, more
   likely full retraining. Cost-prohibitive to attempt this without
   NVIDIA's training infrastructure.

3. **Loss of temporal information.** Frame index carries "when did we
   see this" signal that can be useful for weighting recent observations
   higher (drift correction). To preserve this in the atlas-keyed
   scheme, attach a `last_updated_iter` byte to each atlas slot — uses
   one of the 4 RGBA bytes the address doesn't need for spatial
   identity.

4. **Quantization error.** World coords get bucketed into the atlas
   resolution. At 8-bit per axis (256³), a 2.5 m world has 1 cm cells —
   probably fine for the kind of geometry Lyra 2 works with, but
   noticeable for very small detail. RGBA16 fixes this entirely.

5. **Doesn't help the unbounded-walk case directly.** Lyra 2's spec is
   *autoregressive long-horizon generation*. For each new chunk, the
   camera is somewhere new and we need to extend the world. A pure
   bounded atlas saturates; a hierarchical one is required for the full
   spec to be preserved.

---

## Performance cross-reference (before vs. after)

All numbers calibrated to Lyra 2's documented config: **832 × 480
resolution, 80 frames per chunk, num_spatial_hist = 5, Sparse3DCache
downsample = 4**. ✶ = calculated from the codebase. † = estimated.
‡ = qualitative.

| Dimension | Current (Lyra 2 frame-keyed) | Proposed (world-coord atlas) | Delta |
|---|---|---|---|
| **Canonical-coord image size, per chunk** | 5 × 3 × 480 × 832 × fp16 = **11.4 MB** ✶ | 5 × 3 × 480 × 832 × uint8 = **1.14 MB** (RGBA8) / 2.28 MB (RGBA16) ✶ | **10× smaller** (RGBA8) / 5× (RGBA16) |
| **Cache memory: `_world_points`, after 80 frames** | 80 × (480/4) × (832/4) × 3 × fp32 = **24 MB** ✶ | 256³ atlas × 4 B (RGBA8) = **64 MB** fixed (1024³ → ~4 GB texture array) † | Larger up front, **flat with time** vs. linear |
| **Cache memory: with `store_values=True` (full depth + w2c + K + RGB latents)** | 80 × (480 × 832 × fp32 depth + intrinsics + latents) ≈ **130 MB + frame latents** ✶ | Atlas + per-slot payload bytes; no per-frame redundant storage. ~64 MB to ~1 GB scene-dependent. † | Scene-complexity-bound, not time-bound |
| **Memory growth on revisits** (camera re-enters region already seen) | Linear: every frame adds a slot whether new or not | Flat: same world-coord slot gets reinforced, no new allocation | **O(time) → O(scene)** |
| **`Sparse3DCache.retrieve()` cost per target view** | Scores all N candidate slots by visibility overlap; iterate + sort | Single 4-instruction warp + texture sample per target pixel | **O(N) → O(1)** per pixel |
| **Cross-attention KV count** | num_spatial_hist (5) × tokens-per-frame | num_spatial_hist (5) × tokens-per-frame (unchanged — same model arch) | Same |
| **Distinct keys per 3D point across views** | One per past-frame-that-saw-it; the model must learn equivalence | **Exactly one** (the quantized world coord) | Inductive load ↓ |
| **Cache write atomicity (multi-view fusion)** | Sequential per-frame appends; merge logic in `retrieve()` | Atomic increment / max into atlas slots; GPU `atomicAdd` on `R32_UINT` | Trivially parallel |
| **GPU cache locality (spatial neighbours)** | Random — frames stored independently; access pattern depends on the warp permutation | Preserved — 16×16 tile layout keeps 3D neighbours within ~256 px in the 2D atlas | Texture-cache friendly |
| **Quantization error** | None (fp32 world coords retained internally) | **1 LSB per axis**: 1 cm at 256³/2.5 m world (RGBA8), <0.1 mm at RGBA16. None at RGBA32 ✶ | RGBA8: low / RGBA16: negligible |
| **Disocclusion-hole rendering quality** | Cleanly black where no past frame saw the pixel | Same — atlas lookup returns (0,0,0) where no slot has been written | Identical |
| **Long-horizon (Lyra 2 spec, 90 m walk-through)** | Frame count grows linearly with trajectory length; retrieval slows | **Requires hierarchical sparse atlas** — bounded variant saturates beyond world_max ‡ | Open engineering question, advantage *if* sparse-tiled is built |
| **Conditioning interpretability (visualize the canonical image)** | Hard to read (normalized image-coords + frame idx) | RGB literally encodes (u, v, w) — visual debug is the bytes ‡ | Strict win for development |
| **Code surface area (LoC change)** | — | `_build_canonical_spatial_coords` rewrite (~50 LoC) + `Sparse3DCache` dual-key mode (~100 LoC) + new `uvw_atlas.py` (~150 LoC). **~300 LoC net add.** † | Small |
| **Training-compat impact** | — | **Full retrain or fine-tune the conditioning head**: model expects different statistics on the input channels. ‡ | Significant — needs NVIDIA training infra |
| **Inference latency (per 80-frame step)** | ~194 s on GB200 (full), ~15 s (DMD-distilled) ✶ (from paper) | Estimate: ~1–3 % reduction from cheaper canonical-coord pass + O(1) retrieval; possible 5–10 % reduction at long-horizon revisits where current retrieve() dominates ‡ | Modest, possibly meaningful at scale |

### What this table is *not*

- **Not a quality-score comparison.** SSIM, LPIPS, FID — none of these can be predicted from the architecture change alone. They require a retrain to measure. The above is purely the *computational* and *memory* analysis.
- **Not a benchmark.** Numbers above are calculated or estimated from the codebase + the paper. No actual runs were performed (the proposers don't have GB200s).
- **Not a free win.** The retraining cost is the elephant in the room. The argument is the per-inference math is cleaner and the memory growth on long trajectories is bounded — *if* a retrain is in the budget anyway, this is the kind of architectural shift to fold in.

---

## Concrete porting sketch (if anyone wants to try)

Three files change, roughly:

### 1. `lyra_2/_src/models/lyra2_model.py::_build_canonical_spatial_coords`

```python
def _build_canonical_spatial_coords(H, W, num_spatial_hist, device, dtype, *,
                                    world_min, world_max, bits=8):
    # Generate a per-pixel world-coord initialization that the warp will
    # then transport. Each spatial slot still distinguishes via the 3rd
    # axis — but the axis is "depth tag" not "frame slot" so warps from
    # different past frames that landed on the same 3D point produce the
    # same canonical RGB.
    xs = torch.linspace(world_min[0], world_max[0], W, ...)
    ys = torch.linspace(world_min[1], world_max[1], H, ...)
    ...
    return coords  # [N, 3, H, W] in WORLD units (will be re-quantized post-warp)
```

After `forward_warp_multiframes` runs, quantize the warped output to
`uint8 / uint16` via `world_to_canonical_rgb`. The downstream
`_pixelshuffle_hw_to_latent` ([line 1568](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/lyra_2/_src/models/lyra2_model.py#L1568))
is dtype-agnostic.

### 2. `lyra_2/_src/models/lyra2_model.py::Sparse3DCache`

Add a world-coord-keyed mode alongside the existing frame-keyed one. A
single `dict[atlas_pos, payload]` (or a sparse tensor) keyed by the
2D atlas position. Existing `add()` writes to *both* (frame-keyed *and*
atlas-keyed) during a transition period; `retrieve()` gains an
`use_atlas: bool` flag.

### 3. New `lyra_2/_src/datasets/uvw_atlas.py`

The bijection helpers. Three functions:

```python
def world_to_atlas(world_xyz, world_min, world_max, bits=8):
    """World 3-vec → (atlas_x, atlas_y) 2-vec."""

def atlas_to_world(atlas_xy, world_min, world_max, bits=8):
    """Inverse: (atlas_x, atlas_y) → world 3-vec."""

def quantize_world_to_rgb(world_xyz, world_min, world_max, bits=8):
    """World 3-vec → packed (R, G, B) byte triple. Color IS coord."""
```

All pure-arithmetic, no learnable parameters. We've open-sourced this
exact module (MIT) at
[`voxgaussian/pipeline/uvw_atlas.py`](https://github.com/MiLO83/DownToEarth/blob/main/voxgaussian/pipeline/uvw_atlas.py)
— Lyra-friendly to copy/adapt under Apache-2.0 compatibility.

---

## What this doesn't claim

- **Quality wins.** I don't have the compute to retrain Lyra 2 and
  measure SSIM / LPIPS / FID. The argument is architectural: fewer
  things for the cross-attention to learn, smaller memory footprint
  for revisiting scenes. Whether that translates to measurable quality
  improvement is an open empirical question, and entirely depends on
  retraining infrastructure NVIDIA has and others don't.

- **A drop-in PR.** A retrain is required. This is a proposal to
  *consider* the encoding shift in a future training run, not a patch.

- **Replacing Sparse3DCache.** The frame-keyed cache has legitimate
  uses (temporal weighting, multi-view RGB storage). The world-coord
  atlas could complement it rather than replace it.

---

## Open questions for the team

1. Has the team explored world-coord-keyed conditioning during Lyra 2's
   design? If yes, what failed?
2. The current `_world_points` storage is already world-coord per
   pixel — at what cost would the cross-attention be restructured to
   query that directly?
3. For the long-horizon unbounded case, is a hierarchical sparse atlas
   competitive with the current frame-keyed approach in terms of
   memory growth as the camera trajectory length grows?
4. The 8-bit quantization is the simplest version; 16-bit and 32-bit
   variants are documented in
   [DownToEarth/README.md → Scaling section](https://github.com/MiLO83/DownToEarth#scaling--what-extra-bytes-per-channel-buy-you).
   Which precision tier would matter for Lyra 2's training data
   distribution?

---

## About the proposers

This grew out of building a Lyra-2-inspired local pipeline
([voxgaussian](https://github.com/MiLO83/DownToEarth/tree/main/voxgaussian))
that runs on consumer GPUs (single iter, ≈ 5 min from a single
Juggernaut XL render to a 167 k-Gaussian splat cloud). We can't match
Lyra 2's quality — the gap is the 14 B-param Wan-2.1 backbone trained
on 64 × GB200s, not the architecture — but the UVW↔RGB bijection
landed as a clean little data structure with one property that struck
us as worth surfacing upstream: **the bytes are the coords**, both
ways, at zero VRAM, and that property is rare enough to be useful
beyond our tiny project.

Happy to discuss, refine, or retract any of the above. If the encoding
shift turns out to be a bad fit, the comparison itself is still useful
data for anyone considering similar architectures.
