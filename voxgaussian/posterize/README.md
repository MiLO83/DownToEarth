# Progressive bit-depth deposterizer

Pixel-space implementation of [LYRA2_PROPOSAL.md §6.6.2](../../LYRA2_PROPOSAL.md). Builds the bit-depth ladder dataset, trains a small UNet that walks rung-to-rung, and samples by tightening the per-pixel bound at each step.

Designed to slot in as the **post-VAE pixel-refinement stage** on top of the fp8 Lyra 2 backbone — see "Combining with fp8 Lyra 2" below.

## Pipeline

```
1. scrape -> per-image bit-depth ladder (8 BPP -> 1 BPP, luma only)
2. train deposterizer (tiny UNet, ~1M params)
   one step: rung_in -> rung_(in+1)  with bounded delta
3. sample: start at rung 1, walk rung -> rung -> rung -> 8
   each step's delta is clipped to +/- half-bin-width
```

## Files

| File | Role |
|------|------|
| `posterize.py` | Core: `posterize_luma_ladder(rgb)` returns all 8 rungs |
| `scrape_dataset.py` | Parallel-fetch ladder dataset (picsum / URLs / local dir) |
| `realtime_posterize.py` | Live webcam / video preview with grid + per-rung focus |
| `noise_schedule.py` | Bounded-noise primitives + bin widths per rung |
| `deposterizer.py` | TinyUNet (~0.8M params) + dataset + training loop |
| `sample.py` | Multi-step inference: rung 1 -> rung 8 with bounded clipping |

## Speed numbers (verified)

**Posterize (LUT-based):**

| Resolution | Exact (Rec.709 + indices) | Fast (Rec.601, display only) |
|-----------|--------------------------:|------------------------------:|
| 720p | 80 fps | 326 fps |
| 1080p | 36 fps | 168 fps |
| 4K | 11 fps | 66 fps |

**Scrape (16 workers, picsum, 1024×448):** ~23 img/s sustained. 2000 images in 88s.

**Train (RTX 5060 Ti, 0.84M params, batch 32, crop 256):** ~1.7 it/s at batch 32 (PNG-backed I/O bound; faster with NPZ cache). Loss 0.25 → 0.05 in first 30 steps on cinematic data.

## Training-time estimates

Per-step cost is dominated by PNG decode, not the model. NPZ-cached loader would 5-10× this.

| Goal | Images | Epochs | Iters | Wall time |
|------|-------:|-------:|------:|----------:|
| Smoke / verify learnable | 2000 | 5 | 300 | ~3 min |
| First POC (loss < 0.02) | 2000 | 50 | 3 k | ~30 min |
| Visibly working sample | 5000 | 100 | 16 k | ~2 hr |
| Decent fidelity | 20 k | 200 | 125 k | ~12 hr |
| Production-grade | 50 k+ | 300+ | 500 k+ | 2-3 days |

The per-pixel decision space at each rung is tiny (3-5 discrete deltas), so the model converges *much* faster than full-RGB diffusion. The above are conservative.

## Quick start

```powershell
# 1. Scrape (already done if you have data/cinematic_2k/)
python scrape_dataset.py --count 2000 --width 1024 --height 448 --workers 16 --no-grid --out ./data/cinematic_2k

# 2. Train (smoke run -- 5 epochs)
python deposterizer.py --data ./data/cinematic_2k --out ./ckpts --epochs 5

# 3. Sample
python sample.py --ckpt ./ckpts/deposterizer_latest.pt --image any_photo.jpg --start-rung 2
```

## Bounded-noise schedule (what the model is actually doing)

Each rung-step has a per-pixel bound = half the bin width of the input rung:

| Input rung | Bin width | +/- bound per pixel |
|-----------:|----------:|--------------------:|
| 1 BPP (2 levels) | 255 | ±127 |
| 2 BPP (4 levels) | 85 | ±42 |
| 3 BPP (8 levels) | 36 | ±18 |
| 4 BPP (16 levels) | 17 | ±8 |
| 5 BPP (32 levels) | 8.2 | ±4 |
| 6 BPP (64 levels) | 4.0 | ±2 |
| 7 BPP (128 levels) | 2.0 | ±1 |

Rung 1 → 2: model has 255 luma values to pick from per pixel  
Rung 2 → 3: only 85 values per pixel (anchored to step-1 output)  
Rung 7 → 8: only 2 values per pixel — purely "should this pixel be the lower or upper half of its bin"

By the time we reach the last step the model literally cannot decide a wall is sky — only "shift this pixel by ±1." That's the bounded-deviation property: structure locks early, fine detail emerges late. The opposite of standard DDPM where structure can drift across the full sampling chain.

## Dataset layout

```
data/cinematic_2k/
  cine_002b469eef/
    rgb.png        # input  (NOT used in deposterizer training; reference only)
    luma.png       # Rec.709 luma (8-bit grayscale) -- effectively bpp8
    bpp1.png       # 1 BPP posterized (2 levels)
    bpp2.png       # 2 BPP (4 levels)
    ...
    bpp8.png       # 8 BPP (256 levels, same as luma.png)
  cine_003752fe42/
    ...
```

## Combining with fp8 Lyra 2

This module is designed to sit as the **post-VAE pixel-space refinement stage** on top of an fp8-quantized Lyra 2 backbone. Three plausible integration paths:

### A. Drop-in post-VAE head (what this is)
- fp8 Lyra 2 generates coarse latents → VAE decode → RGB at 832×480
- Posterize the VAE output to a lower rung (drops VAE high-frequency noise)
- This deposterizer walks it back up to 8 BPP + (future) chroma stage
- **Cost:** ~1M extra params, fits trivially in the 16 GB budget
- **Effort:** zero Lyra 2 retraining; just chain modules

### B. LoRA on the fp8 backbone (future)
- Bake bit-depth refinement into Lyra 2 itself via a LoRA adapter
- Repurpose Lyra 2's existing timestep conditioning: late timesteps = high BPP rungs
- Fine-tune the LoRA on (latent, rung) pairs
- **Cost:** 50-200M LoRA params; fp8-castable
- **Effort:** weeks of fine-tuning on a curated dataset

### C. Unified sibling model (research)
- Train a Lyra-2-shaped DiT from scratch with full position/luminance/bit-depth/chroma decomposition baked into the conditioning
- This is the actual §6.6.3 proposal — months of large-scale training compute

This repository builds path (A). The output of `sample.py` is what you'd hook directly into the Lyra 2 VAE-decoded RGB pipeline.

## Python API

```python
from posterize import posterize_luma_ladder
from deposterizer import TinyUNet
from sample import sample_one
from noise_schedule import HALF_BIN, normalized_t

# Generate ladder rungs from an RGB image
ladder = posterize_luma_ladder(rgb_uint8)

# Sample with a trained model
final_luma, intermediates = sample_one(model, start_luma_uint8, start_rung=2)
```
