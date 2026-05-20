# Session TODO — Lyra 2 Lite acceleration stack

Started: 2026-05-19  ·  Last updated: 2026-05-20

Living document. Future-Opie: **read this on session resume, mark
completions inline (`✅`), promote items that are now ready, and update
the date stamp.** New work items append at the end of each section.

---

## ✅ Done

- ✅ Forked nv-tlabs/lyra → MiLO83/Lyra-2 (PR #61 filed, 6 commits)
- ✅ UVW↔RGB bijection design + atlas implementation
  (`voxgaussian/pipeline/uvw_atlas.py`)
- ✅ `OccupancyBitmap` class — 1-bit mode (occupancy)
- ✅ **`OccupancyBitmap` 2-bit mode** — occupancy + ray-touched bit for
  demand-driven streaming (2026-05-20)
- ✅ Streaming requantize tool for int8/int4/fp8 (`low_vram.py`)
- ✅ `apply_int8/int4/fp8_quantization` patched to handle CPU-source
  weights for the CPU-first low-vram path
- ✅ `model_loader.py` patched: CPU-first instantiate when
  `low_vram_mode != "none"`, `.cuda()` only after quantization
- ✅ HF download of `nvidia/Lyra-2.0` to F:\lyra2 (90 GB, complete)
- ✅ WSL Ubuntu 24.04 set up, milo user passwordless sudo, RAM bumped
  to 28 GB via `.wslconfig`
- ✅ bitsandbytes installed (Windows side)
- ✅ Cinematic dataset (2000 picsum images at 1024×448, all 8 bit-depth
  rungs, RGB + luma) baked at
  `voxgaussian/posterize/data/cinematic_2k`
- ✅ Bit-depth ladder posterizer + deposterizer (TinyUNet 0.84M params,
  RGB 3ch and luma 1ch modes); training pipeline verified loss
  0.247→0.050 in 30 steps on real cinematic data
- ✅ Bounded-noise schedule module (`noise_schedule.py`)
- ✅ Eval harness (`eval_all.py`) for held-out comparison
- ✅ RIFE 4.26 wired + verified — `lyra_pr_branch/rife_upsample.py`,
  9 sec to do 12 fps → 60 fps on 36-frame 832×480 clip on 5060 Ti
- ✅ Entity vocabulary + variable-byte priority encoding
  (`entity_vocab.py`, 74-entity seed)
- ✅ `StructuredPromptBuilder` + `lyra2_caption_hook` integration point
  (`structured_prompt.py`)
- ✅ WebGL2 voxel raymarcher demo with DDA + frustum cull
  (`voxgaussian/voxel_renderer/index.html`)
- ✅ SOTA review of acceleration stack (DMD / TeaCache / torch.compile /
  fp8 / SAGE / PyramidalWan / FastWan) — research agent output
  preserved in transcript
- ✅ README + LYRA2_PROPOSAL + DUMMIES updated with all new pieces
  including the TB-scale streaming-from-disk explanation (each in its
  own voice)
- ✅ Conversation transcript dumped to
  `DownToEarth/conversation_2026-05-20_full.log` (758 KB, 163 user
  turns / 378 assistant turns / 683 tool calls)

---

## ⏳ In flight / blocked on something external

- ⏳ **WSL Lyra 2 setup compile** — `setup_lyra2_wsl.sh` running in
  background; flash-attn 2.6.3 + transformer_engine builds are the
  long pole. Status: blocked on compile completion. Check
  `/home/milo/lyra2_setup.log` inside WSL for progress. When complete,
  next step is the verify-imports block at end of script.
- ⏳ **First Lyra 2 inference run on 5060 Ti** — gated on the above.
  Command lives in `lyra_pr_branch/Lyra-2/INSTALL.md` § Custom Trajectory.
  Use `--low-vram int8 --low-vram-checkpoint`.

---

## 🎯 Next up (tonight-feasible once setup completes)

1. **Bump flash-attn to ≥ 2.8.3** in the WSL env — required for
   torch.compile graph-fusion on Blackwell SM_103 without
   `flash_attn_2_cuda.PyCapsule.fwd` breaks. Setup script installs
   2.6.3 because that's what NVIDIA pinned. Post-install one-liner:
   `pip install --no-build-isolation --no-binary :all: flash-attn>=2.8.3`
2. **TeaCache integration** — clone ali-vilab/TeaCache, write
   `lyra2_teacache_wrapper.py` in `lyra_pr_branch/`. Real production
   code, 2-3× free speedup, Wan-2.1-compatible.
3. **torch.compile patch** for `lyra_2/_src/utils/model_loader.py` —
   add `model = torch.compile(model, mode='reduce-overhead')` after
   the `apply_low_vram_mode` call, gated on a new
   `low_vram_compile=True` flag. ~10 min change.
4. **End-to-end Lyra 2 Lite inference test** — example_0 trajectory,
   `--use_dmd`, the full stack stitched together. Measure actual
   per-chunk wall time on 5060 Ti and compare to the 60× projection.
5. **Verify deposterizer convergence** — restart the killed training
   run with smaller batch / faster I/O (consider pre-loading dataset
   to NPZ instead of decoding PNGs every step). Sample at multiple
   start rungs on held-out eval images.

---

## 🌐 Deployment-side (MiLO's action)

- ⬜ Run `.\sync-docs.ps1` in `DownToEarth/` to copy MDs → public/,
  regen PDFs via Edge headless, push to GitHub, deploy to Cloudflare
  Pages. Three docs are stale on the live site now (README +
  LYRA2_PROPOSAL + DUMMIES all gained ~330 lines tonight).

---

## 🛠 Engineering, no rush

- ⬜ Add `imageAtomicOr` shader wiring for the 2-bit OccupancyBitmap
  touched bit. The CPU side is shipped; the GLSL bind needs a
  runtime voxel-renderer integration to exercise.
- ⬜ Voxel renderer demo upgrade: real chunk-streaming demo with
  pre-baked 4096³ atlas (instead of procedural). Would need a
  Python-side atlas synthesizer + JS-side chunk loader.
- ⬜ Replace the deposterizer's PNG-backed dataloader with NPZ-backed
  (5-10× I/O speedup). The cinematic dataset is already on disk;
  add a `pack_to_npz.py` helper.
- ⬜ ScummC integration prototype — proof-of-concept that we can write
  a SCUMM v5 script via me + ScummC compiler and have it load into a
  ScummVM-derived Unity port. Per MiLO's retirement vision, this is
  the gameplay-injection layer.

---

## 🧪 Research / future direction

- ⬜ **LoRA on SD/Lyra2 for bit-depth ladder** —
  [[project_lora_sd_bitdepth_ladder]] memory. ~$1-5k cloud, weeks.
- ⬜ **PyramidalWan finetune** of Lyra 2 base — ~1 month + ~$1k cloud.
  Adds 2-3× on top of the 60× stack.
- ⬜ **Dynamic objects (moving entities)** — three sketched paths:
  hybrid voxel+mesh, dynamic voxel sub-atlases (Teardown pattern),
  or 4D Gaussian Splatting research. Not yet picked, not yet built.
  Replay system blocked on this.
- ⬜ **6DOF replay** — designed but blocked on dynamics. Once moving
  objects are in, this is ~30 min of plumbing (the
  `OccupancyBitmap.touched_chunks()` machinery + per-frame trajectory
  log is all that's needed).
- ⬜ **Equirectangular 360° video recording** — alternative to 6DOF,
  works in any 360 player. Independent of dynamics solution.
- ⬜ **Cross-scene semantic chunk caching** — content-addressable
  storage of generated chunks across scenes. TeaCache covers
  *within*-generation step skipping; this would be *between*-scene
  reuse. No production code exists; would need design + impl.

---

## 🤝 Coordination

- ⬜ Wait on NVIDIA's response to PR #61
- ⬜ If/when MiLO wants to share Lyra-2-Lite with NVIDIA for use
  permission, draft the email. See
  [[project_lora_sd_bitdepth_ladder]] and
  [[project_voxgaussian_posterize]] for the architectural shape.

---

## How to maintain this file

Future-Opie protocol:

1. On session resume, before doing other work, check this file for
   anything that may have completed since last session (e.g. the WSL
   setup may have finished; the deposterizer training may have run;
   MiLO may have deployed the docs).
2. Mark completions inline with `✅`. Move them to the "Done" section
   if you want to keep the active list short.
3. If you start a new task during the session, append it under the
   appropriate header with `⬜` and a one-line description.
4. End the session by updating the "Last updated" date at top.
5. **Do not silently delete items.** If something becomes irrelevant,
   move it to a "Cancelled" or "Won't do" section with a one-line
   reason.

The memory note at
[[session_todo_lyra2_lite]]
points future-Opie at this file on session start.
