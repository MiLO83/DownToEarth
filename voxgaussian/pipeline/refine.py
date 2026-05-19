"""
refine.py — Main refinement loop. Ties everything together.

  bootstrap → iterate(view-select → render → inpaint → propagate → check) → texture pass

After each iteration, broadcasts the voxel snapshot to the live viewer so you
can watch the topology fill in over time. Stops when:
  - Global voxel state change rate falls below `tolerance` (and min iters met)
  - Or `max_iterations` reached (set to 0 for unlimited)
  - Or Ctrl-C — saves state before exiting
"""
from __future__ import annotations
import argparse
import json
import pathlib
import signal
import sys
import time
import numpy as np

# Allow running as either `python -m pipeline.refine` or `python refine.py`
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from pipeline.voxel_store import VoxelStore, CLASSES   # noqa: E402
    from pipeline.bootstrap import bootstrap_scene          # noqa: E402
    from pipeline.render_voxels import render, write_debug_pngs  # noqa: E402
    from pipeline.select_view import candidate_views, pick_next_view  # noqa: E402
    from pipeline.propagate import propagate                # noqa: E402
    from pipeline.inpaint_client import InpaintClient       # noqa: E402
    from pipeline.live_server import LiveServer             # noqa: E402
    from pipeline.texture_pass import texture_store, save_colored_snapshot  # noqa: E402
    from pipeline.gaussian_fit import fit_and_save          # noqa: E402
else:
    from .voxel_store import VoxelStore, CLASSES
    from .bootstrap import bootstrap_scene
    from .render_voxels import render, write_debug_pngs
    from .select_view import candidate_views, pick_next_view
    from .propagate import propagate
    from .inpaint_client import InpaintClient
    from .live_server import LiveServer
    from .texture_pass import texture_store, save_colored_snapshot
    from .gaussian_fit import fit_and_save


REPO = pathlib.Path(__file__).resolve().parent.parent.parent      # DownToEarth/


def voxel_state_signature(store: VoxelStore) -> dict[tuple, int]:
    """Frozen snapshot of (idx → mode_class) used to compute change-rate
    between iterations."""
    sig: dict[tuple, int] = {}
    for idx, cls, conf in store.occupied():
        if conf > 0.5:
            sig[idx] = cls
    return sig


def change_rate(prev: dict, curr: dict) -> float:
    """Fraction of voxels that flipped class between iterations."""
    keys = set(prev) | set(curr)
    if not keys:
        return 1.0
    flipped = sum(1 for k in keys if prev.get(k) != curr.get(k))
    return flipped / len(keys)


def run_refinement(
    scene_id: str,
    scene_image: pathlib.Path,
    mesh_path: pathlib.Path,
    *,
    extent: float = 4.0,
    resolution: int = 96,
    max_iterations: int = 20,           # 0 = run forever, stop on Ctrl-C or tolerance
    tolerance: float = 0.02,
    min_iterations: int = 4,            # don't stop on tolerance below this
    comfyui_url: str = "http://127.0.0.1:8188",
    snapshot_downsample: int = 2,
    no_inpaint: bool = False,
    bootstrap_only: bool = False,       # skip iteration loop entirely
    live: bool = True,
) -> VoxelStore:
    """Run the full refinement loop for a single scene.

    `no_inpaint=True` runs bootstrap + active-view selection + carving only,
    skipping the diffusion step. Useful for shaking out the pipeline before
    the inpaint workflow is wired.

    `max_iterations=0` runs indefinitely; stop with Ctrl-C and the final
    state is saved gracefully on the way out.
    """
    print(f"\n=== voxgaussian refine: {scene_id} ===\n")

    # Start the live server
    server = LiveServer() if live else None
    if server:
        server.start()
        time.sleep(0.5)   # give the HTTP server a moment

    # Bootstrap from existing assets
    store = bootstrap_scene(scene_id, scene_image, mesh_path,
                            extent=extent, resolution=resolution)
    store.iteration = 0
    history: list[dict] = []           # per-iteration metrics for HUD trend
    if server:
        snap = store.snapshot(downsample=snapshot_downsample)
        snap["history"] = history
        server.broadcast(snap)

    inpaint = InpaintClient(comfyui_url=comfyui_url)
    cams = candidate_views(extent=extent)
    used_views: set[int] = set()

    prev_sig = voxel_state_signature(store)

    runs_dir = REPO / "voxgaussian" / "runs" / scene_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Graceful stop: shared flag flipped by either Ctrl-C OR the viewer's
    # STOP REFINE button (delivered via the LiveServer's WebSocket).
    # The loop checks the flag at the end of every iteration and finishes
    # the in-flight Phase B (texture + gaussians) before exiting.
    stop_requested = {"flag": False}

    def _request_stop(reason: str) -> None:
        if not stop_requested["flag"]:
            print(f"\n[refine] stop requested ({reason}) — "
                  f"finishing current iteration then saving...")
        stop_requested["flag"] = True

    def _sigint(sig, frame):
        if stop_requested["flag"]:
            print("\n[refine] second Ctrl-C — exiting immediately")
            sys.exit(130)
        _request_stop("Ctrl-C")
    signal.signal(signal.SIGINT, _sigint)

    if server:
        def _on_ws_msg(data):
            if isinstance(data, dict) and data.get("type") == "stop":
                _request_stop("viewer STOP button")
        server.on_message = _on_ws_msg

    # Treat max_iterations=0 as "forever" — use a huge number internally
    effective_max = max_iterations if max_iterations > 0 else 10_000_000

    if bootstrap_only:
        print("\n[refine] bootstrap-only mode -- skipping iteration loop, "
              "shipping the mesh-derived state straight to texture+gaussians")
        effective_max = 0   # disables the loop entirely

    max_display = "inf" if max_iterations == 0 else str(max_iterations)
    for it in range(1, effective_max + 1):
        store.iteration = it
        t0 = time.time()
        print(f"\n--- iteration {it}/{max_display} ---")

        # 1. Pick the most informative camera angle
        cam_idx, camera = pick_next_view(store, cams, used_views)
        used_views.add(cam_idx)
        print(f"  pick view #{cam_idx}: pos={camera.position}")

        # 2. Render the current voxel state from that camera
        rendered = render(store, camera)
        n_unknown = int(rendered["unknown_mask"].sum())
        print(f"  rendered: {n_unknown}/{camera.width * camera.height} px unknown "
              f"({100 * n_unknown / (camera.width * camera.height):.1f}%)")
        if it <= 3:
            write_debug_pngs(rendered, str(runs_dir / f"iter_{it:03d}_render"))

        # 3. Inpaint depth + semantic for unknown regions
        if no_inpaint:
            print("  [no-inpaint mode] using rendered output unchanged")
            inpainted_depth = rendered["depth"]
            inpainted_sem = rendered["semantic"]
            inpainted_rgb = None
            # Where rendered was NaN, leave NaN (nothing to project)
        else:
            print("  inpainting via ComfyUI ...")
            inp = inpaint.inpaint(
                depth=rendered["depth"],
                semantic=rendered["semantic"],
                unknown_mask=rendered["unknown_mask"],
                canonical=rendered.get("canonical"),    # Lyra-2 geometry anchor
                input_image_path=scene_image,
                iteration=it,
            )
            inpainted_depth = inp["depth"]
            inpainted_sem = inp["semantic"]
            inpainted_rgb = inp.get("rgb")

            # Persist inpaint RGB + the camera that produced it.
            # gaussian_fit's multi-view blend reads these to colour each splat.
            if inpainted_rgb is not None:
                from PIL import Image as _PILImage
                view_dir = runs_dir / "views" / f"iter_{it:03d}"
                view_dir.mkdir(parents=True, exist_ok=True)
                rgb_byte = (np.clip(inpainted_rgb, 0.0, 1.0) * 255).astype(np.uint8)
                _PILImage.fromarray(rgb_byte).save(view_dir / "rgb.png")
                with open(view_dir / "camera.json", "w") as f:
                    json.dump({
                        "position": list(camera.position),
                        "look_at": list(camera.look_at),
                        "up": list(camera.up),
                        "fov_deg": camera.fov_deg,
                        "width": camera.width,
                        "height": camera.height,
                        "near": camera.near,
                        "far": camera.far,
                    }, f)

        # 4. Project the inpainted view back into voxel votes — but only at
        # pixels that were originally unknown (Lyra-2 fill-holes semantics).
        # Pixels the renderer already saw cleanly are NOT re-voted; their
        # existing histogram entries are kept intact across iterations.
        result = propagate(store, camera, inpainted_depth, inpainted_sem,
                           rgb=inpainted_rgb,
                           unknown_mask=rendered["unknown_mask"],
                           ray_carve=not no_inpaint)
        print(f"  propagate: voted={result['cells_voted']} carved={result['cells_carved']}"
              f" gated={result.get('cells_skipped_already_known', 0)}")

        # 5. Convergence check
        curr_sig = voxel_state_signature(store)
        cr = change_rate(prev_sig, curr_sig)
        prev_sig = curr_sig
        elapsed = time.time() - t0
        stats = store.stats()
        print(f"  change_rate={cr:.4f}  mean_conf={stats['mean_confidence']:.3f}  "
              f"converged={stats['convergence_pct']:.1f}%  "
              f"active={stats['active_voxels']}  elapsed={elapsed:.1f}s")

        # Record this iteration for the HUD trend chart
        history.append({
            "iter": it,
            "change_rate": round(cr, 4),
            "mean_confidence": stats["mean_confidence"],
            "convergence_pct": stats["convergence_pct"],
            "active_voxels": stats["active_voxels"],
            "elapsed_s": round(elapsed, 1),
        })

        if server:
            snap = store.snapshot(downsample=snapshot_downsample)
            snap["history"] = history[-50:]   # last 50 iterations for the chart
            server.broadcast(snap)

        # Per-iter dump for offline A/B against later iterations.
        # Downsampled (same as live viewer snapshot) — keeps each dump ~1-3 MB
        # at res 80, and slice_view.py can load these directly.
        iter_snap = store.snapshot(downsample=snapshot_downsample)
        iter_path = runs_dir / f"voxels_iter_{it:03d}.json"
        with open(iter_path, "w") as f:
            json.dump(iter_snap, f, separators=(",", ":"))

        # Graceful Ctrl-C
        if stop_requested["flag"]:
            print(f"\nstopped manually at iteration {it}")
            break

        # Tolerance-based convergence
        if cr < tolerance and it >= min_iterations:
            print(f"\nconverged at iteration {it} (change_rate {cr:.4f} < tolerance {tolerance})")
            break

    # Save raw geometry-only voxel state
    out_path = runs_dir / "final_voxels.json"
    store.to_json_file(str(out_path))
    print(f"\nOK: wrote {out_path}")

    # ─── Phase B (texture pass) ─────────────────────────────────────────
    print("\n--- texture pass (Phase B) ---")
    voxgaussian_root = pathlib.Path(__file__).resolve().parent.parent
    colors = texture_store(store, scene_id, voxgaussian_root)
    colored_path = runs_dir / "colored_voxels.json"
    save_colored_snapshot(store, colors, colored_path)

    # ─── Gaussian fitting ───────────────────────────────────────────────
    print("\n--- gaussian fitting ---")
    gauss_path = runs_dir / "gaussians.json"
    n_gauss = fit_and_save(store, colors, gauss_path, scene_id)
    print(f"OK: {n_gauss} gaussians ready")

    # Broadcast final colored + gaussian payload so viewer can switch modes
    if server:
        server.broadcast({
            "type": "phase_b_complete",
            "scene_id": scene_id,
            "colored_voxels_url": f"/runs/{scene_id}/colored_voxels.json",
            "gaussians_url": f"/runs/{scene_id}/gaussians.json",
            "n_voxels": len(store.cells),
            "n_gaussians": n_gauss,
        })

    # Keep server alive so viewer can still inspect after pipeline finishes
    if server:
        print("\nLive viewer will stay up. Ctrl-C to exit when done inspecting.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            server.shutdown()

    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hamlet-square",
                    help="Scene id under DownToEarth/")
    ap.add_argument("--max-iterations", type=int, default=1,
                    help="0 = run until tolerance hit, Ctrl-C, or viewer STOP. "
                         "Empirically iter 1 is the quality sweet spot with "
                         "the heavier vote weights — bootstrap + one corroborating "
                         "inpaint. iter 2+ thickens walls into mush as cross-view "
                         "depth disagreements pile in.")
    ap.add_argument("--continuous", action="store_true",
                    help="Shorthand for --max-iterations 0. Run perspective "
                         "iterations forever and self-improve until the STOP "
                         "REFINE button is clicked in the viewer (or Ctrl-C).")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--min-iterations", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=128,
                    help="Coarse voxel grid resolution (per axis). 128 = 6.25cm "
                         "cells over an 8m cube. Higher = finer geometry detail, "
                         "but quadratic growth in surface voxels.")
    ap.add_argument("--extent", type=float, default=4.0,
                    help="Scene half-width in meters")
    ap.add_argument("--no-inpaint", action="store_true",
                    help="Skip diffusion inpaint, just bootstrap + carve")
    ap.add_argument("--bootstrap-only", action="store_true",
                    help="Skip the iteration loop entirely. Ship the mesh-derived "
                         "bootstrap state straight to texture pass + gaussian fit. "
                         "Use when the clean partial geometry is preferable to "
                         "an inpaint-corroborated version.")
    ap.add_argument("--no-live", action="store_true",
                    help="Don't start the live WebSocket viewer server")
    ap.add_argument("--comfyui", default="http://127.0.0.1:8188")
    ap.add_argument("--snapshot-downsample", type=int, default=2,
                    help="Coarseness factor for the live-viewer snapshot. 1 = "
                         "full voxel grid (slower stream, finer cubes); 2 = "
                         "default 2× downsample; 4 = chunky-cube debug mode. "
                         "If iter 0 looks too coarse in the viewer, try --snapshot-downsample 1.")
    args = ap.parse_args()

    scene_image = REPO / "assets-raw" / args.scene / "scene.png"
    mesh_path = REPO / "assets-raw" / args.scene / "mesh.glb"
    if not scene_image.exists():
        sys.exit(f"scene image missing: {scene_image}")
    if not mesh_path.exists():
        sys.exit(f"mesh missing: {mesh_path}  (run scene_to_3d.py first)")

    max_iters = 0 if args.continuous else args.max_iterations
    run_refinement(
        scene_id=args.scene,
        scene_image=scene_image,
        mesh_path=mesh_path,
        extent=args.extent,
        resolution=args.resolution,
        max_iterations=max_iters,
        tolerance=args.tolerance,
        min_iterations=args.min_iterations,
        comfyui_url=args.comfyui,
        snapshot_downsample=args.snapshot_downsample,
        no_inpaint=args.no_inpaint,
        bootstrap_only=args.bootstrap_only,
        live=not args.no_live,
    )


if __name__ == "__main__":
    main()
