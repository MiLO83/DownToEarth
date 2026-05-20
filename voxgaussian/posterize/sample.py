"""
Multi-step sampler: take a heavily-posterized image and walk it up the
ladder rung by rung, each step bounded so structure cannot drift.

This is the inference-time counterpart to deposterizer.py. The model
predicts a signed delta; we add it to the current state and clip to the
+/- bound for the rung we're refining out of. Each step has a tiny
per-pixel search space, so the model never gets to decide "wall vs sky"
on a long sampling chain.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from posterize import rgb_to_luma_709, posterize_luma, posterize_rgb, ladder_to_grid
from noise_schedule import HALF_BIN, normalized_t, clip_to_bound
from deposterizer import TinyUNet


@torch.no_grad()
def sample_one(model: TinyUNet, start_state: np.ndarray, start_rung: int = 1,
               device: str = "cuda", enforce_bound: bool = True,
               save_intermediates: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Walk start_state (rung=start_rung) up the ladder to rung 8.

    start_state may be HxW uint8 (luma) or HxWx3 uint8 (RGB) -- the
    model's channel count determines which it expects.
    """
    model.eval()
    state = start_state.copy()
    intermediates = [state.copy()] if save_intermediates else []
    is_rgb = (state.ndim == 3)

    for rung_in in range(start_rung, 8):
        bound = HALF_BIN[rung_in]
        if is_rgb:
            x = torch.from_numpy(state.astype(np.float32) / 255.0).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)
        else:
            x = torch.from_numpy(state.astype(np.float32) / 255.0)
            x = x.unsqueeze(0).unsqueeze(0).to(device)
        t = torch.tensor([normalized_t(rung_in)], device=device, dtype=torch.float32)
        delta = model(x, t).squeeze(0).cpu().numpy()  # (C, H, W) or (H, W)
        if is_rgb:
            delta = delta.transpose(1, 2, 0)  # (H, W, 3)
        prediction = state.astype(np.float32) + delta * 255.0
        if enforce_bound:
            state = clip_to_bound(prediction, state, bound)
        else:
            state = np.clip(prediction, 0, 255).astype(np.uint8)
        if save_intermediates:
            intermediates.append(state.copy())
    return state, intermediates


def baseline_naive_upscale(start_luma: np.ndarray, start_rung: int) -> np.ndarray:
    """
    Trivial baseline: just leave the posterized image as-is at full
    8-bit precision. Lets us compare model output to "do nothing".
    """
    return start_luma.copy()


def main():
    ap = argparse.ArgumentParser(description="Sample with the trained deposterizer.")
    ap.add_argument("--ckpt", type=Path, required=True, help="path to deposterizer_*.pt")
    ap.add_argument("--image", type=Path, required=True, help="input image (any RGB)")
    ap.add_argument("--start-rung", type=int, default=1, help="BPP rung to start from (1..7)")
    ap.add_argument("--out", type=Path, default=Path("./sample_out"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bound", action="store_true", help="disable bound enforcement")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")
    rgb_clean = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    channels = ckpt.get("channels", 3)
    base_ch = ckpt.get("base_ch", 32)
    model = TinyUNet(base_ch=base_ch, channels=channels).to(args.device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt}  base_ch={base_ch} channels={channels}  step={ckpt.get('step', '?')}  loss={ckpt.get('loss', '?')}")

    if channels == 3:
        start_state = posterize_rgb(rgb_clean, args.start_rung)
        target = rgb_clean
    else:
        target = rgb_to_luma_709(rgb_clean)
        start_state, _ = posterize_luma(target, args.start_rung)

    final, inter = sample_one(model, start_state, start_rung=args.start_rung,
                              device=args.device, enforce_bound=not args.no_bound,
                              save_intermediates=True)

    def save(name, img):
        if img.ndim == 3:
            cv2.imwrite(str(args.out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(args.out / name), img)

    save("01_start_posterized.png", start_state)
    save("02_target_clean.png", target)
    save("03_model_final.png", final)
    for i, im in enumerate(inter):
        save(f"step_{i:02d}.png", im)

    strip = np.concatenate(inter, axis=1)
    save("00_progression_strip.png", strip)

    err_naive = float(np.mean(np.abs(start_state.astype(np.float32) - target.astype(np.float32))))
    err_model = float(np.mean(np.abs(final.astype(np.float32) - target.astype(np.float32))))
    print(f"L1 vs clean target:")
    print(f"  posterized start (rung {args.start_rung}): {err_naive:6.2f}")
    print(f"  model final (rung 8)                     : {err_model:6.2f}")
    if err_naive > 0:
        print(f"  improvement                              : {err_naive - err_model:+6.2f}  ({100*(1-err_model/err_naive):+.1f}%)")
    print(f"wrote {len(inter)} intermediate steps + progression strip -> {args.out}")


if __name__ == "__main__":
    main()
