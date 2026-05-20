"""
Run the trained deposterizer on every rgb.png in an eval directory and
build a master comparison grid + per-image progression strips.

Outputs:
    eval_out/
      <id>/
        start_r{R}.png        # posterized input at start rung
        final_r{R}.png        # model output (rung 8)
        progression_r{R}.png  # rung 1 -> ... -> rung 8 strip
      grid_start_r{R}.png     # all images stacked vertically at start_rung R
      grid_final_r{R}.png     # all images stacked: model output at start_rung R
      grid_target.png         # all clean targets
      L1_summary.txt          # per-image L1 numbers vs clean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from posterize import posterize_rgb, posterize_luma, rgb_to_luma_709
from deposterizer import TinyUNet
from sample import sample_one


def eval_one(model, rgb_clean, channels, start_rung, device):
    if channels == 3:
        start = posterize_rgb(rgb_clean, start_rung)
        target = rgb_clean
    else:
        target = rgb_to_luma_709(rgb_clean)
        start, _ = posterize_luma(target, start_rung)
    final, inter = sample_one(model, start, start_rung=start_rung,
                              device=device, save_intermediates=True)
    return start, final, inter, target


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="directory with <id>/rgb.png samples")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-rungs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    channels = ckpt.get("channels", 3)
    base_ch = ckpt.get("base_ch", 32)
    model = TinyUNet(base_ch=base_ch, channels=channels).to(args.device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt}  base_ch={base_ch} channels={channels}  "
          f"epoch={ckpt.get('epoch','?')}  loss={ckpt.get('loss','?')}")

    sample_dirs = sorted(p for p in args.eval_dir.iterdir() if p.is_dir())
    print(f"evaluating {len(sample_dirs)} samples at start_rungs={args.start_rungs}")

    log_lines = [
        f"# Evaluation report",
        f"",
        f"checkpoint: `{args.ckpt}`",
        f"  base_ch:  {base_ch}",
        f"  channels: {channels}",
        f"  epoch:    {ckpt.get('epoch','?')}",
        f"  step:     {ckpt.get('step','?')}",
        f"  loss:     {ckpt.get('loss','?')}",
        f"",
        f"## Per-image L1 (lower is better)",
        f"",
        f"| sample | start_rung | L1 posterized | L1 model | improvement |",
        f"|--------|-----------:|--------------:|---------:|------------:|",
    ]

    aggregate = {r: {"naive": [], "model": []} for r in args.start_rungs}

    for d in sample_dirs:
        rgb_path = d / "rgb.png"
        if not rgb_path.exists():
            continue
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb_clean = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out_sub = args.out / d.name
        out_sub.mkdir(parents=True, exist_ok=True)

        for r in args.start_rungs:
            start, final, inter, target = eval_one(model, rgb_clean, channels, r, args.device)
            err_naive = float(np.mean(np.abs(start.astype(np.float32) - target.astype(np.float32))))
            err_model = float(np.mean(np.abs(final.astype(np.float32) - target.astype(np.float32))))
            aggregate[r]["naive"].append(err_naive)
            aggregate[r]["model"].append(err_model)
            improve = f"{(err_naive - err_model):+.2f}" if err_naive > 0 else "n/a"
            log_lines.append(
                f"| {d.name} | {r} | {err_naive:.2f} | {err_model:.2f} | {improve} |"
            )
            cv2.imwrite(str(out_sub / f"start_r{r}.png"), to_bgr(start))
            cv2.imwrite(str(out_sub / f"final_r{r}.png"), to_bgr(final))
            strip = np.concatenate(inter, axis=1)
            cv2.imwrite(str(out_sub / f"progression_r{r}.png"), to_bgr(strip))
            cv2.imwrite(str(out_sub / f"target.png"), to_bgr(target))
            print(f"  {d.name} r={r}  naive={err_naive:5.2f}  model={err_model:5.2f}")

    log_lines += ["", "## Aggregate L1 across all eval samples", ""]
    log_lines += ["| start_rung | mean_naive | mean_model | mean_improvement |",
                  "|-----------:|-----------:|-----------:|-----------------:|"]
    for r in args.start_rungs:
        n = float(np.mean(aggregate[r]["naive"]))
        m = float(np.mean(aggregate[r]["model"]))
        log_lines.append(f"| {r} | {n:.2f} | {m:.2f} | {n-m:+.2f} ({100*(1-m/n) if n>0 else 0:+.1f}%) |")

    (args.out / "L1_summary.md").write_text("\n".join(log_lines))
    print(f"wrote {args.out / 'L1_summary.md'}")


if __name__ == "__main__":
    main()
