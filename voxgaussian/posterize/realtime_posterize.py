"""
Live bit-depth ladder preview.

Default: opens the default webcam and shows a 3x3 grid (RGB + 8 ladder
rungs from 1 BPP to 8 BPP) at the camera's native framerate.

Controls:
  ESC / Q  quit
  S        save current grid as ladder_<timestamp>.png
  SPACE    pause / unpause
  1..8     toggle which single rung to show fullscreen (0 = grid view)

Usage:
  python realtime_posterize.py                  # webcam
  python realtime_posterize.py --video clip.mp4
  python realtime_posterize.py --image still.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from posterize import posterize_luma_ladder, ladder_to_grid


def render(rgb: np.ndarray, focus: int | None) -> np.ndarray:
    ladder = posterize_luma_ladder(rgb, fast=True, with_index=False)
    if focus is None or focus == 0:
        return ladder_to_grid(ladder)
    d = ladder[focus]["display"]
    return np.stack([d, d, d], axis=-1)


def fit_to_screen(img: np.ndarray, max_w: int = 1600, max_h: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description="Live bit-depth ladder preview.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", type=Path, help="play a video file")
    src.add_argument("--image", type=Path, help="static image (loops)")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    if args.image is not None:
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read {args.image}")
        rgb_static = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cap = None
    else:
        rgb_static = None
        if args.video is not None:
            cap = cv2.VideoCapture(str(args.video))
        else:
            cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise SystemExit("could not open video source")

    paused = False
    focus = 0
    last_grid = None
    t_last = time.time()
    fps_avg = 0.0
    win = "posterize ladder (ESC quit, S save, SPACE pause, 1-8 focus, 0 grid)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            if rgb_static is not None:
                rgb = rgb_static
            else:
                ok, bgr = cap.read()
                if not ok:
                    if args.video is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            grid = render(rgb, focus if focus > 0 else None)
            last_grid = grid

        if last_grid is not None:
            disp = fit_to_screen(last_grid)
            t = time.time()
            dt = t - t_last
            t_last = t
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps
            label = f"{disp.shape[1]}x{disp.shape[0]}  {fps_avg:5.1f} fps"
            if focus > 0:
                label += f"  [focus: {focus} BPP, {2**focus} levels]"
            if paused:
                label += "  [PAUSED]"
            cv2.putText(disp, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('s') or key == ord('S'):
            if last_grid is not None:
                fn = f"ladder_{int(time.time())}.png"
                cv2.imwrite(fn, cv2.cvtColor(last_grid, cv2.COLOR_RGB2BGR))
                print(f"saved {fn}")
        if ord('0') <= key <= ord('8'):
            focus = key - ord('0')

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
