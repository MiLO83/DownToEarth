"""
Tiny conditional UNet that learns to refine rung-N posterized luma into
rung-(N+1) -- the progressive bit-depth ladder of LYRA2_PROPOSAL.md
section 6.6.2.

Conditioning: the input rung index N is broadcast across the spatial
extent as an extra channel. One model handles all 7 ladder steps.

Inference is just `sample.py` looping the model from rung 1 to rung 8.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from noise_schedule import (
    DEFAULT_SCHEDULE, HALF_BIN, normalized_t, apply_bounded_noise, clip_to_bound,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """
    Cch posterized + 1ch rung-index map  ->  Cch predicted delta to clean.

    The model predicts a DELTA (signed). At sampling time we add the
    delta to the input and clip to the bound for that rung -- this is
    the bounded-deviation property that prevents structural drift.

    C=1 = luma deposterizer.  C=3 = RGB image-to-image deposterizer
    (per-channel posterization recovered to full RGB).
    """
    def __init__(self, base_ch: int = 32, channels: int = 3):
        super().__init__()
        self.channels = channels
        c = base_ch
        self.in_block = ConvBlock(channels + 1, c)
        self.down1 = ConvBlock(c, c * 2)
        self.down2 = ConvBlock(c * 2, c * 4)
        self.mid = ConvBlock(c * 4, c * 4)
        self.up2 = ConvBlock(c * 4 + c * 4, c * 2)
        self.up1 = ConvBlock(c * 2 + c * 2, c)
        self.out_block = ConvBlock(c + c, c)
        self.out = nn.Conv2d(c, channels, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, posterized: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        posterized: (B, C, H, W) float in [0, 1]
        t_norm    : (B,) float in [0, 1] (rung index normalized)
        returns   : (B, C, H, W) signed delta in [-1, 1]
        """
        b, _, h, w = posterized.shape
        t_map = t_norm.view(b, 1, 1, 1).expand(b, 1, h, w)
        x0 = self.in_block(torch.cat([posterized, t_map], dim=1))
        x1 = self.down1(self.pool(x0))
        x2 = self.down2(self.pool(x1))
        m = self.mid(self.pool(x2))
        u2 = self.up2(torch.cat([self.up(m), x2], dim=1))
        u1 = self.up1(torch.cat([self.up(u2), x1], dim=1))
        u0 = self.out_block(torch.cat([self.up(u1), x0], dim=1))
        return torch.tanh(self.out(u0))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LadderDataset(Dataset):
    """
    Reads a directory of <id>/bppN.png (luma) or <id>/rgb_bppN.png (RGB)
    folders produced by scrape_dataset.py / build_rgb_rungs.py.

    Each __getitem__ samples a random ladder step (rung_in, rung_out)
    and returns (input_norm, target_norm, t_norm).

    mode='rgb' loads 3ch rgb_bppN.png pairs.
    mode='luma' loads 1ch bppN.png pairs.
    """
    def __init__(self, root: Path, crop_size: int = 256,
                 rung_to_clean: bool = False,
                 add_noise: bool = True,
                 schedule = None,
                 mode: str = "rgb"):
        self.root = Path(root)
        self.crop = crop_size
        self.rung_to_clean = rung_to_clean
        self.add_noise = add_noise
        self.schedule = schedule or DEFAULT_SCHEDULE
        if mode not in ("rgb", "luma"):
            raise ValueError(f"mode must be 'rgb' or 'luma', got {mode}")
        self.mode = mode
        probe = "rgb_bpp1.png" if mode == "rgb" else "bpp1.png"
        self.ids = sorted(p for p in self.root.iterdir()
                          if p.is_dir() and (p / probe).exists())
        if not self.ids:
            raise FileNotFoundError(f"no <id>/{probe} samples under {root}")

    def __len__(self):
        return len(self.ids)

    def _load_rung(self, sample_dir: Path, bpp: int) -> np.ndarray:
        if self.mode == "rgb":
            path = sample_dir / ("rgb.png" if bpp == 8 else f"rgb_bpp{bpp}.png")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"could not read {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # HxWx3 uint8 RGB
        else:
            path = sample_dir / ("luma.png" if bpp == 8 else f"bpp{bpp}.png")
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"could not read {path}")
            return img

    def __getitem__(self, idx: int):
        sample_dir = self.ids[idx]
        rung_in, rung_out = self.schedule[np.random.randint(len(self.schedule))]
        if self.rung_to_clean:
            rung_out = 8

        a = self._load_rung(sample_dir, rung_in)
        b = self._load_rung(sample_dir, rung_out)

        if a.ndim == 3:
            h, w, _ = a.shape
        else:
            h, w = a.shape
        ch = min(self.crop, h)
        cw = min(self.crop, w)
        y = np.random.randint(0, h - ch + 1)
        x = np.random.randint(0, w - cw + 1)
        a = a[y:y + ch, x:x + cw]
        b = b[y:y + ch, x:x + cw]

        if self.add_noise:
            bound = HALF_BIN[rung_in]
            a = apply_bounded_noise(a, bound)

        if a.ndim == 3:
            # HWC -> CHW
            a_t = torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
            b_t = torch.from_numpy(b.astype(np.float32) / 255.0).permute(2, 0, 1)
        else:
            a_t = torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)
            b_t = torch.from_numpy(b.astype(np.float32) / 255.0).unsqueeze(0)
        t_t = torch.tensor(normalized_t(rung_in), dtype=torch.float32)
        return a_t, b_t, t_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_root: Path,
    out_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    crop_size: int = 256,
    lr: float = 3e-4,
    base_ch: int = 32,
    workers: int = 4,
    device: str = "cuda",
    save_every: int = 5,
    rung_to_clean: bool = False,
    add_noise: bool = True,
    mode: str = "rgb",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = LadderDataset(data_root, crop_size=crop_size,
                       rung_to_clean=rung_to_clean, add_noise=add_noise,
                       mode=mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=workers > 0)
    channels = 3 if mode == "rgb" else 1
    model = TinyUNet(base_ch=base_ch, channels=channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: TinyUNet base_ch={base_ch} channels={channels}  params={n_params/1e6:.2f}M  device={device}")
    print(f"data : {len(ds)} samples  mode={mode}  batch={batch_size}  crop={crop_size}  batches/epoch={len(dl)}")

    step = 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for posterized, target, t_norm in dl:
            posterized = posterized.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            t_norm = t_norm.to(device, non_blocking=True)
            # The model predicts a signed delta in [-1, 1]. Target delta is
            # the difference target - posterized, also in [-1, 1] roughly.
            pred_delta = model(posterized, t_norm)
            target_delta = (target - posterized).clamp(-1.0, 1.0)
            loss = F.l1_loss(pred_delta, target_delta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            step += 1
        ep_loss /= max(len(dl), 1)
        elapsed = time.time() - t_start
        print(f"epoch {epoch:3d}/{epochs}  step={step:6d}  loss={ep_loss:.5f}  "
              f"elapsed={elapsed/60:5.1f}m  rate={step/elapsed:.1f} it/s")
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = {
                "model": model.state_dict(),
                "base_ch": base_ch,
                "channels": channels,
                "mode": mode,
                "epoch": epoch,
                "step": step,
                "loss": ep_loss,
            }
            torch.save(ckpt, out_dir / f"deposterizer_e{epoch:03d}.pt")
            torch.save(ckpt, out_dir / "deposterizer_latest.pt")
            print(f"  saved -> {out_dir / 'deposterizer_latest.pt'}")


def main():
    ap = argparse.ArgumentParser(description="Train the bit-depth ladder deposterizer.")
    ap.add_argument("--data", type=Path, required=True, help="ladder dataset root (per-id subdirs)")
    ap.add_argument("--out", type=Path, default=Path("./ckpts"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rung-to-clean", action="store_true",
                    help="train rung-N -> rung-8 in one shot (simpler, less iterative)")
    ap.add_argument("--no-noise", action="store_true",
                    help="skip the small uniform noise augmentation")
    ap.add_argument("--mode", choices=["rgb", "luma"], default="rgb",
                    help="train on RGB rungs (default) or luma rungs")
    args = ap.parse_args()
    train(
        data_root=args.data, out_dir=args.out, epochs=args.epochs,
        batch_size=args.batch, crop_size=args.crop, lr=args.lr,
        base_ch=args.base_ch, workers=args.workers, device=args.device,
        rung_to_clean=args.rung_to_clean, add_noise=not args.no_noise,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
