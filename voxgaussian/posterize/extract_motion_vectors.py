"""
Codec motion-vector extractor -- RESEARCH STUB.

Status: scaffolded, not implemented. The decode path is verified working
(see notes below) but actual usage as a training signal is future work.

================================================================
WHAT THIS IS FOR
================================================================

Every modern video codec (H.264, H.265, AV1) computes per-macroblock
motion vectors during encode. They're already sitting in the bitstream.
Pulling them out gives you:

  - A coarse but FREE optical-flow field on any encoded video
  - A natural training target for "MV synthesis" models (learn what the
    codec would have produced -- a strong prior for video diffusion)
  - A cheap conditioning signal for video diffusion: feed last-frame MVs
    into the noise predictor, much smaller than dense RAFT/GMA flow

This pairs with the bit-depth ladder work because MVs add a *temporal*
prior to the spatial decomposition in LYRA2_PROPOSAL.md §6.6: position
(UVW) -> luminance (256-level) -> bit-depth refinement -> chroma. MVs
are an obvious fifth axis -- "where each block moved from."

================================================================
VERIFIED WORKING (smoke-tested 2026-05-20)
================================================================

PyAV 15.1.0 on Windows. To enable MV export, set the codec context
flag BEFORE the first decode:

    AV_CODEC_FLAG2_EXPORT_MVS = 1 << 28

    container = av.open(path)
    stream = container.streams.video[0]
    stream.codec_context.flags2 |= AV_CODEC_FLAG2_EXPORT_MVS

Then iterate frames; MVs appear as a side_data with type name
"MOTION_VECTORS". `side.to_ndarray()` returns a structured array with
fields below. Verified: ~1800 MVs/frame on a 640x360 H.264 clip; only
I-frames lack MVs.

================================================================
AVMotionVector dtype (as PyAV exposes it)
================================================================

    source        i4   -1 = past reference, 1 = future reference
    w             u1   block width in pixels  (typically 8 or 16)
    h             u1   block height in pixels
    src_x         i2   source center x
    src_y         i2   source center y
    dst_x         i2   destination center x
    dst_y         i2   destination center y
    flags         u8   codec-specific flags (padded for alignment)
    motion_x      i4   motion in 1/motion_scale pixels along x
    motion_y      i4   motion in 1/motion_scale pixels along y
    motion_scale  u2   typically 4 (quarter-pel motion)

So pixel-domain displacement is: (motion_x / motion_scale,
motion_y / motion_scale). Center of source block is (src_x, src_y);
center of dest block is (dst_x, dst_y); they differ by exactly the
displacement.

================================================================
TODO (future implementation)
================================================================

1. Per-frame extractor -> npz dump (one file per frame OR one streamed
   sequence per video; pick based on training-time read pattern).

2. Dense-flow projection: scatter MVs onto a (H, W, 2) float32 grid for
   use as a diffusion conditioning signal. Pixels outside any MB stay
   at zero; could fill with nearest-neighbor or leave sparse.

3. HSV flow viz (cv2.cartToPolar -> HSV) for sanity-checking.

4. Cleaner I-frame handling: emit a zero-MV array with a flag so the
   training loader can choose to skip or treat as "all-zero flow."

5. Batch CLI: walk a directory of mp4s, extract all, build a single
   sharded dataset.

6. The interesting next step -- "MV synthesis" head: train a small
   network conditioned on (last RGB frame, last bit-depth ladder state)
   to predict the next frame's MV field. If that works, it gives us
   essentially-free motion priors for video diffusion.

================================================================
"""
from __future__ import annotations

AV_CODEC_FLAG2_EXPORT_MVS = 1 << 28  # libavcodec constant; verified


def extract(*args, **kwargs):
    """STUB. See module docstring for the verified decode recipe."""
    raise NotImplementedError(
        "Motion-vector extractor is currently a research stub. "
        "The PyAV decode path is verified working; full implementation "
        "is deferred. See module docstring for the recipe."
    )


def mv_field_to_flow(*args, **kwargs):
    """STUB. Will project AVMotionVector array onto a dense (H, W, 2) flow."""
    raise NotImplementedError("future work")


def flow_to_hsv(*args, **kwargs):
    """STUB. Standard HSV optical-flow visualization."""
    raise NotImplementedError("future work")


if __name__ == "__main__":
    print(__doc__)
