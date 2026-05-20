"""
Bounded-noise schedule for progressive bit-depth deposterization.

The big idea from LYRA2_PROPOSAL.md §6.6.2: instead of Gaussian noise at
ever-shrinking variance (standard DDPM), use UNIFORM noise at
ever-shrinking BOUNDS, anchored to the previous step's output.

Per-pixel choice space:
    step 0 (start): rung 1 BPP (binary)              -- corruption ~ +/- 64
    step 1        : refine to rung 2 BPP             -- corruption ~ +/- 32
    step 2        : refine to rung 3 BPP             -- corruption ~ +/- 16
    step 3        : refine to rung 4 BPP             -- corruption ~ +/-  8
    step 4        : refine to rung 5 BPP             -- corruption ~ +/-  4
    step 5        : refine to rung 6 BPP             -- corruption ~ +/-  2
    step 6        : refine to rung 7 BPP             -- corruption ~ +/-  1
    step 7        : final 8 BPP luma (clean)         -- corruption     0

Each refinement step has a TINY per-pixel decision space (3-5 deltas),
and the bound from the previous step locks structure -- the model
literally cannot decide "wall is sky" at step 6 because it can only
shift the pixel by +/- 1.

The 8-bpp luma is then the conditioning for a final chroma stage
(future work, see LYRA2_PROPOSAL.md §6.6.3).
"""
from __future__ import annotations

import numpy as np

# Per-rung bin width: rung-N luma values are spaced 255 / (2**N - 1) apart.
# The "bound" of corruption introduced by posterizing to rung N (vs the
# clean 8-bpp luma) is at most half a bin width.
BIN_WIDTH = {bpp: 255.0 / ((1 << bpp) - 1) for bpp in range(1, 9)}
HALF_BIN = {bpp: 0.5 * BIN_WIDTH[bpp] for bpp in range(1, 9)}

# Schedule the trainer/sampler actually uses. List of (rung_in, rung_out) pairs.
# Reading bottom-to-top: each step refines by one rung up the ladder.
DEFAULT_SCHEDULE = [(r, r + 1) for r in range(1, 8)]


def rung_bound_pixels(bpp: int) -> float:
    """Maximum +/- deviation in pixel units between rung-bpp and clean 8-bpp."""
    return HALF_BIN[bpp]


def normalized_t(bpp: int) -> float:
    """Map a rung index 1..8 to a [0, 1] timestep value for the model."""
    return (bpp - 1) / 7.0


def bounded_uniform_noise(shape, bound: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Uniform integer noise in [-bound, +bound] per element, returned as int16."""
    if rng is None:
        rng = np.random.default_rng()
    if bound <= 0:
        return np.zeros(shape, dtype=np.int16)
    b = int(round(bound))
    return rng.integers(-b, b + 1, size=shape, dtype=np.int16)


def apply_bounded_noise(luma_uint8: np.ndarray, bound: float,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Add uniform [-bound, +bound] noise; clip to [0, 255]; return uint8."""
    if bound <= 0:
        return luma_uint8.copy()
    noise = bounded_uniform_noise(luma_uint8.shape, bound, rng)
    out = luma_uint8.astype(np.int16) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def clip_to_bound(prediction: np.ndarray, anchor: np.ndarray, bound: float) -> np.ndarray:
    """
    Project `prediction` onto the [anchor - bound, anchor + bound] box per
    pixel. This is the inference-time enforcement of the bounded-deviation
    property -- the model is free to predict anything, but we constrain
    the result so structure can't drift.
    """
    a = anchor.astype(np.float32)
    p = prediction.astype(np.float32)
    p = np.clip(p, a - bound, a + bound)
    return np.clip(p, 0, 255).astype(np.uint8)


# Aliases that read better in user code.
def corruption_bound_for_rung(bpp: int) -> float:
    """How far rung-bpp pixels can be from the true 8-bpp luma."""
    return rung_bound_pixels(bpp)
