"""
voxel_store.py — Sparse histogram-per-voxel data structure.

The core idea (from the design conversation):
  Every voxel in the scene's bounding cube holds a HISTOGRAM of class-id votes
  accumulated across iterative inpaint passes. The 'true' class at any moment
  is the mode of the histogram; confidence is mode_count / total_count. This
  gives us:
    - automatic outlier rejection (one bad vote can't flip a 9-vote-strong cell)
    - per-voxel convergence detection (histogram stops changing → voxel is done)
    - boundary identification (two close competing classes → boundary voxel)
    - cheap rollback (subtract a pass's contributions if it was bad)

Storage is sparse: only voxels that have ever received a vote are stored.
Most of any scene's bounding cube is empty space and never gets touched.

Multi-resolution: we run coarse-first (e.g. 32³) then subdivide regions with
low confidence into a finer grid (256³ inside the contested area). Coarse
voxels become "summary" nodes that point to a finer grid for that octant.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
import json
import math
import numpy as np


# Class taxonomy from the design conversation.
# Keep IDs stable: walkability rules + WebXR viewer colors map by these.
CLASSES: dict[int, str] = {
    0:  "empty",
    1:  "sky",
    2:  "ground",       # walkable: grass, dirt
    3:  "path",         # walkable: stone, cobble, wood
    4:  "water",        # non-walkable, visible
    5:  "wall",         # vertical occluder
    6:  "building",     # compound structure
    7:  "vegetation",   # tree, bush — visual but blocking
    8:  "prop",         # well, barrel, sign, statue
    9:  "character",    # humanoid
    10: "fx",           # smoke, fire, glow
}

# Class display colors (hex). Used by viewer + debug dumps.
CLASS_COLORS: dict[int, str] = {
    0:  "#000000",
    1:  "#7ec8ff",  # sky blue
    2:  "#5cb85c",  # grass green
    3:  "#bfa67a",  # path tan
    4:  "#3a7da8",  # water blue
    5:  "#9a9a9a",  # wall grey
    6:  "#a06848",  # building brown
    7:  "#2d8a3a",  # vegetation dark green
    8:  "#c87a48",  # prop orange-brown
    9:  "#e8c4a0",  # character skin
    10: "#ffae40",  # fx amber
}

WALKABLE_CLASSES = {2, 3}   # ground + path
SOLID_CLASSES    = {5, 6, 7, 8, 9}   # block movement
EMPTY_CLASSES    = {0, 1}             # don't render as occupied


@dataclass
class VoxelStore:
    """Sparse histogram-per-voxel grid in scene-local coordinates.

    The grid spans [-extent, +extent] on each axis, divided into `resolution`
    cells per axis. Each cell that has received any vote stores a histogram
    keyed by class_id → int count.

    Lookups use integer voxel-index tuples (ix, iy, iz). Conversion from
    world-space (x, y, z) uses `world_to_voxel`.
    """
    extent: float = 4.0                # half-width of scene cube in meters
    resolution: int = 128              # cells per axis at this level
    parent_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Per-voxel class histograms. Sparse — only touched voxels stored.
    # Key: (ix, iy, iz) tuple. Value: dict[class_id, count]
    cells: dict[tuple[int, int, int], dict[int, int]] = field(default_factory=dict)

    # Per-voxel running RGB sum: idx -> [r_sum, g_sum, b_sum, n].
    # RGB sampled from each inpaint at the hit pixel; mean gives the voxel's
    # multi-view averaged color.
    colors: dict[tuple[int, int, int], list[float]] = field(default_factory=dict)

    # Per-voxel running sub-cell offset sum: idx -> [ox, oy, oz, n] (world m).
    # Lets the viewer render the splat at the actual surface position inside
    # the cell, not at the cell center — kills the "chunky cube" look.
    offsets: dict[tuple[int, int, int], list[float]] = field(default_factory=dict)

    # Per-voxel running surface-normal sum: idx -> [nx, ny, nz, n].
    # Each vote records the direction from voxel center toward the camera that
    # hit it (a ray from camera lit this surface => the normal faces the camera).
    # Used for backface culling in the viewer.
    normals: dict[tuple[int, int, int], list[float]] = field(default_factory=dict)

    # Per-voxel cap on history length — prevents stale votes from dominating
    # (older votes age out implicitly when we cap total count).
    history_cap: int = 64

    # Iteration counter so we can correlate state with WS pushes.
    iteration: int = 0

    # ─── Coordinate transforms ───────────────────────────────────────────

    @property
    def cell_size(self) -> float:
        return (2 * self.extent) / self.resolution

    def world_to_voxel(self, pos: tuple[float, float, float]) -> tuple[int, int, int]:
        """World-space (x, y, z) → (ix, iy, iz) integer voxel index.

        World y=0 is the floor; positive y up. Voxel grid is centered at
        parent_origin in world space.
        """
        x, y, z = pos
        ox, oy, oz = self.parent_origin
        cs = self.cell_size
        ix = int((x - ox + self.extent) / cs)
        iy = int((y - oy + self.extent) / cs)
        iz = int((z - oz + self.extent) / cs)
        return (ix, iy, iz)

    def voxel_to_world(self, idx: tuple[int, int, int]) -> tuple[float, float, float]:
        """Voxel index → world-space CENTER of that voxel."""
        ix, iy, iz = idx
        ox, oy, oz = self.parent_origin
        cs = self.cell_size
        x = ox - self.extent + (ix + 0.5) * cs
        y = oy - self.extent + (iy + 0.5) * cs
        z = oz - self.extent + (iz + 0.5) * cs
        return (x, y, z)

    def in_bounds(self, idx: tuple[int, int, int]) -> bool:
        ix, iy, iz = idx
        return 0 <= ix < self.resolution and 0 <= iy < self.resolution and 0 <= iz < self.resolution

    # ─── Vote operations ─────────────────────────────────────────────────

    def vote_appearance(self,
                        idx: tuple[int, int, int],
                        rgb: tuple[float, float, float] | None,
                        offset_world: tuple[float, float, float],
                        normal_world: tuple[float, float, float] | None = None) -> None:
        """Accumulate a per-voxel color sample, sub-cell surface offset, and
        surface normal.

        `rgb` is in 0..1, or None to skip the color part of this vote.
        `offset_world` is the world-space displacement from voxel center to
        the actual ray-hit point.
        `normal_world` is a unit vector from the hit point toward the camera
        that voted this voxel (which is approximately the surface normal —
        a ray that lit the surface came from "outside" it).
        All three maintain a running mean across votes.
        """
        if not self.in_bounds(idx):
            return
        if rgb is not None:
            c = self.colors.get(idx)
            if c is None:
                self.colors[idx] = [rgb[0], rgb[1], rgb[2], 1.0]
            else:
                c[0] += rgb[0]; c[1] += rgb[1]; c[2] += rgb[2]; c[3] += 1.0
        o = self.offsets.get(idx)
        if o is None:
            self.offsets[idx] = [offset_world[0], offset_world[1], offset_world[2], 1.0]
        else:
            o[0] += offset_world[0]; o[1] += offset_world[1]; o[2] += offset_world[2]; o[3] += 1.0
        if normal_world is not None:
            nrm = self.normals.get(idx)
            if nrm is None:
                self.normals[idx] = [normal_world[0], normal_world[1], normal_world[2], 1.0]
            else:
                nrm[0] += normal_world[0]; nrm[1] += normal_world[1]; nrm[2] += normal_world[2]; nrm[3] += 1.0

    def vote(self, idx: tuple[int, int, int], class_id: int, weight: int = 1) -> None:
        """Add a class-id vote to a voxel's histogram."""
        if not self.in_bounds(idx):
            return
        hist = self.cells.setdefault(idx, {})
        hist[class_id] = hist.get(class_id, 0) + weight
        # Cap total count by aging the smallest entry if we overflow
        total = sum(hist.values())
        if total > self.history_cap:
            # Halve all counts (geometric decay, preserves relative ratios)
            for k in list(hist.keys()):
                hist[k] = max(1, hist[k] // 2)
            # Drop entries that decayed to 1 if mode is much stronger
            mode_count = max(hist.values())
            for k in list(hist.keys()):
                if hist[k] < max(2, mode_count // 8):
                    del hist[k]
            if not hist:
                del self.cells[idx]

    # ─── Queries ─────────────────────────────────────────────────────────

    def mode(self, idx: tuple[int, int, int]) -> tuple[int, float]:
        """Return (mode_class_id, confidence_in_[0,1]) for a voxel.

        Confidence = mode_count / total_count. For untouched voxels, returns
        (0, 0.0) — empty class with zero confidence.
        """
        hist = self.cells.get(idx)
        if not hist:
            return (0, 0.0)
        total = sum(hist.values())
        best_class = max(hist, key=hist.get)
        return (best_class, hist[best_class] / total)

    def is_converged(self, idx: tuple[int, int, int], min_confidence: float = 0.7, min_observations: int = 4) -> bool:
        """A voxel is converged when its histogram has enough observations
        AND the dominant class holds a clear majority."""
        hist = self.cells.get(idx)
        if not hist:
            return False
        total = sum(hist.values())
        if total < min_observations:
            return False
        mode_count = max(hist.values())
        return (mode_count / total) >= min_confidence

    def uncertainty(self, idx: tuple[int, int, int]) -> float:
        """Inverse-confidence score. 0 = certain, 1 = no info. Used for
        active-view selection (pick views whose frustum maximizes summed
        uncertainty)."""
        hist = self.cells.get(idx)
        if not hist:
            return 1.0  # totally unknown
        total = sum(hist.values())
        mode_count = max(hist.values())
        return 1.0 - (mode_count / total)

    def is_solid(self, idx: tuple[int, int, int]) -> bool:
        """Is this voxel an opaque/blocking occupant?"""
        cls, conf = self.mode(idx)
        return cls in SOLID_CLASSES and conf > 0.4

    def is_walkable(self, idx: tuple[int, int, int]) -> bool:
        cls, conf = self.mode(idx)
        return cls in WALKABLE_CLASSES and conf > 0.4

    # ─── Iteration & stats ──────────────────────────────────────────────

    def occupied(self) -> Iterator[tuple[tuple[int, int, int], int, float]]:
        """Yield (idx, mode_class, confidence) for every cell with any vote."""
        for idx, hist in self.cells.items():
            total = sum(hist.values())
            best_class = max(hist, key=hist.get)
            yield idx, best_class, hist[best_class] / total

    def stats(self) -> dict:
        """Statistics for the live viewer's HUD — includes per-class counts,
        confidence distribution, and overall convergence percentage."""
        n_total = len(self.cells)
        n_converged = 0
        per_class: dict[int, int] = {}
        # Confidence histogram in 10% buckets
        conf_hist = [0] * 10
        sum_conf = 0.0
        for idx, cls, conf in self.occupied():
            per_class[cls] = per_class.get(cls, 0) + 1
            if self.is_converged(idx):
                n_converged += 1
            sum_conf += conf
            bucket = min(9, int(conf * 10))
            conf_hist[bucket] += 1
        return {
            "iteration": self.iteration,
            "resolution": self.resolution,
            "extent": self.extent,
            "active_voxels": n_total,
            "converged_voxels": n_converged,
            "convergence_pct": round(100 * n_converged / max(1, n_total), 1),
            "mean_confidence": round(sum_conf / max(1, n_total), 3),
            "confidence_histogram": conf_hist,   # 10 bins: [0-10%, 10-20%, ..., 90-100%]
            "per_class_counts": per_class,
            "class_names": CLASSES,
        }

    # ─── Serialization for WebSocket live streaming ────────────────────

    def snapshot(self, only_occupied: bool = True, downsample: int | None = None) -> dict:
        """JSON-serializable snapshot for streaming to the viewer.

        Compact row format per cell:
          [ix, iy, iz, cls, conf, r, g, b, ox, oy, oz, nx, ny, nz, obs, mrg]

        - conf: 0..255 byte (mode_count / total)
        - r/g/b: 0..255 bytes, or -1 sentinel if no color sampled yet
        - ox/oy/oz: signed bytes -127..+127 → [-half_cell, +half_cell]
        - nx/ny/nz: signed bytes -127..+127 → unit-vector surface normal
          (each component in [-1,+1]). All zeros = no normal sampled.
        - obs: 0..255 byte = clamp(log2(total_votes + 1) * 16, 0, 255).
          Distinguishes a 3-vote 100%-confident voxel from a 300-vote one.
        - mrg: 0..255 byte = (top_count - runner_up_count) / total * 255.
          The ambiguity margin: high = confident decision, low = controversial.
          Drives active-view-selection priority alongside `obs`.

        Snapshot metadata also carries the world-space `centroid` of all
        non-empty occupied voxels, so the viewer can do the "dollhouse
        cutaway" cull (interior faces visible from any angle).
        """
        cells_list: list[list[int]] = []

        def _encode_offset(offset_m: float, half_cell: float) -> int:
            if half_cell <= 1e-9:
                return 0
            v = max(-1.0, min(1.0, offset_m / half_cell))
            return int(round(v * 127))

        def _encode_unit(v: float) -> int:
            return max(-127, min(127, int(round(max(-1.0, min(1.0, v)) * 127))))

        def _obs_margin(hist: dict[int, int]) -> tuple[int, int]:
            """Compute (obs_log_byte, margin_byte) from a class-count histogram.

            obs_log = clamp(log2(total + 1) * 16, 0, 255)  →  observation count
                                                              on log scale.
            margin  = (top - runner_up) / total * 255       →  decision ambiguity.

            For untouched voxels (empty hist) returns (0, 0).
            """
            if not hist:
                return 0, 0
            total = sum(hist.values())
            if total <= 0:
                return 0, 0
            sorted_counts = sorted(hist.values(), reverse=True)
            top = sorted_counts[0]
            runner = sorted_counts[1] if len(sorted_counts) > 1 else 0
            obs_b = min(255, max(0, int(math.log2(total + 1) * 16)))
            mrg_b = min(255, max(0, int((top - runner) / total * 255)))
            return obs_b, mrg_b

        # Running centroid sums (mean position of non-empty occupied voxels)
        cent_sx = cent_sy = cent_sz = 0.0
        cent_n = 0

        if downsample and downsample > 1:
            # Aggregate fine voxels into coarse cells. Track dominant class
            # and per-coarse-cell mean RGB + mean world-offset + mean normal.
            seen_coarse: dict[tuple[int, int, int], dict[int, int]] = {}
            color_sum: dict[tuple, list[float]] = {}
            offset_sum: dict[tuple, list[float]] = {}
            normal_sum: dict[tuple, list[float]] = {}
            for idx, cls, conf in self.occupied():
                cidx = (idx[0] // downsample, idx[1] // downsample, idx[2] // downsample)
                bucket = seen_coarse.setdefault(cidx, {})
                bucket[cls] = bucket.get(cls, 0) + 1
                fc = self.colors.get(idx)
                if fc is not None and fc[3] > 0:
                    n = fc[3]
                    cs = color_sum.setdefault(cidx, [0.0, 0.0, 0.0, 0.0])
                    cs[0] += fc[0] / n; cs[1] += fc[1] / n; cs[2] += fc[2] / n; cs[3] += 1.0
                fo = self.offsets.get(idx)
                if fo is not None and fo[3] > 0:
                    n = fo[3]
                    # Convert fine-cell-local offset to coarse-cell-local:
                    # fine_world_offset + (fine_center - coarse_center)
                    fine_center = self.voxel_to_world(idx)
                    # coarse_center = origin - extent + (c + 0.5) * coarse_cs
                    #               = origin - extent + (c*ds + 0.5*ds) * fine_cs
                    coarse_center = (
                        self.parent_origin[0] - self.extent + (cidx[0] * downsample + 0.5 * downsample) * self.cell_size,
                        self.parent_origin[1] - self.extent + (cidx[1] * downsample + 0.5 * downsample) * self.cell_size,
                        self.parent_origin[2] - self.extent + (cidx[2] * downsample + 0.5 * downsample) * self.cell_size,
                    )
                    dx = fo[0] / n + (fine_center[0] - coarse_center[0])
                    dy = fo[1] / n + (fine_center[1] - coarse_center[1])
                    dz = fo[2] / n + (fine_center[2] - coarse_center[2])
                    os_ = offset_sum.setdefault(cidx, [0.0, 0.0, 0.0, 0.0])
                    os_[0] += dx; os_[1] += dy; os_[2] += dz; os_[3] += 1.0
                fn = self.normals.get(idx)
                if fn is not None and fn[3] > 0:
                    n = fn[3]
                    ns = normal_sum.setdefault(cidx, [0.0, 0.0, 0.0, 0.0])
                    ns[0] += fn[0] / n; ns[1] += fn[1] / n; ns[2] += fn[2] / n; ns[3] += 1.0

            effective_res = self.resolution // downsample
            coarse_cell = (2 * self.extent) / effective_res
            half_cell = coarse_cell / 2
            for cidx, classes in seen_coarse.items():
                cls = max(classes, key=classes.get)
                if only_occupied and cls in EMPTY_CLASSES:
                    continue
                total = sum(classes.values())
                conf_byte = min(255, int(255 * classes[cls] / total))
                cs = color_sum.get(cidx)
                if cs and cs[3] > 0:
                    r = max(0, min(255, int(255 * cs[0] / cs[3])))
                    g = max(0, min(255, int(255 * cs[1] / cs[3])))
                    b = max(0, min(255, int(255 * cs[2] / cs[3])))
                else:
                    r = g = b = -1
                os_ = offset_sum.get(cidx)
                if os_ and os_[3] > 0:
                    ox = _encode_offset(os_[0] / os_[3], half_cell)
                    oy = _encode_offset(os_[1] / os_[3], half_cell)
                    oz = _encode_offset(os_[2] / os_[3], half_cell)
                else:
                    ox = oy = oz = 0
                ns = normal_sum.get(cidx)
                if ns and ns[3] > 0:
                    # Renormalize the mean normal back to unit length
                    nx = ns[0] / ns[3]; ny = ns[1] / ns[3]; nz = ns[2] / ns[3]
                    mag = (nx*nx + ny*ny + nz*nz) ** 0.5
                    if mag > 1e-6:
                        nx /= mag; ny /= mag; nz /= mag
                    nx_b = _encode_unit(nx); ny_b = _encode_unit(ny); nz_b = _encode_unit(nz)
                else:
                    nx_b = ny_b = nz_b = 0
                obs_b, mrg_b = _obs_margin(classes)
                cells_list.append([cidx[0], cidx[1], cidx[2], cls, conf_byte, r, g, b,
                                   ox, oy, oz, nx_b, ny_b, nz_b, obs_b, mrg_b])
                # Centroid: world-space center of this coarse cell
                wx = self.parent_origin[0] - self.extent + (cidx[0] + 0.5) * coarse_cell
                wy = self.parent_origin[1] - self.extent + (cidx[1] + 0.5) * coarse_cell
                wz = self.parent_origin[2] - self.extent + (cidx[2] + 0.5) * coarse_cell
                cent_sx += wx; cent_sy += wy; cent_sz += wz; cent_n += 1
            effective_extent = self.extent
        else:
            half_cell = self.cell_size / 2
            for idx, cls, conf in self.occupied():
                if only_occupied and cls in EMPTY_CLASSES:
                    continue
                fc = self.colors.get(idx)
                if fc is not None and fc[3] > 0:
                    n = fc[3]
                    r = max(0, min(255, int(255 * fc[0] / n)))
                    g = max(0, min(255, int(255 * fc[1] / n)))
                    b = max(0, min(255, int(255 * fc[2] / n)))
                else:
                    r = g = b = -1
                fo = self.offsets.get(idx)
                if fo is not None and fo[3] > 0:
                    n = fo[3]
                    ox = _encode_offset(fo[0] / n, half_cell)
                    oy = _encode_offset(fo[1] / n, half_cell)
                    oz = _encode_offset(fo[2] / n, half_cell)
                else:
                    ox = oy = oz = 0
                fn = self.normals.get(idx)
                if fn is not None and fn[3] > 0:
                    n = fn[3]
                    nx = fn[0] / n; ny = fn[1] / n; nz = fn[2] / n
                    mag = (nx*nx + ny*ny + nz*nz) ** 0.5
                    if mag > 1e-6:
                        nx /= mag; ny /= mag; nz /= mag
                    nx_b = _encode_unit(nx); ny_b = _encode_unit(ny); nz_b = _encode_unit(nz)
                else:
                    nx_b = ny_b = nz_b = 0
                obs_b, mrg_b = _obs_margin(self.cells.get(idx, {}))
                cells_list.append([
                    idx[0], idx[1], idx[2], cls, min(255, int(255 * conf)),
                    r, g, b, ox, oy, oz, nx_b, ny_b, nz_b, obs_b, mrg_b,
                ])
                wx, wy, wz = self.voxel_to_world(idx)
                cent_sx += wx; cent_sy += wy; cent_sz += wz; cent_n += 1
            effective_res = self.resolution
            effective_extent = self.extent

        if cent_n > 0:
            centroid = [cent_sx / cent_n, cent_sy / cent_n, cent_sz / cent_n]
        else:
            centroid = list(self.parent_origin)

        return {
            "type": "voxel_snapshot",
            "iteration": self.iteration,
            "resolution": effective_res,
            "extent": effective_extent,
            "origin": list(self.parent_origin),
            "centroid": centroid,
            "class_colors": CLASS_COLORS,
            "class_names": CLASSES,
            "cells": cells_list,
            "stats": self.stats(),
        }

    def to_json_file(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, separators=(",", ":"))


# ─── Multi-resolution refinement (poor-man's octree) ────────────────────
# We keep multiple VoxelStore instances at different resolutions.
# Coarse-level convergence freezes those cells; fine-level grids are spawned
# only inside coarse cells that remain uncertain.

@dataclass
class MultiResolutionVoxels:
    """Hierarchy: a coarse VoxelStore drives global topology; fine stores
    refine only the regions where the coarse grid is uncertain."""
    extent: float = 4.0
    coarse_resolution: int = 32
    fine_resolution: int = 128

    coarse: VoxelStore = field(init=False)
    fine_stores: dict[tuple[int, int, int], VoxelStore] = field(default_factory=dict)

    def __post_init__(self):
        self.coarse = VoxelStore(extent=self.extent, resolution=self.coarse_resolution)

    def maybe_subdivide(self, coarse_idx: tuple[int, int, int]) -> VoxelStore | None:
        """If a coarse voxel is occupied but its confidence is low, attach
        a fine sub-grid for it. Returns the (possibly new) fine store."""
        if coarse_idx in self.fine_stores:
            return self.fine_stores[coarse_idx]
        cls, conf = self.coarse.mode(coarse_idx)
        if cls in EMPTY_CLASSES or conf > 0.85:
            return None  # confidently empty or confidently classified — no need
        # Spawn fine store centered on this coarse voxel
        cx, cy, cz = self.coarse.voxel_to_world(coarse_idx)
        coarse_size = self.coarse.cell_size
        fine = VoxelStore(
            extent=coarse_size / 2,
            resolution=self.fine_resolution,
            parent_origin=(cx, cy, cz),
        )
        self.fine_stores[coarse_idx] = fine
        return fine

    def stats(self) -> dict:
        return {
            "coarse": self.coarse.stats(),
            "fine_stores": len(self.fine_stores),
            "fine_active_voxels": sum(len(s.cells) for s in self.fine_stores.values()),
        }
