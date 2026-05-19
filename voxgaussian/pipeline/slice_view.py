"""
slice_view.py - One-shot tool: connect to the live WS, grab the latest
voxel snapshot, and print a horizontal slice through the scene's midpoint
as ASCII art. Used to check whether the voxel field is hollow (good,
surface-only) or solid (bad, inpaint depth-clipping).
"""
import asyncio
import json
import sys
import websockets


CLASS_GLYPHS = {
    0: ".",  # empty (shouldn't appear in only_occupied snapshot)
    1: "~",  # sky
    2: "g",  # ground
    3: "p",  # path
    4: "w",  # water
    5: "#",  # wall
    6: "B",  # building
    7: "T",  # vegetation
    8: "o",  # prop
    9: "C",  # character
    10:"*",  # fx
}


async def fetch_snapshot(uri: str) -> dict:
    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        msg = await ws.recv()
        return json.loads(msg)


def render_slice(snap: dict, band: int = 1) -> None:
    res = snap["resolution"]
    cells = snap["cells"]
    y_mid = res // 2

    grid = [["." for _ in range(res)] for _ in range(res)]
    counts = {}
    for row in cells:
        ix, iy, iz, cls = row[0], row[1], row[2], row[3]
        if abs(iy - y_mid) <= band:
            grid[iz][ix] = CLASS_GLYPHS.get(cls, "?")
            counts[cls] = counts.get(cls, 0) + 1

    print(f"\nsnapshot: iter={snap.get('iteration')} resolution={res} total_cells={len(cells)}")
    print(f"slice: y={y_mid} +/-{band} (top-down view, X horizontal, Z vertical)")
    print("  " + "".join(str(i % 10) for i in range(res)))
    for iz in range(res - 1, -1, -1):
        print(f"{iz % 100:>2}" + "".join(grid[iz]))

    total_slice = sum(counts.values())
    slice_area = res * res
    print(f"\noccupied in slice band: {total_slice} / {slice_area} cells "
          f"({100 * total_slice / slice_area:.1f}%)")

    names = snap.get("class_names", {})
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        key = str(cls) if str(cls) in names else cls
        print(f"  class {cls} ({names.get(key, '?')}): {n}  '{CLASS_GLYPHS.get(cls, '?')}'")

    # Hollow vs solid heuristic: count "enclosed" cells (cells with non-empty
    # cells in all 4 cardinal directions within the slice) — if >>0, looks solid.
    occ = {(ix, iz): grid[iz][ix] for iz in range(res) for ix in range(res) if grid[iz][ix] != "."}
    enclosed = 0
    for (ix, iz) in occ:
        n = sum(1 for d in [(1,0),(-1,0),(0,1),(0,-1)] if (ix+d[0], iz+d[1]) in occ)
        if n == 4:
            enclosed = enclosed + 1
    print(f"\nfully-enclosed cells in slice: {enclosed}  "
          f"({100 * enclosed / max(1,total_slice):.1f}% of occupied)")
    print("  high % = solid fill (bad), low % = surface shell (good)")


async def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
    band = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if src.startswith("ws://") or src.startswith("wss://"):
        print(f"connecting to {src} ...")
        snap = await fetch_snapshot(src)
    else:
        print(f"loading {src} ...")
        with open(src, "r") as f:
            snap = json.load(f)
    render_slice(snap, band=band)


if __name__ == "__main__":
    asyncio.run(main())
