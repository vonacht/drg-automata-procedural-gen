import math, random
import numpy as np


def periodic_wrap(
    room,
    rng=random,
    axes=None,
    box=None,
    yaw=None,
    seams="split",
    min_points=2,
):
    """
    Wraps the room into a box smaller than itself: anything outside the box
    re-enters from the opposite side. Radii are untouched.

    axes:       which axes to wrap, any of "x", "y", "z" in the (turned) box frame;
                None = random ("xy" most often; "xyz" also changes floor levels)
    box:        box size along each wrapped axis, as a fraction of the room's extent
                (None = random 0.55-0.9; smaller = more of the room gets moved)
    yaw:        how far to turn the box about the vertical axis, in radians
                (None = random), so the seams don't line up with the world axes
    seams:      what to do with a polyline that crosses a seam
                "link"  - keep it whole: the engine draws a long link across the room
                "split" - cut it at the seam, which gives clean separate pieces
    min_points: with seams="split", pieces with fewer spheres than this are dropped
    Returns (groups, path); with "split" the paths are remapped to the new pieces.
    """
    groups, path = room
    spheres = [s for g in groups for s in g]
    if len(spheres) < 2:
        return [[dict(s) for s in g] for g in groups], [list(p) for p in path]

    C = np.array([s["center"] for s in spheres], dtype=float)
    mid = (C.min(0) + C.max(0)) / 2
    axes = axes or rng.choice(("xy", "xy", "x", "y", "xyz"))
    yaw = rng.uniform(0, math.pi) if yaw is None else yaw
    cs, sn = math.cos(yaw), math.sin(yaw)

    # into the box frame (turned by yaw about the vertical axis)
    Q = C - mid
    U = np.stack(
        [cs * Q[:, 0] + sn * Q[:, 1], -sn * Q[:, 0] + cs * Q[:, 1], Q[:, 2]],
        axis=1,
    )
    ext = np.ptp(U, axis=0)

    W = U.copy()
    cell = np.zeros(
        U.shape, dtype=int
    )  # which copy of the box each sphere was in
    for a, name in enumerate("xyz"):
        if name not in axes or ext[a] < 1e-9:
            continue
        L = ext[a] * (rng.uniform(0.55, 0.9) if box is None else box)
        lo = rng.uniform(
            U[:, a].min(), U[:, a].max() - L
        )  # where the box sits in the room
        k = np.floor((U[:, a] - lo) / L)
        W[:, a] = U[:, a] - k * L  # back into [lo, lo + L)
        cell[:, a] = k.astype(int)

    new = mid + np.stack(
        [cs * W[:, 0] - sn * W[:, 1], sn * W[:, 0] + cs * W[:, 1], W[:, 2]],
        axis=1,
    )

    aligned = len(path) == len(groups)
    out_groups, out_path, base = [], [], 0
    for gi, g in enumerate(groups):
        if not g:
            continue
        pieces, cur = [], [0]
        for i in range(1, len(g)):  # cut where the box cell changes
            if (
                seams == "split"
                and (cell[base + i] != cell[base + i - 1]).any()
            ):
                pieces.append(cur)
                cur = []
            cur.append(i)
        pieces.append(cur)

        for piece in pieces:
            if seams == "split" and len(piece) < min_points:
                continue
            out_groups.append(
                [
                    {**g[i], "center": tuple(float(x) for x in new[base + i])}
                    for i in piece
                ]
            )
            if aligned:
                pos = {old: n for n, old in enumerate(piece)}
                out_path.append([pos[i] for i in path[gi] if i in pos])
            else:
                out_path.append(list(range(len(piece))))
        base += len(g)
    return out_groups, out_path
