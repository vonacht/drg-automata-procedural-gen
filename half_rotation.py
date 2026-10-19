import math, random
import numpy as np


def _rodrigues(axis, ang):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _random_unit(rng):
    v = np.array([rng.gauss(0, 1) for _ in range(3)])
    return v / np.linalg.norm(v)


def rotate_half(
    room,
    rng=random,
    fraction=None,
    axis=None,
    angle=None,
    angle_range=(0.4, 2.6),
    blend=0.0,
):
    """
    Rotates roughly half of the spheres around a pivot on the cutting plane.
    Radii are untouched, and so is the nested structure and `path`.

    fraction:    share of spheres that rotate (None = random 0.4-0.6)
    axis:        "hinge" (in the cut plane, folds), "normal" (swivels),
                 "random", or None to pick one at random
    angle:       rotation in radians (None = random magnitude from angle_range,
                 random sign)
    blend:       width of a soft transition across the plane, as a fraction of
                 the room's extent; 0 = rigid half, 0.3 = gradual twist/fold
    """
    groups, path = room
    spheres = [s for g in groups for s in g]
    if len(spheres) < 4:
        return [[s.copy() for s in g] for g in groups], [list(p) for p in path]

    C = np.array([s["center"] for s in spheres], dtype=float)

    # cutting plane: random normal, offset so `fraction` of the spheres lie beyond it
    n = _random_unit(rng)
    fraction = rng.uniform(0.4, 0.6) if fraction is None else fraction
    proj = C @ n
    thr = np.quantile(proj, 1 - fraction)
    signed = proj - thr  # > 0: the rotating side

    # pivot: the middle of the seam, i.e. the spheres closest to the plane, on the plane
    seam = np.argsort(np.abs(signed))[: max(3, len(C) // 6)]
    pivot = C[seam].mean(axis=0)
    pivot = pivot - (pivot @ n - thr) * n

    kind = (
        axis
        if axis in ("hinge", "normal", "random")
        else rng.choice(("hinge", "normal", "random"))
    )
    if kind == "normal":
        ax = n
    elif kind == "hinge":
        v = _random_unit(rng)
        ax = v - (v @ n) * n
        ax = ax / np.linalg.norm(ax)
    else:
        ax = _random_unit(rng)

    ang = (
        angle
        if angle is not None
        else rng.uniform(*angle_range) * rng.choice((-1, 1))
    )

    # rotation weight per sphere: 0 = stays, 1 = full rotation
    if blend > 0:
        band = blend * np.ptp(proj)
        t = np.clip(signed / band + 0.5, 0.0, 1.0)
        w = t * t * (3 - 2 * t)
    else:
        w = (signed > 0).astype(float)

    R_full = _rodrigues(ax, ang)
    new = C.copy()
    for i in range(len(C)):
        if w[i] <= 0:
            continue
        R = R_full if w[i] >= 1 else _rodrigues(ax, ang * w[i])
        new[i] = pivot + R @ (C[i] - pivot)

    result, i = [], 0
    for g in groups:
        out = []
        for s in g:
            out.append({**s, "center": tuple(float(x) for x in new[i])})
            i += 1
        result.append(out)
    return result, [list(p) for p in path]
