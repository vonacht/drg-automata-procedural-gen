import math, random
import numpy as np


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# ---- directions -------------------------------------------------------------
def _vertical(rng):
    return np.array([0.0, 0.0, 1.0])


def _tilted(rng, max_tilt=0.45, min_tilt=0.15):
    th, az = rng.uniform(min_tilt, max_tilt), rng.uniform(0, 2 * math.pi)
    return np.array(
        [
            math.sin(th) * math.cos(az),
            math.sin(th) * math.sin(az),
            math.cos(th),
        ]
    )


def _horizontal(rng):
    az = rng.uniform(0, 2 * math.pi)
    return np.array([math.cos(az), math.sin(az), 0.0])


def _diagonal(rng):
    el, az = rng.uniform(0.5, 1.0), rng.uniform(
        0, 2 * math.pi
    )  # 30-57 degrees up
    return np.array(
        [
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ]
    )


PILLAR_TYPES = {
    "column": dict(w=3.0, dir=_vertical, shape="straight"),
    "leaning": dict(w=2.0, dir=_tilted, shape="straight"),
    "bowed": dict(w=2.0, dir=_vertical, shape="bow"),  # C or S curve
    "twisted": dict(w=1.0, dir=_vertical, shape="helix"),
    "stalagmite": dict(
        w=1.0, dir=_vertical, shape="straight", partial="floor"
    ),
    "stalactite": dict(
        w=1.0, dir=_vertical, shape="straight", partial="ceiling"
    ),
    "beam": dict(w=1.5, dir=_horizontal, shape="straight"),
    "arch": dict(
        w=1.0, dir=_horizontal, shape="arch"
    ),  # horizontal, bowing up
    "brace": dict(w=1.0, dir=_diagonal, shape="straight"),
}


# ---- geometry helpers -------------------------------------------------------
def _ray_extent(C, R2, p, d, flat=None):
    """Contiguous [t_lo, t_hi] around t=0 where p + t*d is inside the union of
    spheres (or hemispheres, if `flat` is "up"/"down")."""
    oc = p - C
    b = oc @ d
    disc = b * b - ((oc * oc).sum(1) - R2)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    lo, hi = -b - sq, -b + sq
    if flat is not None:
        s = 1.0 if flat == "up" else -1.0
        if abs(d[2]) < 1e-12:
            ok &= s * (p[2] - C[:, 2]) >= 0
        else:
            tz = (C[:, 2] - p[2]) / d[2]
            if s * d[2] > 0:
                lo = np.maximum(lo, tz)
            else:
                hi = np.minimum(hi, tz)
            ok &= lo <= hi
    lo, hi = lo[ok], hi[ok]

    t_hi = 0.0
    while (
        True
    ):  # grow forward through every interval that touches the current end
        m = (lo <= t_hi + 1e-9) & (hi > t_hi + 1e-9)
        if not m.any():
            break
        t_hi = hi[m].max()
    t_lo = 0.0
    while True:
        m = (hi >= t_lo - 1e-9) & (lo < t_lo - 1e-9)
        if not m.any():
            break
        t_lo = lo[m].min()
    return t_lo, t_hi


def _inside(C, R2, P, flat=None):
    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    ok = d2 <= R2[None, :]
    if flat is not None:
        s = 1.0 if flat == "up" else -1.0
        ok &= s * (P[:, None, 2] - C[None, :, 2]) >= 0
    return ok.any(axis=1)


def _curve(A, B, shape, amp, rng, points):
    """Points from A to B, bowed sideways by `amp`. Both ends stay on the A-B line."""
    axis = B - A
    d = _unit(axis)
    if shape == "straight" or amp == 0:
        t = np.linspace(0.0, 1.0, points)[:, None]
        return A + axis * t
    if shape == "helix":
        points *= 2
    t = np.linspace(0.0, 1.0, points)[:, None]
    base = A + axis * t
    if shape == "arch":
        return base + np.array([0.0, 0.0, 1.0]) * amp * np.sin(np.pi * t)
    v = np.array([rng.gauss(0, 1) for _ in range(3)])
    perp = _unit(v - (v @ d) * d)  # random direction across the pillar
    if shape == "c":
        return base + perp * amp * np.sin(np.pi * t)
    if shape == "s":
        return base + perp * amp * np.sin(2 * np.pi * t)
    if shape == "helix":
        perp2 = np.cross(d, perp)
        th = 2 * np.pi * rng.choice((1, 2)) * t + rng.uniform(0, 2 * math.pi)
        env = np.sin(np.pi * t)  # pinned to the axis at both ends
        return base + amp * env * (np.cos(th) * perp + np.sin(th) * perp2)
    raise ValueError(shape)


# ---- main -------------------------------------------------------------------
def sample_pillars(
    cave,
    rng=random,
    n=None,
    types=None,
    min_len=None,
    min_sep=1.2,
    margin=0.12,
    points=8,
    flat_face="up",
    max_tries=None,
    return_types=False,
):
    """
    cave:      list of polylines, each a list of {"center": (x,y,z), "radius": r}
    n:         number of pillars (None = random 2-4)
    types:     subset of PILLAR_TYPES keys to use (None = all)
    min_len:   shortest pillar accepted (None = median sphere radius)
    min_sep:   anchors must be at least this many radii apart
    margin:    how far pillar ends are pushed into the rock, as a fraction of the
               anchor sphere's radius, so they fuse with the surface
    points:    number of points per pillar (twisted pillars get double)
    flat_face: "up" / "down" if the spheres are hemispheres (open side), None for full spheres
    Returns a list of pillars, each a list of (x, y, z) points, or (type, points) pairs.
    """
    spheres = [s for poly in cave for s in poly]
    C = np.array([s["center"] for s in spheres], dtype=float)
    Rad = np.array([s["radius"] for s in spheres], dtype=float)
    R2 = Rad**2
    min_len = float(np.median(Rad)) if min_len is None else min_len
    n = rng.randint(2, 4) if n is None else n
    names = list(types or PILLAR_TYPES)
    weights = [PILLAR_TYPES[k]["w"] for k in names]
    side = {"up": 1.0, "down": -1.0}.get(flat_face)

    anchors, out = [], []
    for _ in range(max_tries or 40 * n):
        if len(out) >= n:
            break

        # 1. anchor: a random point inside a random sphere, bigger spheres more likely
        i = rng.choices(range(len(spheres)), weights=list(Rad))[0]
        v = _unit([rng.gauss(0, 1) for _ in range(3)])
        p = C[i] + v * Rad[i] * 0.8 * rng.random() ** (1 / 3)
        if side is not None and side * (p[2] - C[i][2]) < 0:
            p[2] = 2 * C[i][2] - p[2]  # reflect into the open half
        if any(np.linalg.norm(p - a) < min_sep * Rad[i] for a in anchors):
            continue

        # 2. type, direction and how far open space extends along it
        name = rng.choices(names, weights)[0]
        spec = PILLAR_TYPES[name]
        d = _unit(spec["dir"](rng))
        t_lo, t_hi = _ray_extent(C, R2, p, d, flat_face)
        L = t_hi - t_lo
        if L < min_len:
            continue
        A, B = p + d * t_lo, p + d * t_hi  # where the line meets the walls
        m = margin * Rad[i] * d

        # 3. which part of that span the pillar occupies
        part = spec.get("partial")
        if part == "floor":  # grows up from the floor
            start, end = A - m, A + d * L * rng.uniform(0.3, 0.65)
        elif part == "ceiling":  # hangs down from the ceiling
            start, end = B + m, B - d * L * rng.uniform(0.3, 0.65)
        else:
            start, end = A - m, B + m

        # 4. curve it, backing off the bend until it stays inside the cave
        shape = (
            rng.choice(("c", "s")) if spec["shape"] == "bow" else spec["shape"]
        )
        length = np.linalg.norm(end - start)
        if shape == "helix":
            amp = rng.uniform(0.3, 0.6) * Rad[i]
        elif shape == "arch":
            amp = rng.uniform(0.1, 0.3) * length
        else:
            amp = rng.uniform(0.08, 0.25) * length
        pts = None
        for k in (1.0, 0.7, 0.45, 0.25):
            cand = _curve(start, end, shape, amp * k, rng, points)
            if (
                shape == "straight"
                or _inside(C, R2, cand[1:-1], flat_face).all()
            ):
                pts = cand
                break
        if pts is None:  # nothing fit: fall back to straight
            pts = _curve(start, end, "straight", 0, rng, points)

        anchors.append(p)
        out.append((name, [tuple(float(x) for x in q) for q in pts]))

    return out if return_types else [pts for _, pts in out]


def generate_pillars(room):
    pillars = sample_pillars(room)
    pillars = [
        [
            {
                "Location": c,
                "Range": random.randrange(100, 250),
                "NoiseRange": random.randrange(100, 250),
            }
            for c in p
        ]
        for p in pillars
    ]
    return pillars
    # pillars.append(
    #     [
    #         {
    #             "Location": p,
    #             "Range": random.randrange(100, 250),
    #             "NoiseRange": random.randrange(100, 250),
    #         }
    #         for p in formation
    #     ]
    # )
