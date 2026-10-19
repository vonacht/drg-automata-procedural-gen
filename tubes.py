import math, random
import numpy as np

# a tube = a 2-point polyline: [{"center": a, "radius": ra}, {"center": b, "radius": rb}]


def _rot_z(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


def kaleidoscope_room(
    rng=random,
    n_fold=1,
    seeds=None,
    mirror=False,
    size=6000.0,
    radius=0.1,
    extra_bridges=1,
):
    """Random tubes in one wedge, repeated n-fold around the vertical axis."""
    n_fold = n_fold or rng.randint(3, 8)
    seeds = seeds or rng.randint(2, 4)
    mirror = rng.random() < 0.5 if mirror is None else mirror
    r0 = radius * size
    wedge = 2 * math.pi / n_fold

    def pt():  # a point inside the wedge, so the copies tile the full turn
        ang, rad = rng.uniform(0, wedge), rng.uniform(0.15, 1.0) * size / 2
        return np.array(
            [
                rad * math.cos(ang),
                rad * math.sin(ang),
                rng.uniform(-0.3, 0.3) * size,
            ]
        )

    base = [
        (pt(), pt(), rng.uniform(0.7, 1.3) * r0, rng.uniform(0.7, 1.3) * r0)
        for _ in range(seeds)
    ]
    tubes = []
    for k in range(n_fold):
        for a, b, ra, rb in base:
            for flip in ((1, -1) if mirror else (1,)):
                A = _rot_z(a * np.array([1, flip, 1]), k * wedge)
                B = _rot_z(b * np.array([1, flip, 1]), k * wedge)
                tubes.append(
                    [
                        {"center": tuple(map(float, A)), "radius": ra},
                        {"center": tuple(map(float, B)), "radius": rb},
                    ]
                )
    tubes = connect_tubes(tubes, extra=extra_bridges)
    return tubes, [[0, 1] for _ in tubes]


def _closest_on_segments(p1, q1, p2, q2):
    """Closest points between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    denom = a * e - b * b
    s = float(np.clip((b * f - c * e) / denom, 0, 1)) if denom > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t, s = 0.0, float(np.clip(-c / a, 0, 1))
    elif t > 1:
        t, s = 1.0, float(np.clip((b - c) / a, 0, 1))
    return p1 + d1 * s, p2 + d2 * t


def connect_tubes(tubes, extra=1, bridge_scale=0.6):
    """Returns the tubes plus bridging tubes that make the room one connected piece,
    plus `extra` more bridges between the nearest already-linked tubes (loops).
    """
    segs = [
        (
            np.array(t[0]["center"]),
            np.array(t[1]["center"]),
            (t[0]["radius"] + t[1]["radius"]) / 2,
        )
        for t in tubes
    ]
    n = len(segs)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = _closest_on_segments(
                segs[i][0], segs[i][1], segs[j][0], segs[j][1]
            )
            gap = float(np.linalg.norm(a - b)) - (
                segs[i][2] + segs[j][2]
            )  # < 0: touching
            pairs.append((gap, i, j, a, b))
    pairs.sort(key=lambda x: x[0])

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bridges, spare = [], []
    for gap, i, j, a, b in pairs:
        ri, rj = find(i), find(j)
        if gap <= 0:
            parent[ri] = rj  # already touching
        elif ri != rj:
            parent[ri] = rj
            bridges.append((i, j, a, b))  # joins two separate pieces
        else:
            spare.append((i, j, a, b))  # nearest unlinked pairs become loops
    for i, j, a, b in bridges + spare[:extra]:
        r = bridge_scale * min(segs[i][2], segs[j][2])
        tubes = tubes + [
            [
                {"center": tuple(map(float, a)), "radius": r},
                {"center": tuple(map(float, b)), "radius": r},
            ]
        ]
    return tubes
