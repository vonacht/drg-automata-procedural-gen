import math, random


def _lerp(a, b, t):
    return a + (b - a) * t


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ---- axis paths: t in [0,1] -> (x, y) offset of the shaft's center line ----
def _axis_straight(rng, reach):
    return lambda t: (0.0, 0.0)


def _axis_lean(rng, reach):
    a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0) * reach
    return lambda t: (math.cos(a) * d * t, math.sin(a) * d * t)


def _axis_s_curve(rng, reach):
    a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0) * reach
    return lambda t: (
        math.cos(a) * d * math.sin(2 * math.pi * t),
        math.sin(a) * d * math.sin(2 * math.pi * t),
    )


def _axis_dogleg(rng, reach):  # straight, a sideways jog, straight again
    a, d, t0 = (
        rng.uniform(0, 2 * math.pi),
        rng.uniform(0.7, 1.2) * reach,
        rng.uniform(0.3, 0.7),
    )
    return lambda t: (
        math.cos(a) * d * _smooth((t - t0) / 0.2 + 0.5),
        math.sin(a) * d * _smooth((t - t0) / 0.2 + 0.5),
    )


# ---- radius profiles: t in [0,1] -> multiplier of the base radius ----
def _prof_constant(rng):
    return lambda t: 1.0


def _prof_funnel(rng):
    a, b = rng.choice([(0.45, 1.4), (1.4, 0.45)])
    return lambda t: _lerp(a, b, t)


def _prof_hourglass(rng):
    k, t0 = rng.uniform(0.4, 0.65), rng.uniform(0.35, 0.65)
    return lambda t: 1 - k * math.exp(-(((t - t0) / 0.18) ** 2))


def _prof_bottle(rng):  # narrow neck, then a wide chamber at the end
    neck, belly = rng.uniform(0.4, 0.6), rng.uniform(1.3, 1.7)
    return lambda t: _lerp(neck, belly, _smooth((t - 0.65) / 0.2))


def _prof_beaded(rng):
    k, amp = rng.randint(2, 5), rng.uniform(0.3, 0.55)
    return lambda t: 1 - amp * (0.5 + 0.5 * math.cos(2 * math.pi * k * t))


def _prof_wobble(rng):
    p1, p2 = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
    return (
        lambda t: 1
        + 0.25 * math.sin(7 * t + p1)
        + 0.15 * math.sin(13 * t + p2)
    )


AXES = [_axis_straight, _axis_lean, _axis_s_curve, _axis_dogleg]
AXIS_WEIGHTS = [3, 2, 2, 2]  # keep plain shafts common enough to be a contrast
PROFILES = [
    _prof_constant,
    _prof_funnel,
    _prof_hourglass,
    _prof_bottle,
    _prof_beaded,
    _prof_wobble,
]


# ---- features: each returns a list of extra polylines ----
def _at(shaft, t):
    """Center and radius of the shaft at fraction t (interpolated)."""
    f = t * (len(shaft) - 1)
    i = min(int(f), len(shaft) - 2)
    u = f - i
    a, b = shaft[i], shaft[i + 1]
    c = tuple(_lerp(p, q, u) for p, q in zip(a["center"], b["center"]))
    return c, _lerp(a["radius"], b["radius"], u)


def _feat_alcoves(shaft, height, rng):  # resting spots on alternating sides
    out, a = [], rng.uniform(0, 2 * math.pi)
    for t in sorted(
        rng.sample((0.2, 0.35, 0.5, 0.65, 0.8), rng.randint(2, 3))
    ):
        (cx, cy, cz), r = _at(shaft, t)
        a += math.pi + rng.uniform(-0.6, 0.6)
        out.append(
            [
                {
                    "center": (
                        cx + math.cos(a) * r * k,
                        cy + math.sin(a) * r * k,
                        cz,
                    ),
                    "radius": r * rr,
                }
                for k, rr in ((0.9, 0.45), (1.5, 0.4), (2.1, 0.5))
            ]
        )
    return out


def _feat_exit(
    shaft, height, rng
):  # a side tunnel leaving the shaft, sloping down
    (cx, cy, cz), r = _at(shaft, rng.uniform(0.15, 0.85))
    a = rng.uniform(0, 2 * math.pi)
    return [
        [
            {
                "center": (
                    cx + math.cos(a) * r * (0.9 + 0.8 * k),
                    cy + math.sin(a) * r * (0.9 + 0.8 * k),
                    cz - 0.2 * r * k,
                ),
                "radius": r * 0.4,
            }
            for k in range(5)
        ]
    ]


def _feat_landing(shaft, height, rng):  # a ring of chambers around one end
    (cx, cy, cz), r = _at(shaft, rng.choice((0.0, 1.0)))
    k, a0 = rng.randint(4, 6), rng.uniform(0, 2 * math.pi)
    return [
        [
            {
                "center": (
                    cx + math.cos(a0 + 2 * math.pi * i / k) * r * 0.9,
                    cy + math.sin(a0 + 2 * math.pi * i / k) * r * 0.9,
                    cz,
                ),
                "radius": r * rng.uniform(0.8, 1.2),
            }
            for i in range(k)
        ]
    ]


def _feat_helix(shaft, height, rng):  # your spiral ramp, following the shaft
    from main import sample_helix

    turns, hand = rng.choice((1, 2, 3)), rng.choice((-1, 1))
    r0 = _at(shaft, 0.5)[1]
    out = []
    for p in sample_helix(r0 * 1.3, height, turns, 20 * turns):
        t = min(max(p[2] / height, 0.0), 1.0)
        (cx, cy, _), r = _at(shaft, t)
        out.append(
            {"center": (cx + p[0], cy + hand * p[1], p[2]), "radius": r * 0.5}
        )
    return [out]


FEATURES = [_feat_alcoves, _feat_exit, _feat_landing, _feat_helix]


def _make_axis_and_profile(rng, radius):
    axis = rng.choices(AXES, AXIS_WEIGHTS)[0](rng, reach=radius * 2.5)
    prof = rng.choice(PROFILES)(rng)
    if rng.random() < 0.5:  # flip end for end
        p0 = prof
        prof = lambda t: p0(1 - t)
    return axis, prof


def _topo_single(radius, height, n, rng):
    axis, prof = _make_axis_and_profile(rng, radius)
    shaft = []
    for i in range(n):
        t = i / (n - 1)
        x, y = axis(t)
        shaft.append(
            {"center": (x, y, t * height), "radius": radius * prof(t)}
        )
    return [shaft]


def _topo_twin(radius, height, n, rng):
    """Two shafts that start together, part in the middle, and rejoin."""
    axis, prof = _make_axis_and_profile(rng, radius)
    a = rng.uniform(0, 2 * math.pi)
    ux, uy = math.cos(a), math.sin(a)
    sep = radius * rng.uniform(1.6, 2.6)  # max distance from the axis
    scales = rng.choice(
        [(0.7, 0.7), (0.85, 0.5)]
    )  # even twins, or one dominant

    polys = []
    for side, sc in zip((-1, 1), scales):
        poly = []
        for i in range(n):
            t = i / (n - 1)
            x, y = axis(t)
            bump = math.sin(math.pi * t)  # 0 at both ends, 1 in the middle
            off = side * sep * bump
            r = (
                radius * prof(t) * (sc + (1 - sc) * (1 - bump))
            )  # full width at the ends
            poly.append(
                {
                    "center": (x + ux * off, y + uy * off, t * height),
                    "radius": r,
                }
            )
        polys.append(poly)

    # optional crossover: a short passage joining the two shafts
    if rng.random() < 0.6:
        i = round(rng.uniform(0.3, 0.7) * (n - 1))
        (ax, ay, az), (bx, by, bz) = (
            polys[0][i]["center"],
            polys[1][i]["center"],
        )
        polys.append(
            [
                {
                    "center": (
                        _lerp(ax, bx, u),
                        _lerp(ay, by, u),
                        _lerp(az, bz, u),
                    ),
                    "radius": radius * 0.35,
                }
                for u in (0.0, 0.5, 1.0)
            ]
        )
    return polys


def _topo_stepped(radius, height, rng):
    """A staircase of landings, each offset sideways from the last, joined by short drops."""
    k = min(7, max(3, round(height / (radius * rng.uniform(0.9, 1.4))) + 1))
    w = [rng.uniform(0.5, 1.5) for _ in range(k - 1)]
    drops = [height * v / sum(w) for v in w]  # uneven steps, total = height
    zig = rng.random() < 0.5
    base = rng.uniform(0, 2 * math.pi)
    heading = base
    x = y = 0.0
    z = height  # start at the top, descend

    poly = []
    for i in range(k):
        poly.append(
            {"center": (x, y, z), "radius": radius * rng.uniform(0.8, 1.1)}
        )
        if i == k - 1:
            break
        if zig:  # alternate left and right of a base heading
            heading = base + (0.7 if i % 2 else -0.7) * rng.uniform(0.7, 1.3)
        else:  # keep turning the same way: a spiral staircase
            heading += rng.uniform(0.5, 1.0)
        d = radius * rng.uniform(1.4, 2.0)
        nx, ny = x + math.cos(heading) * d, y + math.sin(heading) * d
        # narrow connector midway, lower than the landing: the "drop"
        poly.append(
            {
                "center": ((x + nx) / 2, (y + ny) / 2, z - 0.6 * drops[i]),
                "radius": radius * 0.6,
            }
        )
        x, y, z = nx, ny, z - drops[i]
    return [poly]


TOPOLOGY_WEIGHTS = {"single": 5, "twin": 0, "stepped": 2}
FEATURES_BY_TOPOLOGY = {
    "single": [_feat_alcoves, _feat_exit, _feat_landing, _feat_helix],
    "twin": [_feat_exit, _feat_landing],
    "stepped": [_feat_exit, _feat_landing],
}


def generate_pit_room(max_height, rng=random):
    from room_trees import shift_room_multiple

    radius = rng.randrange(700, 1200)
    height = rng.uniform(1500, max_height)
    n = min(14, max(4, round(height / (radius * rng.uniform(0.9, 1.3)))))

    topo = rng.choices(
        list(TOPOLOGY_WEIGHTS), list(TOPOLOGY_WEIGHTS.values())
    )[0]
    if topo == "single":
        polys = _topo_single(radius, height, n, rng)
    elif topo == "twin":
        polys = _topo_twin(radius, height, n, rng)
    else:
        polys = _topo_stepped(radius, height, rng)

    pool = FEATURES_BY_TOPOLOGY[topo]
    count = min(rng.choice((0, 1, 1, 2)), len(pool))
    for feat in rng.sample(pool, count):
        polys += feat(polys[0], height, rng)

    paths, i = [], 0
    for poly in polys:
        paths.append(list(range(len(poly))))
        i += len(poly)

    new_room = shift_room_multiple(polys, (0, 0, height / 2))
    return new_room, paths
