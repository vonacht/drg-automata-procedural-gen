import math, random
import numpy as np
from types import SimpleNamespace

UP = np.array([0.0, 0.0, 1.0])

FLOOR_SHAPE = "CYLINDER"


def _fd(s):
    return min(s.get("floor_depth", 0.0), 0.0)


def _halfball_extent(c, r, fd, p, d):
    """The same for one chamber alone (c is its true center)."""
    lo, hi = _sphere_intervals(
        np.array([c], dtype=float), np.array([r]), np.array([fd]), p, d
    )
    return (min(lo[0], 0.0), max(hi[0], 0.0)) if lo[0] < hi[0] else (0.0, 0.0)


def _sphere_intervals(C, R, FD, p, d):
    """For each sphere, the interval [lo, hi] of t where p + t*d is in its open space
    (lo=inf, hi=-inf if none): the dome plus, below the center plane, the wall down to the floor.
    """
    oc = p - C
    b = oc @ d
    disc = b * b - ((oc * oc).sum(1) - R * R)
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    lo, hi = -b - sq, -b + sq  # the ball
    floor = C[:, 2] + (
        0.0 if FLOOR_SHAPE == "cylinder" else FD
    )  # where the ball part starts
    if abs(d[2]) > 1e-12:
        tz = (floor - p[2]) / d[2]
        lo, hi = (
            (np.maximum(lo, tz), hi) if d[2] > 0 else (lo, np.minimum(hi, tz))
        )
    else:
        ok &= p[2] >= floor
    ok &= lo < hi
    lo, hi = np.where(ok, lo, np.inf), np.where(ok, hi, -np.inf)
    if FLOOR_SHAPE == "cylinder":  # the straight wall below the dome
        a = d[0] ** 2 + d[1] ** 2
        ocx = oc[:, :2]
        cc = (ocx**2).sum(1) - R * R
        if a > 1e-12:
            bb = ocx @ d[:2]
            disc2 = bb * bb - a * cc
            ok2 = disc2 > 0
            sq2 = np.sqrt(np.where(ok2, disc2, 0.0))
            lo2, hi2 = (-bb - sq2) / a, (-bb + sq2) / a
        else:
            ok2 = cc < 0
            lo2, hi2 = np.full(len(R), -np.inf), np.full(len(R), np.inf)
        zlo, zhi = C[:, 2] + FD, C[:, 2]
        if abs(d[2]) > 1e-12:
            ta, tb = (zlo - p[2]) / d[2], (zhi - p[2]) / d[2]
            lo2, hi2 = np.maximum(lo2, np.minimum(ta, tb)), np.minimum(
                hi2, np.maximum(ta, tb)
            )
        else:
            ok2 &= (p[2] >= zlo) & (p[2] <= zhi)
        ok2 &= (lo2 < hi2) & (FD < 0)
        lo, hi = np.where(ok2, np.minimum(lo, lo2), lo), np.where(
            ok2, np.maximum(hi, hi2), hi
        )
    return lo, hi


def _union_extent(C, R, FD, p, d):
    """[t_lo, t_hi] around 0 where p + t*d is inside any of the spheres, following the ray
    through every sphere it enters."""
    lo, hi = _sphere_intervals(C, R, FD, p, d)
    ok = lo < hi
    lo, hi = lo[ok], hi[ok]
    t_hi = 0.0
    while True:
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


# ------------------------------- the formations --------------------------------
def _ring(c, R, rng, ctx):
    """Pillars on a circle: straight, leaning in or out, twisted (a cooling-tower
    hyperboloid), or alternately twisted both ways (a crossing lattice)."""
    r_p = ctx.r_p
    rho = rng.uniform(0.45, 0.7) * R
    n = min(rng.randint(5, 10), int(2 * math.pi * rho / (2.5 * r_p)))
    if n < 3:
        return []
    style = rng.choice(("straight", "lean_in", "lean_out", "twist", "lattice"))
    tau = (
        rng.uniform(0.15, 0.35)
        if style.startswith("lean")
        else rng.uniform(0.2, 0.45)
    )
    th0, out = rng.uniform(0, 2 * math.pi), []
    for k in range(n):
        th = th0 + 2 * math.pi * k / n
        radial = np.array([math.cos(th), math.sin(th), 0.0])
        tang = np.array([-math.sin(th), math.cos(th), 0.0])
        if style == "straight":
            d = UP
        elif style == "lean_in":
            d = UP * math.cos(tau) - radial * math.sin(tau)
        elif style == "lean_out":
            d = UP * math.cos(tau) + radial * math.sin(tau)
        elif style == "twist":
            d = UP * math.cos(tau) + tang * math.sin(tau)
        else:
            d = UP * math.cos(tau) + tang * math.sin(tau) * (
                1 if k % 2 == 0 else -1
            )
        pil = _pillar(c, R, c + radial * rho, d, r_p, rng, ctx)
        if pil:
            out.append(pil)
    if rng.random() < 0.3:  # sometimes a fat pillar in the middle
        pil = _pillar(c, R, c.copy(), UP, r_p * 1.6, rng, ctx)
        if pil:
            out.append(pil)
    return out if len(out) >= 3 else []


def _grid(c, R, rng, ctx):
    """A square or staggered (hex) grid of vertical pillars, with some jitter and
    some missing, so it reads as ruins."""
    r_p = ctx.r_p
    sp = max(rng.uniform(0.3, 0.5) * R, 2.6 * r_p)
    hexa = rng.random() < 0.5
    ang = rng.uniform(0, 2 * math.pi)
    ca, sa = math.cos(ang), math.sin(ang)
    jit, miss = rng.uniform(0, 0.12) * sp, rng.uniform(0, 0.25)
    lean = (
        0.0 if rng.random() < 0.8 else rng.uniform(0.1, 0.25)
    )  # rarely, all lean the same way
    phi = rng.uniform(0, 2 * math.pi)
    d = UP * math.cos(lean) + np.array(
        [math.cos(phi), math.sin(phi), 0.0]
    ) * math.sin(lean)
    cells, m = [], int(R / sp) + 2
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            x = i * sp + (0.5 * sp if hexa and j % 2 else 0.0)
            y = j * sp * (0.866 if hexa else 1.0)
            xr, yr = ca * x - sa * y, sa * x + ca * y
            if math.hypot(xr, yr) <= 0.8 * R:
                cells.append((xr, yr))
    if len(cells) < 4:
        return []
    out = []
    for xr, yr in cells:
        if rng.random() < miss and len(cells) > 6:
            continue
        base = c + np.array(
            [xr + rng.uniform(-jit, jit), yr + rng.uniform(-jit, jit), 0.0]
        )
        pil = _pillar(c, R, base, d, r_p, rng, ctx)
        if pil:
            out.append(pil)
    return out if len(out) >= 3 else []


def _forest(c, R, rng, ctx):
    """A sunflower (golden-angle) scatter: even but organic, with a mix of full
    pillars, stalagmites and stalactites of different widths."""
    r_p = ctx.r_p
    N = min(rng.randint(8, 24), int((0.8 * R / (1.7 * r_p)) ** 2))
    if N < 6:
        return []
    cc = 0.8 * R / math.sqrt(N)
    golden, rot, out = (
        math.pi * (3 - math.sqrt(5)),
        rng.uniform(0, 2 * math.pi),
        [],
    )
    for k in range(1, N + 1):
        r, th = cc * math.sqrt(k), rot + k * golden
        base = c + np.array([r * math.cos(th), r * math.sin(th), 0.0])
        tilt, az = rng.uniform(0, 0.08), rng.uniform(0, 2 * math.pi)
        d = UP * math.cos(tilt) + np.array(
            [math.cos(az), math.sin(az), 0.0]
        ) * math.sin(tilt)
        kind = rng.choices(("full", "floor", "ceiling"), (0.45, 0.35, 0.2))[0]
        pil = _pillar(
            c,
            R,
            base,
            d,
            max(r_p * rng.uniform(0.75, 1.3), ctx.min_radius),
            rng,
            ctx,
            None if kind == "full" else kind,
            rng.uniform(0.25, 0.7),
        )
        if pil:
            out.append(pil)
    return out if len(out) >= 4 else []


def _henge(c, R, rng, ctx):
    """A ring of short thick stumps with lintels laid across their tops."""
    r_p = ctx.r_p * 1.3
    rho = rng.uniform(0.45, 0.7) * R
    n = min(rng.randint(5, 9), int(2 * math.pi * rho / (2.6 * r_p)))
    if n < 4:
        return []
    h, th0, m = (
        rng.uniform(0.25, 0.45) * R,
        rng.uniform(0, 2 * math.pi),
        ctx.margin * R,
    )
    out, tops = [], []
    for k in range(n):
        th = th0 + 2 * math.pi * k / n
        x, y = c[0] + rho * math.cos(th), c[1] + rho * math.sin(th)
        npts = max(3, int(math.ceil((h + m) / (ctx.spacing * r_p))) + 1)
        t = np.linspace(0, 1, npts)
        z = c[2] - m + (h + m) * t
        r = np.maximum(
            r_p * (1 + ctx.flare * np.exp(-((t / 0.18) ** 2))), ctx.min_radius
        )
        out.append(
            [
                {
                    "center": (float(x), float(y), float(zz)),
                    "radius": float(rr),
                }
                for zz, rr in zip(z, r)
            ]
        )
        tops.append(np.array([x, y, c[2] + h + 0.5 * r_p]))
    if rng.random() < 0.8:
        rb = max(0.8 * r_p, ctx.min_radius)
        for k in range(n):
            if rng.random() < 0.85:  # a few lintels fallen, for a ruined look
                a, b = tops[k], tops[(k + 1) % n]
                out.append(
                    [
                        {
                            "center": tuple(float(v) for v in a),
                            "radius": float(rb),
                        },
                        {
                            "center": tuple(float(v) for v in b),
                            "radius": float(rb),
                        },
                    ]
                )
    return out


def _colonnade(c, R, rng, ctx):
    """Two parallel rows of pillars with an aisle between them, like a hall."""
    r_p = ctx.r_p
    ang = rng.uniform(0, 2 * math.pi)
    u = np.array([math.cos(ang), math.sin(ang), 0.0])
    v = np.array([-math.sin(ang), math.cos(ang), 0.0])
    w = max(rng.uniform(0.3, 0.55) * R, 3.2 * r_p)  # aisle width
    half = rng.uniform(0.5, 0.75) * R
    m = min(rng.randint(3, 6), int(2 * half / (2.6 * r_p)) + 1)
    if m < 3:
        return []
    out = []
    for s in np.linspace(-half, half, m):
        for side in (-0.5, 0.5):
            base = c + u * s + v * (w * side)
            if math.hypot(*(base - c)[:2]) > 0.82 * R:
                continue
            pil = _pillar(c, R, base, UP, r_p, rng, ctx)
            if pil:
                out.append(pil)
    return out if len(out) >= 4 else []


FORMATIONS = {
    "ring": (_ring, 3.0),
    "grid": (_grid, 2.0),
    "forest": (_forest, 2.0),
    "henge": (_henge, 1.5),
    "colonnade": (_colonnade, 1.5),
}


def _make_ctx(
    rng, R, min_radius, thickness, world, through, reach, center, fd
):
    return SimpleNamespace(
        min_radius=min_radius,
        margin=0.06,
        min_len=0.3,
        spacing=1.4,
        flare=rng.uniform(0.2, 0.45),
        waist=rng.uniform(0.0, 0.2),
        tip=rng.uniform(0.4, 0.7),
        r_p=max(rng.uniform(*thickness) * R, 1.05 * min_radius),
        world=world,
        through=through,
        reach=reach,
        center=center,
        fd=fd,
    )


def _pillar(c, R, base, d, r_p, rng, ctx, partial=None, frac=0.5):
    """One pillar standing at `base` (a floor point), along direction d.
    partial: None = floor to ceiling, "floor" = a stalagmite, "ceiling" = a stalactite.
    """
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    p = base + d * (0.02 * R)
    lo, hi = _halfball_extent(ctx.center, R, ctx.fd, p, d)
    if ctx.world is not None and ctx.through != "none":
        ulo, uhi = _union_extent(ctx.world.C, ctx.world.R, ctx.world.FD, p, d)
        reach = ctx.reach * R  # how far past its own chamber it may run
        if ctx.through in ("up", "both"):
            hi = max(
                hi, min(uhi, hi + reach)
            )  # keep going into a sphere above
        if ctx.through in ("down", "both"):
            lo = min(lo, max(ulo, lo - reach))  # or into one below
    L = hi - lo
    if L < ctx.min_len * R:
        return None
    A, B = (
        p + d * lo,
        p + d * hi,
    )  # where the line meets the floor and the ceiling
    m = ctx.margin * R
    if partial == "floor":
        s, e = A - d * m, A + d * L * frac
    elif partial == "ceiling":
        s, e = B + d * m, B - d * L * frac
    else:
        s, e = A - d * m, B + d * m
    n = max(3, int(math.ceil(np.linalg.norm(e - s) / (ctx.spacing * r_p))) + 1)
    t = np.linspace(0.0, 1.0, n)
    P = s + (e - s) * t[:, None]
    prof = 1 + ctx.flare * np.exp(-((t / 0.18) ** 2))
    if partial is None:
        prof = (
            prof
            + ctx.flare * np.exp(-(((1 - t) / 0.18) ** 2))
            - ctx.waist * np.sin(np.pi * t)
        )
    else:
        prof = prof * (1 - (1 - ctx.tip) * t)
    prof = prof * (1 + 0.04 * np.array([rng.uniform(-1, 1) for _ in t]))
    r = np.maximum(r_p * prof, ctx.min_radius)
    return [
        {"center": tuple(float(x) for x in q), "radius": float(rr)}
        for q, rr in zip(P, r)
    ]


def pillar_formations(
    cave,
    rng=random,
    big_radius=1500,
    n_big=None,
    kinds=None,
    thickness=(0.06, 0.12),
    min_radius=100,
    through="up",
    reach=1.0,
    return_names=False,
):
    """
    Builds a pillar formation in some of the cave's big chambers.

    big_radius: chambers at least this large get formations
    n_big:      how many big chambers to fill (None = random 1-2; larger ones more likely)
    kinds:      subset of FORMATIONS to allow (None = all)
    thickness:  pillar radius as a fraction of the chamber radius (never below min_radius)
    Returns a list of pillars (each a list of {"center", "radius"} dicts), or
    (formation name, pillars) pairs with return_names=True. Empty if nothing fits.
    through: where a pillar may continue past its own chamber into overlapping spheres:
             "up" (a chamber above the dome), "down", "both", or "none" (the old behavior)
    reach:   the most it may extend past its chamber, as a fraction of the chamber's radius,
             so it can't run through a whole stack of chambers
    (the other arguments are as before)
    """
    spheres = [s for poly in cave for s in poly]
    bigs = [s for s in spheres if s["radius"] >= big_radius]
    if not bigs:
        return []
    world = SimpleNamespace(
        C=np.array([s["center"] for s in spheres], dtype=float),
        R=np.array([s["radius"] for s in spheres], dtype=float),
        FD=np.array([_fd(s) for s in spheres]),
    )
    pool, chosen = list(bigs), []
    for _ in range(
        min(rng.randint(0, 2) if n_big is None else n_big, len(pool))
    ):
        s = rng.choices(pool, weights=[x["radius"] for x in pool])[0]
        chosen.append(s)
        pool = [x for x in pool if x is not s]

    names, out = list(kinds or FORMATIONS), []
    for s in chosen:
        center, R, fd = (
            np.array(s["center"], dtype=float),
            float(s["radius"]),
            _fd(s),
        )
        c = center.copy()
        c[2] += fd  # the formations' base height is the floor
        for name in sorted(
            names,
            key=lambda k: rng.random() ** (1 / FORMATIONS[k][1]),
            reverse=True,
        ):
            ctx = _make_ctx(
                rng,
                R,
                min_radius,
                thickness,
                world,
                through,
                reach,
                center,
                fd,
            )
            pillars = FORMATIONS[name][0](c, R, rng, ctx)
            if pillars:
                out.append((name, pillars))
                break
    return out if return_names else [p for _, ps in out for p in ps]


# ---------------------------------- the entry point ----------------------------


def generate_big_pillars(room):

    pillars_1 = pillar_formations(room, big_radius=1800)
    pillars = [
        [
            {
                "Location": c["center"],
                "Range": c["radius"],
                "NoiseRange": random.randrange(100, 250),
            }
            for c in p
        ]
        for p in pillars_1
    ]
    if pillars:
        print("big_pillars")
    # if pillars_2:
    #     print("really big pillars")
    return pillars
