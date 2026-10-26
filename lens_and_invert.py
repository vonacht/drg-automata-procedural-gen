import math, random
import numpy as np


def _random_unit(rng):  # same helper as in rotate_half
    v = np.array([rng.gauss(0, 1) for _ in range(3)])
    return v / np.linalg.norm(v)


def _extract_room(room):
    groups, path = room
    spheres = [s for g in groups for s in g]
    C = np.array([s["center"] for s in spheres], dtype=float)
    R = np.array([s["radius"] for s in spheres], dtype=float)
    return groups, path, spheres, C, R


def _rebuild_room(groups, path, centers, radii=None):
    result, i = [], 0
    for g in groups:
        out = []
        for s in g:
            d = {**s, "center": tuple(float(x) for x in centers[i])}
            if radii is not None:
                if float(radii[i]) > 400:
                    d["radius"] = float(radii[i])
                else:
                    d["radius"] = 450
            out.append(d)
            i += 1
        result.append(out)
    return result, [list(p) for p in path]


# ================================== lens ======================================
def lens(
    room,
    rng=random,
    mode="power",
    strength=None,
    reach=None,
    center=None,
    aspect=None,
    soften=0.2,
    scale_radii=True,
    preserve_volume=True,
):
    """
    Radial magnification or compression around a center, fading to nothing at the rim.

    mode:      "power"  - r' = R * (r/R)**gamma; strong, and never folds
               "smooth" - the earlier (1 + k * (1-d)**2) lens; subtle
    strength:  power mode: gamma. Below 1 magnifies (0.25 is extreme), above 1 compresses
               (3 is extreme). smooth mode: k, from -0.95 to 2.9.
               None = random strong value.
    reach:     lens radius as a fraction of the room's largest extent (None = random 0.4-0.85)
    center:    (x, y, z), or None for the center of a random sphere
    aspect:    (ax, ay, az) turns the lens into an ellipsoid; None = random
    soften:    power mode: share of the radius used to blend into the untouched
               outside, to avoid a crease at the rim (0 = hard edge)
    scale_radii:     scale each sphere by the local magnification, so magnified
                     chambers look bigger and compressed ones tighter
    preserve_volume: then rescale to keep the total volume (keeps the op free)
    """
    groups, path, spheres, C, R = _extract_room(room)
    if len(C) < 2:
        return _rebuild_room(groups, path, C)
    extent = np.ptp(C, axis=0).max()
    c = (
        C[rng.randrange(len(C))]
        if center is None
        else np.asarray(center, dtype=float)
    )
    reach = rng.uniform(0.4, 0.85) if reach is None else reach
    a = np.array(
        (
            aspect
            if aspect is not None
            else rng.choice(
                [(1, 1, 1), (1, 1, 1), (1.4, 1.4, 0.6), (0.7, 0.7, 1.6)]
            )
        ),
        dtype=float,
    )

    rel = C - c
    d = np.linalg.norm(rel / a, axis=1) / (
        reach * extent
    )  # 0 at the center, 1 at the rim
    inside = d < 1

    if mode == "power":
        if strength is None:
            strength = (
                rng.uniform(0.25, 0.6)
                if rng.random() < 0.6
                else rng.uniform(1.6, 3.0)
            )
        m = np.maximum(d, 1e-3) ** (strength - 1)  # r' = r * m
        if soften > 0:
            u = np.clip((d - (1 - soften)) / soften, 0, 1)
            m = (1 - u * u * (3 - 2 * u)) * m + u * u * (3 - 2 * u)
        m = np.where(inside, m, 1.0)
    else:
        k = (
            strength
            if strength is not None
            else (
                rng.uniform(0.6, 2.0)
                if rng.random() < 0.6
                else -rng.uniform(0.4, 0.8)
            )
        )
        k = min(max(k, -0.95), 2.9)
        m = 1 + k * np.where(inside, (1 - d) ** 2, 0.0)

    new = c + rel * m[:, None]
    radii = None
    if scale_radii:
        radii = R * np.clip(m, 0.25, 4.0)
        if preserve_volume:
            radii = radii * (np.sum(R**3) / np.sum(radii**3)) ** (1 / 3)
    return _rebuild_room(groups, path, new, radii)


# ================================ inversion ===================================
def invert(
    room,
    rng=random,
    center_distance=1.6,
    scale_radii=True,
    preserve_volume=True,
    min_dist=0.15,
):
    """
    Inversion in a sphere centered outside the room. The map is conformal, so
    local shapes survive while straight runs bend into arcs and near and far
    swap (the room comes out mirrored and distorted).

    center_distance: how far the inversion center is from the room's middle, in
                     units of its extent. This is the intensity dial: ~0.4 is
                     extreme, ~1 strong, 2+ nearly a plain mirror image.
                     None = random 0.6-1.2.
    scale_radii:     scale each radius by the local magnification, so spheres
                     keep their proportions to the space around them
    preserve_volume: rescale the radii to keep the total volume unchanged,
                     so the op stays free in your cost system
    min_dist:        closest allowed distance to the center, as a fraction of
                     the extent (avoids points flying off to infinity)
    """
    groups, path, spheres, C, R = _extract_room(room)
    if len(C) < 2:
        return _rebuild_room(groups, path, C)
    extent = np.ptp(C, axis=0).max()
    mid = (C.min(0) + C.max(0)) / 2
    dist = (
        rng.uniform(0.6, 1.2) if center_distance is None else center_distance
    )
    c = mid + _random_unit(rng) * dist * extent

    rel = C - c
    D = np.maximum(np.linalg.norm(rel, axis=1), min_dist * extent)
    rho2 = np.median(D) ** 2  # magnification ~1 at the median distance
    mag = rho2 / D**2  # local magnification factor
    new = c + rel * mag[:, None]

    # refit to the original size and place, since the inversion's overall scale is arbitrary
    new_mid = (new.min(0) + new.max(0)) / 2
    fit = extent / max(np.ptp(new, axis=0).max(), 1e-9)
    new = mid + (new - new_mid) * fit

    radii = None
    if scale_radii:
        radii = R * mag * fit
        if preserve_volume:
            radii = radii * (np.sum(R**3) / np.sum(radii**3)) ** (1 / 3)
    return _rebuild_room(groups, path, new, radii)


def _nz(x, eps=1e-3):
    return np.where(np.abs(x) < eps, eps, x)


def _udir(a):
    return complex(math.cos(a), math.sin(a))


def _edge_point(z, zmid, half, d, gap):
    """A point just beyond the room's edge in direction d (a unit complex number).
    Every sphere is at least gap * half away from it."""
    t_edge = np.max(((z - zmid) * np.conj(d)).real)
    return zmid + d * (t_edge + gap * half)


# ---- the maps: u = complex position relative to the origin, in units of half the room ----
def _make_exp(rng, w, thick=None, sweep=None):
    thick = rng.uniform(0.3, 0.8) if thick is None else thick
    sweep = (
        rng.uniform(2.5, 6.0) if sweep is None else min(sweep, 6.2)
    )  # stays under a full turn
    return lambda u: np.exp(u.real * thick + 1j * u.imag * sweep / 2)


def _make_log(rng, w):
    rot = _udir(rng.uniform(0, 2 * math.pi))
    return lambda u: np.log(np.maximum(np.abs(u * rot), 0.05)) + 1j * np.angle(
        u * rot
    )


def _make_power(rng, w, p, axis_hat):
    return lambda u: np.power(
        u * np.conj(axis_hat), p
    )  # room centered on the +real axis: no seam


def _make_joukowski(rng, w):
    c = 0.9 * np.abs(w).min()  # circle just inside the nearest sphere
    return lambda u: u + c * c / u


def _make_mobius(rng, w, p2, rho=None, phi=None):
    p1 = (
        0.5 * math.sqrt(rng.random()) * _udir(rng.uniform(0, 2 * math.pi))
    )  # attracting point, in the room
    k = (rng.uniform(0.5, 1.6) if rho is None else rho) * _udir(
        rng.uniform(-1.2, 1.2) if phi is None else phi
    )

    def f(u):  # p2, the repelling point, is outside the room
        u2 = k * (u - p1) / _nz(u - p2)
        return (p1 - p2 * u2) / _nz(1 - u2)

    return f


def _make_sin(rng, w):
    rot = _udir(rng.uniform(0, 2 * math.pi))
    s = (
        rng.uniform(0.6, 1.0)
        * (math.pi / 2)
        / max(np.abs((w * rot).real).max(), 1e-9)
    )
    return lambda u: np.sin(u * rot * s)  # |Re| < pi/2: one-to-one


_MAKERS = {
    "exp": _make_exp,
    "log": _make_log,
    "power": _make_power,
    "joukowski": _make_joukowski,
    "mobius": _make_mobius,
    "sin": _make_sin,
}
_PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def complex_map(
    room,
    rng=random,
    kind=None,
    plane="xy",
    params=None,
    alpha=0.5,
    scale_radii=True,
    preserve_volume=True,
):
    """
    Treats two coordinates as a complex number w and replaces it with f(w).

    kind:   "exp", "log", "power", "joukowski", "mobius", "sin" (None = random)
    params: options of the chosen map, e.g. {"sweep": 6.0}, {"p": 3}, {"rho": 1.4, "phi": 0.8}
    alpha:  how much of the local stretch the radii follow (0 = none, 1 = all);
            0.5 keeps the spheres in proportion without letting a few dominate
    """
    groups, path, spheres, C, R = _extract_room(room)
    if len(C) < 2:
        return _rebuild_room(groups, path, C)
    kind = kind or rng.choice(list(_MAKERS))
    prm = dict(params or {})
    i, j = _PLANES[plane]
    xy = C[:, [i, j]]
    z = xy[:, 0] + 1j * xy[:, 1]
    mid = (xy.min(0) + xy.max(0)) / 2
    zmid = complex(mid[0], mid[1])
    half = max(np.ptp(xy, axis=0).max(), 1e-9) / 2
    d = _udir(rng.uniform(0, 2 * math.pi))

    extra = {}
    if kind in ("exp", "sin", "mobius"):
        origin = zmid
        if kind == "mobius":
            extra["p2"] = (
                _edge_point(z, zmid, half, d, rng.uniform(0.25, 0.8)) - zmid
            ) / half
    elif kind == "log":
        origin = z[
            rng.randrange(len(z))
        ]  # inside the room: a seam is expected
    elif kind == "joukowski":
        origin = _edge_point(z, zmid, half, d, rng.uniform(0.15, 0.6))
    else:  # power
        p = prm.pop("p", None) or rng.choice((2, 3, 0.5, 1.5))
        gap = 0.3
        for _ in range(14):  # back away until the room fits in a wedge
            origin = _edge_point(z, zmid, half, d, gap)
            axis_hat = (zmid - origin) / abs(zmid - origin)
            span = np.abs(
                np.angle(((z - origin) / half) * np.conj(axis_hat))
            ).max()
            if span < 0.9 * math.pi / p:
                break
            gap *= 1.3
        extra = {"p": p, "axis_hat": axis_hat}

    w = (z - origin) / half
    f = _MAKERS[kind](rng, w, **extra, **prm)

    def F(P):
        out = f(((P[:, 0] + 1j * P[:, 1]) - origin) / half)
        return np.c_[out.real, out.imag]

    new = F(xy)
    h = 1e-3 * half
    mag = (
        np.linalg.norm(F(xy + np.array([h, 0.0])) - new, axis=1) / h
    )  # local stretch
    if not (np.isfinite(new).all() and np.isfinite(mag).all()):
        return _rebuild_room(groups, path, C)

    # robust refit: percentiles instead of min/max, then pull far outliers in smoothly
    lo, hi = np.percentile(new, [2, 98], axis=0)
    fit = 2 * half / max((hi - lo).max(), 1e-9)
    new = mid + (new - (lo + hi) / 2) * fit
    rho = np.linalg.norm(new - mid, axis=1)
    r0, r1 = 1.2 * half, 1.7 * half
    rho2 = np.where(
        rho > r0, r0 + (r1 - r0) * np.tanh((rho - r0) / (r1 - r0)), rho
    )
    new = mid + (new - mid) * (rho2 / np.maximum(rho, 1e-12))[:, None]

    out_C = C.copy()
    out_C[:, [i, j]] = new
    radii = None
    if scale_radii:
        radii = R * np.clip((mag * fit) ** alpha, 0.5, 2.0)
        if preserve_volume:
            radii = radii * (np.sum(R**3) / np.sum(radii**3)) ** (1 / 3)
    return _rebuild_room(groups, path, out_C, radii)


# ---- quality gate -----------------------------------------------------------------
def swallowed_fraction(room):
    """Share of spheres that lie entirely inside a bigger one (hemisphere containment)."""
    C = np.array([s["center"] for g in room[0] for s in g], dtype=float)
    R = np.array([s["radius"] for g in room[0] for s in g], dtype=float)
    if len(C) < 2:
        return 0.0
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    inside = (
        (D + R[:, None] <= R[None])  # ball i inside ball j
        & (
            C[:, None, 2] >= C[None, :, 2] - 1e-9
        )  # floor of i not below floor of j
        & (R[None] > R[:, None] * 1.001)
    )  # j really bigger (ignores duplicates)
    return float(inside.any(axis=1).mean())


def safe_wildcard(op, room, rng=random, tries=6, max_swallowed=0.2, **kw):
    """Runs a wildcard op and rejects NaNs, collapsed or exploded results, and results
    where many spheres got buried inside bigger ones. Returns the original if no try passes.
    """
    _, _, _, C, _ = _extract_room(room)
    extent = np.ptp(C, axis=0).max()
    limit = max(max_swallowed, swallowed_fraction(room) + 0.05)
    for _ in range(tries):
        out = op(room, rng, **kw)
        C2 = np.array([s["center"] for g in out[0] for s in g], dtype=float)
        R2 = np.array([s["radius"] for g in out[0] for s in g], dtype=float)
        if np.isfinite(C2).all() and np.isfinite(R2).all() and (R2 > 0).all():
            if 0.3 * extent <= np.ptp(C2, axis=0).max() <= 3 * extent:
                if swallowed_fraction(out) <= limit:
                    return out
    return room


# ---- the flows: each maps an (N, 3) array of points to their velocities ----
def _lorenz(p, s=10.0, r=28.0, b=8 / 3):
    x, y, z = p.T
    return np.stack([s * (y - x), x * (r - z) - y, x * y - b * z], axis=1)


def _rossler(p, a=0.2, b=0.2, c=5.7):
    x, y, z = p.T
    return np.stack([-y - z, x + a * y, b + z * (x - c)], axis=1)


def _thomas(p, b=0.208186):
    x, y, z = p.T
    return np.stack(
        [np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z], axis=1
    )


def _aizawa(p, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = p.T
    return np.stack(
        [
            (z - b) * x - d * y,
            d * x + (z - b) * y,
            c
            + a * z
            - z**3 / 3
            - (x * x + y * y) * (1 + e * z)
            + f * z * x**3,
        ],
        axis=1,
    )


def _halvorsen(p, a=1.89):
    x, y, z = p.T
    return np.stack(
        [
            -a * x - 4 * y - 4 * z - y * y,
            -a * y - 4 * z - 4 * x - z * z,
            -a * z - 4 * x - 4 * y - x * x,
        ],
        axis=1,
    )


# name: (flow, rough center, rough half-size, characteristic time)
ATTRACTORS = {
    "lorenz": (_lorenz, (0.0, 0.0, 25.0), 22.0, 1.0),
    "rossler": (_rossler, (0.0, 0.0, 3.0), 10.0, 6.0),
    "thomas": (_thomas, (0.0, 0.0, 0.0), 4.0, 6.0),
    "aizawa": (_aizawa, (0.0, 0.0, 0.5), 1.3, 2.0),
    "halvorsen": (_halvorsen, (-1.5, -1.5, -1.5), 5.0, 2.0),
}


def _rk4(f, p, T, steps, lo, hi):
    dt = T / steps
    for _ in range(steps):
        k1 = f(p)
        k2 = f(p + 0.5 * dt * k1)
        k3 = f(p + 0.5 * dt * k2)
        k4 = f(p + dt * k3)
        p = np.clip(
            p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), lo, hi
        )  # keep runaways bounded
    return p


def _random_rotation(rng):
    Q, _ = np.linalg.qr(
        np.array([[rng.gauss(0, 1) for _ in range(3)] for _ in range(3)])
    )
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def attractor_flow(
    room,
    rng=random,
    kind=None,
    duration=None,
    zoom=None,
    z_scale=0.7,
    scale_radii=True,
    preserve_volume=True,
):
    """
    Carries every sphere along the flow of a chaotic attractor.

    kind:     "lorenz", "rossler", "thomas", "aizawa", "halvorsen" (None = random)
    duration: how long to flow, in the attractor's characteristic times
              (None = random 0.3-1.5). Short = gentle swirl and shear,
              medium = stretched and folded, long = collapses onto the
              attractor's own sheets and spirals.
    zoom:     fraction of the attractor's size that the room spans
              (None = random 0.35-1.0); bigger rooms feel more of the dynamics
    z_scale:  multiplies the room's vertical spread afterward (< 1 flattens)
    scale_radii:     scale each radius by the local volume change of the flow
    preserve_volume: then rescale so the total volume is unchanged (keeps the op free)
    """
    groups, path, spheres, C, R = _extract_room(room)
    if len(C) < 2:
        return _rebuild_room(groups, path, C)
    kind = kind or rng.choice(list(ATTRACTORS))
    # print(kind)
    f, ctr, s, tau = ATTRACTORS[kind]
    ctr = np.array(ctr, dtype=float)
    duration = rng.uniform(0.6, 1) if duration is None else duration
    # print(duration)
    zoom = rng.uniform(0.35, 1.0) if zoom is None else zoom

    mid = (C.min(0) + C.max(0)) / 2
    half = max(np.ptp(C, axis=0).max(), 1e-9) / 2
    Q = _random_rotation(rng)  # which way the room sits in the flow
    place = (
        ctr + _random_unit(rng) * rng.uniform(0, 0.3) * s
    )  # where in the attractor it sits
    k = zoom * s / half
    A0 = place + ((C - mid) @ Q.T) * k

    steps = max(30, int(40 * duration))
    lo, hi = ctr - 8 * s, ctr + 8 * s
    A1 = _rk4(f, A0, duration * tau, steps, lo, hi)
    new = mid + ((A1 - place) / k) @ Q

    mag = np.ones(len(C))
    if scale_radii:  # local volume change = |det of the flow's Jacobian|
        h = 1e-3 * s
        cols = [
            (_rk4(f, A0 + h * e, duration * tau, steps, lo, hi) - A1) / h
            for e in np.eye(3)
        ]
        mag = np.cbrt(np.abs(np.linalg.det(np.stack(cols, axis=2))))
    if not (np.isfinite(new).all() and np.isfinite(mag).all()):
        return _rebuild_room(groups, path, C)  # degenerate: no change

    # refit to the original size and place, then adjust the vertical spread
    new_mid = (new.min(0) + new.max(0)) / 2
    fit = 2 * half / max(np.ptp(new, axis=0).max(), 1e-9)
    new = mid + (new - new_mid) * fit
    new[:, 2] = mid[2] + (new[:, 2] - mid[2]) * z_scale

    radii = None
    med = np.median(mag)
    if scale_radii and med > 0:
        mag = np.clip(mag, 0.2 * med, 5 * med)
        radii = R * mag * fit
        if preserve_volume:
            radii = radii * (np.sum(R**3) / np.sum(radii**3)) ** (1 / 3)
    return _rebuild_room(groups, path, new, radii)


def attractor_flow_retry(room):
    return safe_wildcard(attractor_flow, room)


def invert_retry(room):
    return safe_wildcard(invert, room)


def complex_map_retry(room):
    return safe_wildcard(complex_map, room)


def room_features(room):
    groups = room[0]
    C = np.array([s["center"] for g in groups for s in g], dtype=float)
    R = np.array([s["radius"] for g in groups for s in g], dtype=float)
    dz = dh = 0.0
    turns = []
    for g in groups:
        P = np.array([s["center"] for s in g], dtype=float)
        if len(P) < 2:
            continue
        seg = np.diff(P, axis=0)
        dz += np.abs(seg[:, 2]).sum()
        dh += np.linalg.norm(seg[:, :2], axis=1).sum()
        if len(seg) >= 2:
            a, b = seg[:-1], seg[1:]
            cosang = (a * b).sum(1) / np.maximum(
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9
            )
            turns.append(np.arccos(np.clip(cosang, -1, 1)))
    return {
        "relief": float(np.ptp(C[:, 2]) / np.median(R)),
        "climb": float(dz / max(dh, 1e-9)),
        "radius_spread": float(np.std(np.log(R))),
        "turning": (
            float(np.concatenate(turns).mean()) if turns else float("nan")
        ),
        "polylines": float(len(groups)),
        "swallowed": swallowed_fraction(room),
    }


def calibrate(make_room, rng=random, n=150):
    """Run once per generator (and tree volume) and keep the result: the feature
    distributions of your own rooms, which the judge compares against."""
    feats = [room_features(make_room(rng)) for _ in range(n)]
    return {
        k: np.sort([f[k] for f in feats if np.isfinite(f[k])])
        for k in feats[0]
    }


_BETTER = {
    "relief": 1,
    "climb": 1,
    "radius_spread": 1,
    "turning": 1,
    "polylines": 1,
    "swallowed": -1,
}
_WEIGHT = {
    "relief": 2.0,
    "climb": 1.0,
    "radius_spread": 1.0,
    "turning": 1.0,
    "polylines": 0.5,
    "swallowed": 1.0,
}


def judge(f, ref, min_score=0.4, hard=(("relief", 0.15),)):
    """Each feature becomes a percentile rank among your reference rooms (flipped where
    lower is better). Accepts if the weighted average rank >= min_score and every
    `hard` feature ranks above its minimum. Returns (ok, score, ranks)."""
    ranks = {}
    for k, sign in _BETTER.items():
        v = f.get(k)
        if v is None or not np.isfinite(v) or len(ref.get(k, [])) == 0:
            continue
        p = np.searchsorted(ref[k], v) / len(ref[k])
        ranks[k] = p if sign > 0 else 1 - p
    if not ranks:
        return True, 1.0, ranks
    score = sum(_WEIGHT[k] * r for k, r in ranks.items()) / sum(
        _WEIGHT[k] for k in ranks
    )
    ok = score >= min_score and all(ranks.get(k, 1.0) >= m for k, m in hard)
    return ok, score, ranks


def heave(room, rng=random, amplitude=None):
    """Free repair for flat rooms: a smooth vertical swell. Depends only on position,
    so coincident spheres (tube junctions) move together."""
    groups, path, spheres, C, R = _extract_room(room)
    if len(C) < 2:
        return _rebuild_room(groups, path, C)
    ext = max(np.ptp(C[:, 0]), np.ptp(C[:, 1]), 1e-9)
    A = (
        rng.uniform(1.5, 4.0) * np.median(R)
        if amplitude is None
        else amplitude
    )
    u = (C[:, 0] - C[:, 0].mean()) / ext
    v = (C[:, 1] - C[:, 1].mean()) / ext
    dz = (
        A
        * np.sin(
            2 * math.pi * rng.uniform(0.8, 2.0) * u
            + rng.uniform(0, 2 * math.pi)
        )
        * np.cos(
            2 * math.pi * rng.uniform(0.8, 2.0) * v
            + rng.uniform(0, 2 * math.pi)
        )
    )
    new = C.copy()
    new[:, 2] += dz
    return _rebuild_room(groups, path, new)


def roll_interesting(make_room, rng, ref, tries=6, repair=False, **judge_kw):
    """make_room(rng) builds a complete room. Returns the first acceptable one, or the
    best-scoring of all tries if none passes."""
    best, best_score = None, -1.0
    for _ in range(tries):
        room = make_room(rng)
        ok, score, ranks = judge(room_features(room), ref, **judge_kw)
        # print(ok)
        if (
            not ok and repair and ranks.get("relief", 1.0) < 0.3
        ):  # flat: lift it before giving up on it
            room = heave(room, rng)
            ok, score, ranks = judge(room_features(room), ref, **judge_kw)
        if ok:
            return room
        if score > best_score:
            best, best_score = room, score
    return best
