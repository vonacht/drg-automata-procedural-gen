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
def lens(room, rng=random, k=20, reach=None, center=None, aspect=None):
    """
    Pushes points away from (or pulls them toward) a center, fading smoothly to
    nothing at distance reach * extent. Radii are untouched.

    k:      > 0 magnifies the middle (spreads it out, opens a hollow);
            < 0 compresses it. Clamped to (-0.95, 2.9), which keeps the map fold-free.
    reach:  lens radius as a fraction of the room's largest extent (None = random 0.25-0.6)
    center: (x, y, z), or None for the center of a random sphere
    aspect: (ax, ay, az) stretches the lens into an ellipsoid; None = random
            (mostly round, sometimes a flat disc or a tall column)
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
    reach = rng.uniform(0.25, 0.6) if reach is None else reach
    if k is None:
        k = (
            rng.uniform(0.6, 2.0)
            if rng.random() < 0.6
            else -rng.uniform(0.4, 0.8)
        )
    k = min(max(k, -0.95), 2.9)
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
    w = np.where(d < 1, (1 - d) ** 2, 0.0)  # smooth, zero slope at the rim
    return _rebuild_room(groups, path, c + rel * (1 + k * w)[:, None])


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
    """Replace values too close to zero, to avoid dividing by them."""
    return np.where(np.abs(x) < eps, eps, x)


# ---- the maps: each takes a complex array w (position in units of the room's
#      half-extent around the origin) and returns a complex array -----------------
def _make_exp(rng, thick=None, sweep=None):
    thick = rng.uniform(0.3, 0.8) if thick is None else thick  # ring thickness
    sweep = (
        rng.uniform(2.5, 2 * math.pi) if sweep is None else sweep
    )  # angle covered
    return lambda w: np.exp(w.real * thick + 1j * w.imag * sweep / 2)


def _make_log(rng):
    rot = np.exp(1j * rng.uniform(0, 2 * math.pi))  # where the seam falls

    def f(w):
        w = w * rot
        return np.log(np.maximum(np.abs(w), 0.05)) + 1j * np.angle(w)

    return f


def _make_power(rng, p=None):
    p = rng.choice((2, 3, 0.5, 1.5)) if p is None else p
    rot = np.exp(1j * rng.uniform(0, 2 * math.pi))
    return lambda w: np.power(w * rot, p)


def _make_joukowski(rng, s=None):
    s = rng.uniform(0.6, 1.3) if s is None else s
    return lambda w: (lambda u: u + 1 / u)(_nz(w * s, 0.05))


def _make_mobius(rng, rho=None, phi=None):
    p1 = (
        0.6
        * math.sqrt(rng.random())
        * np.exp(1j * rng.uniform(0, 2 * math.pi))
    )
    p2 = p1 + rng.uniform(0.5, 1.2) * np.exp(1j * rng.uniform(0, 2 * math.pi))
    rho = rng.uniform(0.5, 1.6) if rho is None else rho  # pull toward p1 / p2
    phi = rng.uniform(-1.2, 1.2) if phi is None else phi  # spiral angle
    k = rho * np.exp(1j * phi)

    def f(w):  # (M(w) - p1) / (M(w) - p2) = k (w - p1) / (w - p2)
        u2 = k * (w - p1) / _nz(w - p2)
        return (p1 - p2 * u2) / _nz(1 - u2)

    return f


def _make_sin(rng, s=None):
    s = rng.uniform(1.0, 2.2) if s is None else s
    return lambda w: np.sin(w * s)


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
    scale_radii=True,
    preserve_volume=True,
):
    """
    Treats two coordinates as a complex number w and replaces it with f(w);
    the third coordinate is unchanged.

    kind:   "exp" (corridors become rings), "log" (rings unroll into strips),
            "power" (w**2 folds the room onto itself, w**0.5 unfolds it),
            "joukowski" (inside of the unit circle flips outside),
            "mobius" (spirals toward a fixed point), "sin" (elliptical bands);
            None = random
    plane:  "xy" (floor plan), "xz" or "yz" (vertical slices)
    params: optional dict of the chosen map's own options, e.g. {"sweep": 6.0}
            for exp, {"p": 3} for power, {"rho": 1.4, "phi": 0.8} for mobius
    scale_radii / preserve_volume: as in `invert`
    """
    groups, path, spheres, C, R = _extract_room(room)
    if len(C) < 2:
        return _rebuild_room(groups, path, C)
    kind = kind or rng.choice(list(_MAKERS))
    print(kind)
    i, j = _PLANES[plane]

    xy = C[:, [i, j]]
    mid = (xy.min(0) + xy.max(0)) / 2
    half = max(np.ptp(xy, axis=0).max(), 1e-9) / 2
    # exp, sin and mobius act around the room's middle; the others around a random
    # sphere, which sometimes sits inside the room and produces the wrap-around seams
    origin = (
        xy[rng.randrange(len(xy))]
        if kind in ("log", "power", "joukowski")
        else mid
    )
    f = _MAKERS[kind](rng, **(params or {}))

    def F(p):
        w = ((p[:, 0] - origin[0]) + 1j * (p[:, 1] - origin[1])) / half
        out = f(w)
        return np.c_[out.real, out.imag]

    new = F(xy)
    h = 1e-3 * half
    mag = (
        np.linalg.norm(F(xy + np.array([h, 0.0])) - new, axis=1) / h
    )  # local stretch
    if not (np.isfinite(new).all() and np.isfinite(mag).all()):
        return _rebuild_room(groups, path, C)  # degenerate: no change

    # refit to the original size and place
    new_mid = (new.min(0) + new.max(0)) / 2
    fit = 2 * half / max(np.ptp(new, axis=0).max(), 1e-9)
    new = mid + (new - new_mid) * fit

    out_C = C.copy()
    out_C[:, [i, j]] = new
    radii = None
    med = np.median(mag)
    if scale_radii and med > 0:
        mag = np.clip(mag, 0.2 * med, 5 * med)
        radii = R * mag * fit
        if preserve_volume:
            radii = radii * (np.sum(R**3) / np.sum(radii**3)) ** (1 / 3)
    return _rebuild_room(groups, path, out_C, radii)


def safe_wildcard(op, room, rng=random, tries=5, **kw):
    """Runs any wildcard op, rejecting NaNs, collapsed or exploded results,
    and returns the original room if no try passes."""
    _, _, _, C, _ = _extract_room(room)
    extent = np.ptp(C, axis=0).max()
    for _ in range(tries):
        out = op(room, rng, **kw)
        C2 = np.array([s["center"] for g in out[0] for s in g], dtype=float)
        R2 = np.array([s["radius"] for g in out[0] for s in g], dtype=float)
        if np.isfinite(C2).all() and np.isfinite(R2).all() and (R2 > 0).all():
            if 0.3 * extent <= np.ptp(C2, axis=0).max() <= 3 * extent:
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
    kind="lorenz",
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
    f, ctr, s, tau = ATTRACTORS[kind]
    ctr = np.array(ctr, dtype=float)
    duration = rng.uniform(0.3, 0.7) if duration is None else duration
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
