import math, random
import numpy as np


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
    z_scale=1.0,
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
    duration = rng.uniform(0.3, 1.5) if duration is None else duration
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
