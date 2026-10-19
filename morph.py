import math, random
import numpy as np


def _poly_arrays(g):
    return (
        np.array([s["center"] for s in g], dtype=float),
        np.array([s["radius"] for s in g], dtype=float),
    )


def _resample(P, R, n):
    """n points evenly spaced by arc length along a polyline (positions and radii)."""
    if len(P) == 1:
        return np.repeat(P, n, axis=0), np.repeat(R, n)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-9:  # all spheres at one spot
        return (
            np.repeat(P[:1], n, axis=0),
            np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(R)), R),
        )
    s = np.linspace(0, d[-1], n)
    return np.stack(
        [np.interp(s, d, P[:, k]) for k in range(3)], axis=1
    ), np.interp(s, d, R)


def _rotz(P, a, pivot):
    """Rotate points (any array shape ending in 3) about a vertical axis through pivot."""
    c, s = math.cos(a), math.sin(a)
    Q = P - pivot
    return (
        np.stack(
            [
                c * Q[..., 0] - s * Q[..., 1],
                s * Q[..., 0] + c * Q[..., 1],
                Q[..., 2],
            ],
            axis=-1,
        )
        + pivot
    )


def _pair_costs(Sa, ra, Sb, rb):
    """Cost of pairing every polyline of A with every one of B, trying both directions."""
    fwd = np.linalg.norm(Sa[:, None] - Sb[None], axis=-1).mean(-1)
    rev = np.linalg.norm(Sa[:, None] - Sb[None, :, ::-1], axis=-1).mean(-1)
    cost = np.minimum(fwd, rev) + 0.5 * np.abs(ra[:, None] - rb[None])
    return cost, rev < fwd


def morph_rooms(
    room_a,
    room_b,
    rng=random,
    t=None,
    gradient=None,
    align="yaw",
    leftovers="keep",
    volume="sum",
    K=8,
):
    """
    Blends two rooms into one hybrid.

    t:         blend, 0 = room A, 1 = room B (None = random 0.3-0.7)
    gradient:  0 = the same blend everywhere; > 0 makes each passage start like A
               and end like B (None = random, often 0)
    align:     "yaw"    - move B onto A's centroid and turn it about the vertical axis
                          to best match A (floors stay level)
               "center" - only move B onto A's centroid
               "none"   - blend the rooms where they are
    leftovers: what to do with polylines that have no partner (the bigger room's extras)
               "keep" - leave them unchanged, "fade" - shrink them by their room's
               weight (so filter_small can remove them), "drop" - delete them
    volume:    "sum"  - rescale radii so the total volume equals A's plus B's, which
                        keeps a combiner's additive volume rule
               "mean" - the average of the two
               None   - no rescaling
    K:         points per polyline used to compare shapes
    """
    ga, gb = room_a[0], room_b[0]
    if not ga or not gb:
        keep = room_a if ga else room_b
        return [[dict(s) for s in g] for g in keep[0]], [
            list(p) for p in keep[1]
        ]

    t = rng.uniform(0.3, 0.7) if t is None else t
    gradient = (
        rng.choice((0.0, 0.0, 0.8, 1.2)) if gradient is None else gradient
    )

    A = [_poly_arrays(g) for g in ga]
    B = [_poly_arrays(g) for g in gb]
    cA = np.concatenate([P for P, _ in A]).mean(0)
    cB = np.concatenate([P for P, _ in B]).mean(0)

    def signatures(arrs):
        return (
            np.array([_resample(P, R, K)[0] for P, R in arrs]),
            np.array([R.mean() for _, R in arrs]),
        )

    Sa, ra = signatures(A)
    if align != "none":
        B = [(P + (cA - cB), R) for P, R in B]
    if align == "yaw":  # pick the best of 24 turns
        Sb, rb = signatures(B)
        best = None
        for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            c, _ = _pair_costs(Sa, ra, _rotz(Sb, ang, cA), rb)
            score = c.min(1).mean() + c.min(0).mean()
            if best is None or score < best[0]:
                best = (score, ang)
        B = [(_rotz(P, best[1], cA), R) for P, R in B]
    Sb, rb = signatures(B)
    cost, rev = _pair_costs(Sa, ra, Sb, rb)

    # greedy matching: cheapest pairs first, each polyline used at most once
    out, usedA, usedB = [], set(), set()
    n_pairs = min(len(ga), len(gb))
    for flat in np.argsort(cost, axis=None):
        i, j = divmod(int(flat), cost.shape[1])
        if i in usedA or j in usedB:
            continue
        usedA.add(i)
        usedB.add(j)
        Pa, Ra = A[i]
        Pb, Rb = B[j]
        flip = bool(rev[i, j])
        if flip:
            Pb, Rb = Pb[::-1], Rb[::-1]
        n = max(len(Pa), len(Pb), 2)
        Pa_n, Ra_n = _resample(Pa, Ra, n)
        Pb_n, Rb_n = _resample(Pb, Rb, n)
        s = np.linspace(0, 1, n)
        w = np.clip(
            t + gradient * (s - 0.5), 0.0, 1.0
        )  # blend weight along the polyline
        P = (1 - w)[:, None] * Pa_n + w[:, None] * Pb_n
        Rr = (1 - w) * Ra_n + w * Rb_n

        poly = []
        for k in range(n):  # carry extra keys from the nearer parent
            if w[k] < 0.5:
                src = ga[i][int(round(float(s[k]) * (len(ga[i]) - 1)))]
            else:
                idx = int(round(float(s[k]) * (len(gb[j]) - 1)))
                src = gb[j][len(gb[j]) - 1 - idx if flip else idx]
            extra = {
                key: v
                for key, v in src.items()
                if key not in ("center", "radius")
            }
            poly.append(
                {
                    **extra,
                    "center": tuple(float(x) for x in P[k]),
                    "radius": float(Rr[k]),
                }
            )
        out.append(poly)
        if len(usedA) == n_pairs:
            break

    if leftovers != "drop":
        for i, g in enumerate(ga):
            if i not in usedA:
                f = (1 - t) if leftovers == "fade" else 1.0
                out.append([{**s_, "radius": s_["radius"] * f} for s_ in g])
        for j, g in enumerate(gb):
            if j not in usedB:
                f = t if leftovers == "fade" else 1.0
                out.append(
                    [
                        {
                            **s_,
                            "center": tuple(float(x) for x in c_),
                            "radius": s_["radius"] * f,
                        }
                        for s_, c_ in zip(g, B[j][0])
                    ]
                )

    if volume in ("sum", "mean"):  # total volume is proportional to sum of r^3
        target = sum((R**3).sum() for _, R in A) + sum(
            (R**3).sum() for _, R in B
        )
        target = target if volume == "sum" else target / 2
        cur = sum(s_["radius"] ** 3 for poly in out for s_ in poly)
        if cur > 0:
            f = (target / cur) ** (1 / 3)
            out = [
                [{**s_, "radius": s_["radius"] * f} for s_ in poly]
                for poly in out
            ]

    return out, [list(range(len(p))) for p in out]
