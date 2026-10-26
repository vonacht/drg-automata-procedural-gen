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


def _copy_room(room):
    return [[dict(s) for s in g] for g in room[0]], [list(p) for p in room[1]]


def _dock(hostC, hostR, newC, newR, overlap=0.3):
    """Translation that slides the new spheres toward the host until the nearest pair overlaps."""
    D = np.linalg.norm(hostC[:, None] - newC[None], axis=-1)
    gap = D - (hostR[:, None] + newR[None])
    i, j = np.unravel_index(np.argmin(gap), gap.shape)
    if D[i, j] < 1e-9:
        return np.zeros(3)
    u = (hostC[i] - newC[j]) / D[i, j]  # from the new sphere toward the host
    return u * (gap[i, j] + overlap * (hostR[i] + newR[j]))


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


# ----------------------------- shared helpers ----------------------------------
def _arrays(groups):
    return [
        (
            np.array([s["center"] for s in g], dtype=float),
            np.array([s["radius"] for s in g], dtype=float),
        )
        for g in groups
    ]


def _signatures(arrs, K=8):
    return (
        np.array([_resample(P, R, K)[0] for P, R in arrs]),
        np.array([R.mean() for _, R in arrs]),
    )


def _align_rooms(A, B, align="yaw", K=8):
    """Moves room B (a list of (centers, radii)) onto A's centroid and, for "yaw",
    turns it about the vertical axis to best match A."""
    if align == "none":
        return B
    cA = np.concatenate([P for P, _ in A]).mean(0)
    cB = np.concatenate([P for P, _ in B]).mean(0)
    B = [(P + (cA - cB), R) for P, R in B]
    if align == "yaw":
        Sa, ra = _signatures(A, K)
        Sb, rb = _signatures(B, K)
        best = None
        for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            c, _ = _pair_costs(Sa, ra, _rotz(Sb, ang, cA), rb)
            score = c.min(1).mean() + c.min(0).mean()
            if best is None or score < best[0]:
                best = (score, ang)
        B = [(_rotz(P, best[1], cA), R) for P, R in B]
    return B


def _match_pairs(A, B, K=8):
    """Greedy pairing of polylines: [(i, j, B_is_reversed), ...], cheapest first."""
    Sa, ra = _signatures(A, K)
    Sb, rb = _signatures(B, K)
    cost, rev = _pair_costs(Sa, ra, Sb, rb)
    pairs, usedA, usedB = [], set(), set()
    for flat in np.argsort(cost, axis=None):
        i, j = divmod(int(flat), cost.shape[1])
        if i in usedA or j in usedB:
            continue
        usedA.add(i)
        usedB.add(j)
        pairs.append((i, j, bool(rev[i, j])))
        if len(pairs) == min(len(A), len(B)):
            break
    return pairs


def _arc_param(P):
    """Each point's position along its polyline, 0 to 1, by arc length."""
    if len(P) < 2:
        return np.zeros(len(P))
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return d / d[-1] if d[-1] > 1e-9 else np.linspace(0, 1, len(P))


def _with(groups, centers):
    return [
        [
            {**s, "center": tuple(float(x) for x in centers[gi][k])}
            for k, s in enumerate(g)
        ]
        for gi, g in enumerate(groups)
    ]


def _merge_paths(pa, ga, pb, gb, out_groups):
    if len(pa) == len(ga) and len(pb) == len(gb):
        return [list(p) for p in pa] + [list(p) for p in pb]
    return [list(range(len(g))) for g in out_groups]


def _r3(groups):
    return sum(s["radius"] ** 3 for g in groups for s in g)


def _rescale(groups, target, cap=1.6):
    """Scales all radii so the sum of r^3 reaches target (volume is proportional to it)."""
    cur = _r3(groups)
    if cur <= 0:
        return groups
    f = min(cap, (target / cur) ** (1 / 3))
    return [[{**s, "radius": s["radius"] * f} for s in g] for g in groups]


def _touch_shift(hostC, hostR, newC, newR):
    """Zero if the two sets already touch; otherwise the shift that makes them."""
    D = np.linalg.norm(hostC[:, None] - newC[None], axis=-1)
    if (D - (hostR[:, None] + newR[None])).min() <= 0:
        return np.zeros(3)
    return _dock(hostC, hostR, newC, newR)


def _ensure_two(groups):
    """A room with one long polyline is cut in half so there is something to splice."""
    if len(groups) == 1 and len(groups[0]) >= 4:
        g, m = groups[0], len(groups[0]) // 2
        return [g[:m], g[m:]]
    return groups


def _assemble(groups, path, centers, radii, keep, min_points=2):
    """Rebuilds polylines from the spheres marked `keep`, splitting where spheres were
    removed (no bridging links); `path` is remapped, or None for zero-based paths.
    """
    out_g, out_p = [], []
    for gi, g in enumerate(groups):
        pieces, cur = [], []
        for k in range(len(g)):
            if keep[gi][k]:
                cur.append(k)
            elif cur:
                pieces.append(cur)
                cur = []
        if cur:
            pieces.append(cur)
        for piece in pieces:
            if len(piece) < min_points:
                continue
            out_g.append(
                [
                    {
                        **g[k],
                        "center": tuple(float(x) for x in centers[gi][k]),
                        "radius": float(radii[gi][k]),
                    }
                    for k in piece
                ]
            )
            if path is not None:
                pos = {old: new for new, old in enumerate(piece)}
                out_p.append([pos[k] for k in path[gi] if k in pos])
            else:
                out_p.append(list(range(len(piece))))
    return out_g, out_p


# ------------------------------------ braid ------------------------------------
def braid(room_a, room_b, rng=random, weaves=None, lift=0.5, align="yaw"):
    """
    Pairs up the polylines of the two rooms and weaves each pair around the other:
    the strands swap sides several times, passing over and under at each crossing.
    Every sphere and radius of both rooms is kept, so the volume is exactly the sum.

    weaves: how many times each pair completes a swap and swaps back (None = random 1-3)
    lift:   height of the over/under, as a fraction of the strands' mean radius;
            below 1 the strands still overlap at a crossing, so players can switch
    align:  "yaw", "center" or "none", as in morph_rooms
    """
    if rng.random() < 0.5:
        room_a, room_b = room_b, room_a
    (ga, pa), (gb, pb) = room_a, room_b
    if not ga or not gb:
        return _copy_room(room_a if ga else room_b)

    A = _arrays(ga)
    B = _align_rooms(A, _arrays(gb), align)
    newA, newB = [P.copy() for P, _ in A], [P.copy() for P, _ in B]
    grid = np.linspace(0, 1, 200)
    up = np.array([0.0, 0.0, 1.0])

    for i, j, flip in _match_pairs(A, B):
        (Pa, Ra), (Pb, Rb) = A[i], B[j]
        Qb, Rq = (Pb[::-1], Rb[::-1]) if flip else (Pb, Rb)
        S = (
            _resample(Qb, Rq, 200)[0] - _resample(Pa, Ra, 200)[0]
        )  # from A's strand to B's strand
        m = rng.choice((1, 2, 3)) if weaves is None else weaves
        sign = rng.choice((-1, 1))
        amp = lift * (Ra.mean() + Rb.mean()) / 2

        def push(
            s,
        ):  # how far a point of A's strand moves; B's moves the opposite way
            Sx = np.stack(
                [np.interp(s, grid, S[:, k]) for k in range(3)], axis=1
            )
            psi = 2 * math.pi * m * s
            return (Sx / 2) * (1 - np.cos(psi))[:, None] + up * (
                sign * amp * np.sin(psi)
            )[:, None]

        newA[i] = Pa + push(_arc_param(Pa))
        sb = _arc_param(Pb)
        newB[j] = Pb - push(1 - sb if flip else sb)

    groups = _with(ga, newA) + _with(gb, newB)
    return groups, _merge_paths(pa, ga, pb, gb, groups)


# ------------------------------------ seam -------------------------------------
def seam(
    room_a, room_b, rng=random, blend=None, cut=0.3, vertical=0.2, align="yaw"
):
    """
    Aligns the rooms, cuts along a random plane through their shared center, and keeps
    one room on each side, with a crossfade in the middle where both are present at
    reduced size. The result is literally half one room and half the other.

    blend:    width of the crossfade, as a fraction of the room's size (None = random 0.08-0.3;
              0 = a hard cut)
    cut:      spheres whose weight falls below this are dropped
    vertical: chance that the cut is horizontal (one room above the other) instead of vertical
    """
    if rng.random() < 0.5:
        room_a, room_b = room_b, room_a
    (ga, pa), (gb, pb) = room_a, room_b
    if not ga or not gb:
        return _copy_room(room_a if ga else room_b)
    target = _r3(ga) + _r3(gb)
    A = _arrays(ga)
    B = _align_rooms(A, _arrays(gb), align)

    if rng.random() < vertical:
        n = np.array([0.0, 0.0, rng.choice((-1.0, 1.0))])
    else:
        a = rng.uniform(0, 2 * math.pi)
        n = np.array([math.cos(a), math.sin(a), 0.0])
    span = max(np.ptp(np.concatenate([P for P, _ in A + B]) @ n), 1e-9)
    thr = (
        np.concatenate([P for P, _ in A]).mean(0) @ n
        + rng.uniform(-0.15, 0.15) * span
    )
    band = max(
        (rng.uniform(0.08, 0.3) if blend is None else blend) * span, 1e-9
    )

    def weight_b(P):  # 0 on A's side, 1 on B's, smooth across the band
        t = np.clip((P @ n - thr) / band + 0.5, 0.0, 1.0)
        return t * t * (3 - 2 * t)

    wA = [1 - weight_b(P) for P, _ in A]
    wB = [weight_b(P) for P, _ in B]
    kA, kB = [w >= cut for w in wA], [w >= cut for w in wB]
    aligned = len(pa) == len(ga) and len(pb) == len(gb)

    def whole():  # nothing to cut: just the two rooms together
        groups = _with(ga, [P for P, _ in A]) + _with(gb, [P for P, _ in B])
        return groups, _merge_paths(pa, ga, pb, gb, groups)

    if not any(k.any() for k in kA) or not any(k.any() for k in kB):
        return whole()

    # make sure the two halves touch
    hostC = np.concatenate([P[k] for (P, _), k in zip(A, kA)])
    hostR = np.concatenate([(R * w)[k] for (_, R), w, k in zip(A, wA, kA)])
    newC = np.concatenate([P[k] for (P, _), k in zip(B, kB)])
    newR = np.concatenate([(R * w)[k] for (_, R), w, k in zip(B, wB, kB)])
    B = [(P + _touch_shift(hostC, hostR, newC, newR), R) for P, R in B]

    out, out_path = _assemble(
        ga + gb,
        (pa + pb) if aligned else None,
        [P for P, _ in A] + [P for P, _ in B],
        [R * w for (_, R), w in zip(A, wA)]
        + [R * w for (_, R), w in zip(B, wB)],
        kA + kB,
    )
    if not out:
        return whole()
    return _rescale(out, target), out_path


# ---------------------------------- crossover ----------------------------------
def crossover(room_a, room_b, rng=random, align="yaw"):
    """
    A genetic-style splice: keeps the first (or last) polylines of one room and the
    complementary end of the other, aligns them and docks them together. It cuts by the
    order of the polylines, not by position, so for generated rooms (where order means
    something, like trunk first and branches later) the parts stay coherent.
    """
    if rng.random() < 0.5:
        room_a, room_b = room_b, room_a
    ga, gb = room_a[0], room_b[0]
    if not ga or not gb:
        return _copy_room(room_a if ga else room_b)
    target = _r3(ga) + _r3(gb)
    ga, gb = _ensure_two(ga), _ensure_two(gb)

    f = rng.uniform(0.35, 0.65)
    nA, nB = len(ga), len(gb)
    kA = (
        min(max(round(f * nA), 1), nA - 1) if nA > 1 else 1
    )  # how many of A to keep
    kB = (
        min(max(round((1 - f) * nB), 1), nB - 1) if nB > 1 else 1
    )  # how many of B to keep
    head = rng.random() < 0.5  # A's head + B's tail, or the reverse
    keepA = ga[:kA] if head else ga[nA - kA :]
    keepB = gb[nB - kB :] if head else gb[:kB]

    A = _arrays(keepA)
    B = _align_rooms(A, _arrays(keepB), align)
    shift = _touch_shift(
        np.concatenate([P for P, _ in A]),
        np.concatenate([R for _, R in A]),
        np.concatenate([P for P, _ in B]),
        np.concatenate([R for _, R in B]),
    )
    B = [(P + shift, R) for P, R in B]

    groups = _with(keepA, [P for P, _ in A]) + _with(keepB, [P for P, _ in B])
    groups = _rescale(groups, target)
    return groups, [list(range(len(g))) for g in groups]
