import math, random, heapq
import numpy as np


# ------------------------------- shared helpers --------------------------------
def hemi_vol(r):
    return 2 / 3 * math.pi * r**3


def room_volume(room):
    return sum(hemi_vol(s["radius"]) for g in room[0] for s in g)


def volume_budget(room, cost, v_tree):
    """Geometry volume an op of tree-cost `cost` has to add to a room currently worth v_tree."""
    return cost * room_volume(room) / v_tree


def _copy_room(room):
    return [[dict(s) for s in g] for g in room[0]], [list(p) for p in room[1]]


def _swell_all(room, added):
    """Fallback when an op can't place new geometry at a legal size: grow every radius."""
    groups, path = _copy_room(room)
    f = ((room_volume(room) + added) / room_volume(room)) ** (1 / 3)
    return [
        [{**s, "radius": s["radius"] * f} for s in g] for g in groups
    ], path


def _dock(hostC, hostR, newC, newR, overlap=0.3):
    """Translation that slides the new spheres toward the host until the nearest pair overlaps."""
    D = np.linalg.norm(hostC[:, None] - newC[None], axis=-1)
    gap = D - (hostR[:, None] + newR[None])
    i, j = np.unravel_index(np.argmin(gap), gap.shape)
    if D[i, j] < 1e-9:
        return np.zeros(3)
    u = (hostC[i] - newC[j]) / D[i, j]  # from the new sphere toward the host
    return u * (gap[i, j] + overlap * (hostR[i] + newR[j]))


# ----------------------------------- echo --------------------------------------
def echo(
    room,
    rng=random,
    min_radius=350,
    scale=None,
    mirror=None,
    spin=None,
    lift=0.3,
):
    """
    Copies some neighboring polylines of the room, mirrors/turns the copy, scales it so
    its volume is exactly `added`, and docks it onto the room.

    scale:  rough size of the copy relative to the original part (None = random 0.6-1.5);
            this sets how much of the room gets copied, the exact scale follows from the budget
    mirror: reflect the copy in a vertical plane (None = random)
    spin:   turn the copy about the vertical axis, in radians (None = random)
    lift:   the copy may sit up to this many radians above or below the horizontal
    """
    added = 6e9
    groups, path = room
    if not groups or added <= 0:
        return _copy_room(room)
    Cs = [np.array([s["center"] for s in g], dtype=float) for g in groups]
    Rs = [np.array([s["radius"] for s in g], dtype=float) for g in groups]
    Vk = np.array([hemi_vol(R).sum() for R in Rs])
    cent = np.array([C.mean(0) for C in Cs])

    # which polylines to copy: a seed (bigger ones more likely) plus its nearest neighbors,
    # until their volume is about added / scale**3
    s0 = rng.uniform(0.6, 1.5) if scale is None else scale
    seed = rng.choices(range(len(groups)), weights=list(Vk))[0]
    subset, acc = [], 0.0
    for k in np.argsort(np.linalg.norm(cent - cent[seed], axis=1)):
        subset.append(int(k))
        acc += Vk[k]
        if acc >= added / s0**3:
            break

    # exact scale: the copy's volume is f**3 * V_subset. Drop polylines whose
    # thinnest sphere would end up below the engine's minimum.
    for _ in range(6):
        f = (added / Vk[subset].sum()) ** (1 / 3)
        bad = [k for k in subset if Rs[k].min() * f < min_radius]
        if not bad:
            break
        subset = [k for k in subset if k not in bad]
        if not subset:
            return _swell_all(room, added)
    else:
        return _swell_all(room, added)

    # mirror, turn and scale about the subset's centroid
    g = np.concatenate([Cs[k] for k in subset]).mean(0)
    ang = rng.uniform(0, 2 * math.pi) if spin is None else spin
    flip = (rng.random() < 0.5) if mirror is None else mirror
    ca, sa = math.cos(ang), math.sin(ang)

    def xform(P):
        q = (P - g) * (np.array([-1.0, 1.0, 1.0]) if flip else 1.0)
        q = np.stack(
            [
                ca * q[:, 0] - sa * q[:, 1],
                sa * q[:, 0] + ca * q[:, 1],
                q[:, 2],
            ],
            axis=1,
        )
        return g + f * q

    Pc = [xform(Cs[k]) for k in subset]

    # place it beyond the room along a random direction, then slide it back until it touches
    hostC, hostR = np.concatenate(Cs), np.concatenate(Rs)
    allC, allR = np.concatenate(Pc), np.concatenate(
        [Rs[k] * f for k in subset]
    )
    az, el = rng.uniform(0, 2 * math.pi), rng.uniform(-lift, lift)
    u = np.array(
        [
            math.cos(az) * math.cos(el),
            math.sin(az) * math.cos(el),
            math.sin(el),
        ]
    )
    shift = u * ((hostC @ u + hostR).max() - (allC @ u - allR).min())
    shift = shift + _dock(hostC, hostR, allC + shift, allR)

    new_groups, new_paths = [], []
    for k, P in zip(subset, Pc):
        new_groups.append(
            [
                {
                    **sph,
                    "center": tuple(float(x) for x in (p + shift)),
                    "radius": float(sph["radius"] * f),
                }
                for sph, p in zip(groups[k], P)
            ]
        )
        new_paths.append(
            list(path[k])
            if len(path) == len(groups)
            else list(range(len(groups[k])))
        )
    out_groups = [[dict(s) for s in g_] for g_ in groups] + new_groups
    out_path = (
        [list(p) for p in path] + new_paths
        if len(path) == len(groups)
        else [list(range(len(g_))) for g_ in out_groups]
    )
    return out_groups, out_path


# ---------------------------------- bypass -------------------------------------
def _geodesic(adj, src, n):
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    return dist


def bypass(
    room, rng=random, min_radius=350, spacing=0.9, vertical=0.3, tries=12
):
    """
    Adds a new passage between two spheres that are close in space but far apart
    by the room's own passages, bowed out so it reads as a separate route. It
    also joins pieces that were not connected at all. Its width is set by the
    budget: more volume means a fatter or longer, more winding bypass.

    vertical: chance that the bow is vertical (an arch or a sag) instead of sideways
    tries:    random starting spheres to search from
    """
    added = 8e9
    groups, path = room
    C = np.array([s["center"] for g in groups for s in g], dtype=float)
    R = np.array([s["radius"] for g in groups for s in g], dtype=float)
    N = len(C)
    if N < 4 or added <= 0:
        return _swell_all(room, added) if added > 0 else _copy_room(room)

    # graph of the room: along polylines, plus wherever spheres overlap
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    adj, base = [[] for _ in range(N)], 0
    for g in groups:
        for i in range(len(g) - 1):
            w = float(D[base + i, base + i + 1])
            adj[base + i].append((base + i + 1, w))
            adj[base + i + 1].append((base + i, w))
        base += len(g)
    for i, j in zip(*np.where(np.triu(D < 0.9 * (R[:, None] + R[None]), 1))):
        adj[i].append((int(j), float(D[i, j])))
        adj[j].append((int(i), float(D[i, j])))

    # candidate pairs: far by the room's passages, near in space, and affordable
    L_max = (
        3 * spacing * added / (2 * math.pi * min_radius**2)
    )  # longest bypass the budget can fund
    cands = []
    for src in rng.sample(range(N), min(N, tries)):
        geo, eu = _geodesic(adj, src, N), D[src]
        ok = (
            (eu > 1.2 * (R[src] + R) / 2) & (eu < 0.9 * L_max) & (geo > 3 * eu)
        )
        cands += [(geo[j] / eu[j], src, int(j)) for j in np.flatnonzero(ok)]
    if not cands:
        return _swell_all(room, added)
    cands.sort(reverse=True)
    _, ia, ib = rng.choice(cands[:3])
    a, b = C[ia], C[ib]
    eu = float(np.linalg.norm(b - a))
    d = (b - a) / eu

    # the bow: sideways in the floor plan, or vertical
    up = np.array([0.0, 0.0, 1.0])
    perp = up - (up @ d) * d if rng.random() < vertical else np.cross(up, d)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([1.0, 0.0, 0.0]) - d[0] * d
    perp = perp / np.linalg.norm(perp) * rng.choice((-1, 1))
    prof = (
        (lambda t: np.sin(np.pi * t))
        if rng.random() < 0.5
        else (lambda t: np.sin(2 * np.pi * t))
    )

    def curve(h, m=300):
        t = np.linspace(0, 1, m)[:, None]
        return a + (b - a) * t + perp * h * prof(t)

    def length(h):
        return float(np.linalg.norm(np.diff(curve(h), axis=0), axis=1).sum())

    # path length the budget supports at a comfortable width, then the bow that gives it
    r_t = max(rng.uniform(0.45, 0.7) * (R[ia] + R[ib]) / 2, 1.05 * min_radius)
    L_t = min(3 * spacing * added / (2 * math.pi * r_t**2), 6 * eu)
    lo, hi = 0.2 * eu, 4 * eu
    if L_t <= length(lo):
        h = lo
    elif L_t >= length(hi):
        h = hi
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if length(mid) < L_t else (lo, mid)
        h = (lo + hi) / 2

    # spheres along the curve, starting just inside the host spheres
    pts = curve(h, 400)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    L = s[-1]
    s_a, s_b = min(0.6 * R[ia], 0.3 * L), L - min(0.6 * R[ib], 0.3 * L)
    n = max(1, int(round((s_b - s_a) / (spacing * r_t))) + 1)
    wob, ph = rng.uniform(1, 3), rng.uniform(0, 2 * math.pi)
    while True:
        ss = (
            np.array([(s_a + s_b) / 2]) if n == 1 else np.linspace(s_a, s_b, n)
        )
        p = 1 + 0.12 * np.sin(
            2 * np.pi * wob * ss / L + ph
        )  # mild width wobble
        r = (3 * added / (2 * math.pi * (p**3).sum())) ** (
            1 / 3
        )  # exact: the spheres sum to `added`
        if r * p.min() >= min_radius or n == 1:
            break
        n -= 1  # fewer, fatter spheres
    if r * p.min() < min_radius:
        return _swell_all(room, added)

    centers = np.stack([np.interp(ss, s, pts[:, k]) for k in range(3)], axis=1)
    new = [
        {"center": tuple(float(x) for x in c_), "radius": float(r * p_)}
        for c_, p_ in zip(centers, p)
    ]
    out_groups = [[dict(s_) for s_ in g] for g in groups] + [new]
    out_path = (
        [list(p_) for p_ in path] + [list(range(len(new)))]
        if len(path) == len(groups)
        else [list(range(len(g))) for g in out_groups]
    )
    return out_groups, out_path


import math, random
import numpy as np


# ================================= cave-in =====================================
def cave_in(room, rng=random, fall=None, sink=None, ground=None):
    """
    Lets every chamber drop until its flat floor lands on the dome of whatever is
    below it. Only heights change, and radii are untouched, so it's a free op.

    fall:   how much of the fall happens, 0-1 (None = random 0.6-1.0)
    sink:   how deep a chamber settles into the dome it lands on, as a fraction of
            its radius; > 0 keeps stacked chambers connected (None = random 0.25-0.5)
    ground: height unsupported chambers drop to (None = the room's lowest floor)
    """
    groups, path, spheres, C, R = _extract_room(room)
    if len(C) < 2:
        return _rebuild_room(groups, path, C)
    fall = rng.uniform(0.6, 1.0) if fall is None else fall
    sink = rng.uniform(0.25, 0.5) if sink is None else sink

    # coincident spheres (tube junctions) are one body, so they land together
    uniq, inv = np.unique(np.round(C, 3), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    m = len(uniq)
    Cb = np.zeros((m, 3))
    np.add.at(Cb, inv, C)
    Cb /= np.bincount(inv, minlength=m)[:, None]
    rb = np.zeros(m)
    np.maximum.at(rb, inv, R)

    z0 = Cb[:, 2].copy()
    z = z0.copy()
    floor = z0.min() if ground is None else ground
    order = np.argsort(
        z0, kind="stable"
    )  # lowest first: supports settle before what's above
    for k, i in enumerate(order):
        target = floor
        j = order[:k]  # chambers that have already settled
        if k:
            d = np.linalg.norm(Cb[i, :2] - Cb[j, :2], axis=1)
            e = np.maximum(
                0.0, d - rb[i]
            )  # closest the floor disc gets to j's axis
            touch = e < rb[j]
            if touch.any():  # highest dome point under the floor disc
                H = z[j][touch] + np.sqrt(rb[j][touch] ** 2 - e[touch] ** 2)
                target = max(floor, H.max() - sink * rb[i])
        z[i] = min(z0[i], target)  # chambers never rise

    new = C.copy()
    new[:, 2] += ((z - z0) * fall)[inv]
    return _rebuild_room(groups, path, new)


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


# ============================== negative space =================================
def _dist_to_halfballs(P, C, R, chunk=800):
    """Distance from each point to the nearest of the solid half-balls (flat face down);
    negative inside."""
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        d = P[s : s + chunk, None, :] - C[None]
        dz = d[..., 2]
        e = np.linalg.norm(d[..., :2], axis=-1)
        dist = np.where(
            dz >= 0,
            np.linalg.norm(d, axis=-1) - R[None],  # above the floor: the dome
            np.where(e <= R[None], -dz, np.hypot(e - R[None], dz)),
        )  # below: floor disc or rim
        out[s : s + chunk] = dist.min(axis=1)
    return out


def negative_space(
    room,
    rng=random,
    added=8e9,
    keep_original=False,
    inflate=0.2,
    pad=0.05,
    candidates=6000,
    max_radius=None,
    min_radius=350,
    match_volume=True,
    max_spheres=400,
):
    """
    Fills the gaps around the room with the largest chambers that fit, so the
    voids become the cave and the old chambers become solid.

    added:         volume of new chambers (None = the room's own volume)
    keep_original: False = replace the room by its negative (free op);
                   True  = keep the room and add the negative to it
                           (costly: pass added=your budget)
    inflate:       new chambers are this much bigger than the gap they sit in, so
                   neighbors overlap and connect
    pad:           how far the filled box extends past the room, as a fraction of its size
    candidates:    random points tried; more = better placement, slower
    max_radius:    largest new chamber (None = 1.25 x the room's largest)
    match_volume:  rescale radii so the new chambers total exactly the target volume
    """
    groups, path = room
    spheres = [s for g in groups for s in g]
    if len(spheres) < 2:
        return _copy_room(room)
    C = np.array([s["center"] for s in spheres], dtype=float)
    R = np.array([s["radius"] for s in spheres], dtype=float)
    target = hemi_vol(R).sum() if added is None else added

    # the box to fill: the room's extent, floors at the room's lowest floor
    lo = np.array([(C[:, 0] - R).min(), (C[:, 1] - R).min(), C[:, 2].min()])
    hi = np.array(
        [(C[:, 0] + R).max(), (C[:, 1] + R).max(), (C[:, 2] + R).max()]
    )
    ext = hi - lo
    lo[:2] -= pad * ext[:2]
    hi += pad * ext * np.array([1, 1, 1])

    gen = np.random.default_rng(rng.randrange(2**32))
    P = lo + gen.random((candidates, 3)) * (hi - lo)
    wall = np.minimum.reduce(
        [
            P[:, 0] - lo[0],
            hi[0] - P[:, 0],
            P[:, 1] - lo[1],
            hi[1] - P[:, 1],
            hi[2] - P[:, 2],
        ]
    )  # the dome must fit under the top
    clear = np.minimum(_dist_to_halfballs(P, C, R), wall)
    cap = 1.25 * R.max() if max_radius is None else max_radius

    # greedy: always the largest chamber that still fits, until the volume is used up
    cen, rad, V_new = [], [], 0.0
    while V_new < target and len(rad) < max_spheres:
        k = int(np.argmax(clear))
        r = min(clear[k] * (1 + inflate), cap)
        if r < min_radius:
            break
        cen.append(P[k].copy())
        rad.append(r)
        V_new += hemi_vol(r)
        clear = np.minimum(clear, np.linalg.norm(P - P[k], axis=1) - r)
    if not rad:
        return _copy_room(room)  # no room for anything: unchanged
    cen, rad = np.array(cen), np.array(rad)
    if match_volume:
        rad = rad * min(1.5, (target / hemi_vol(rad).sum()) ** (1 / 3))

    # connect them: each chamber hangs off the nearest earlier one; the first child
    # continues its parent's polyline, others start a new polyline at a copy of it
    n = len(rad)
    parent = [-1] + [
        int(np.argmin(np.linalg.norm(cen[:i] - cen[i], axis=1) - rad[:i]))
        for i in range(1, n)
    ]
    polys, where = [[0]], {0: 0}
    for i in range(1, n):
        p = parent[i]
        if polys[where[p]][-1] == p:
            polys[where[p]].append(i)
            where[i] = where[p]
        else:
            polys.append([p, i])
            where[i] = len(polys) - 1
    new_groups = [
        [
            {
                "center": tuple(float(x) for x in cen[i]),
                "radius": float(rad[i]),
            }
            for i in poly
        ]
        for poly in polys
    ]
    new_paths = [list(range(len(g))) for g in new_groups]

    if not keep_original:
        return new_groups, new_paths
    old_groups, old_path = _copy_room(room)
    out_groups = old_groups + new_groups
    out_path = (
        old_path + new_paths
        if len(old_path) == len(old_groups)
        else [list(range(len(g))) for g in out_groups]
    )
    return out_groups, out_path
