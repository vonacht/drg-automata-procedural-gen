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
    added=None,
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
