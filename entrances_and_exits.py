import heapq, math, random
from types import SimpleNamespace
import numpy as np


class MissingEntranceOrExit_new(Exception):
    pass


def room_graph(room, overlap=0.9, climb_penalty=0.0):
    """Walkable graph: nodes are spheres; edges are consecutive polyline points and any
    pair of overlapping spheres. Weight = distance between centers, optionally
    increased on slopes. Returns (adj, centers, radii, owner), where owner[i] is
    (polyline index, sphere index)."""
    groups = room[0]
    owner = [(gi, k) for gi, g in enumerate(groups) for k in range(len(g))]
    C = np.array([s["center"] for g in groups for s in g], dtype=float)
    R = np.array([s["radius"] for g in groups for s in g], dtype=float)
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    adj = [dict() for _ in range(len(C))]

    def add(i, j):
        w = float(D[i, j])
        if climb_penalty:
            w *= 1 + climb_penalty * abs(C[i, 2] - C[j, 2]) / max(w, 1e-9)
        if w < adj[i].get(j, np.inf):
            adj[i][j] = adj[j][i] = w

    base = 0
    for g in groups:
        for k in range(len(g) - 1):
            add(base + k, base + k + 1)
        base += len(g)
    for i, j in zip(
        *np.where(np.triu(D < overlap * (R[:, None] + R[None]), 1))
    ):
        add(int(i), int(j))
    return adj, C, R, owner


def walk_distance(adj, sources):
    """Walking distance from the nearest of the source spheres to every sphere."""
    dist = np.full(len(adj), np.inf)
    pq = [(0.0, int(s)) for s in sources]
    for _, s in pq:
        dist[s] = 0.0
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    return dist


def _main_component(adj):
    seen, best = [False] * len(adj), []
    for s in range(len(adj)):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def vector_to_ypr(n):
    """Yaw, pitch, roll in DEGREES for a direction n: yaw about +z measured from +x toward +y,
    pitch positive upward, roll 0 keeps the frame upright (its up is +z)."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    return (
        math.degrees(math.atan2(n[1], n[0])),
        math.degrees(math.asin(max(-1.0, min(1.0, n[2])))),
        0.0,
    )


def ypr_to_vector(yaw, pitch):
    """Inverse of the above, with the angles in degrees (roll doesn't change the pointing direction)."""
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array(
        [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)]
    )


def _links(groups):
    pairs = [(a, b) for g in groups for a, b in zip(g, g[1:])]
    if not pairs:
        return None
    return (
        np.array([a["center"] for a, _ in pairs], dtype=float),
        np.array([b["center"] for _, b in pairs], dtype=float),
        np.array([a["radius"] for a, _ in pairs], dtype=float),
        np.array([b["radius"] for _, b in pairs], dtype=float),
    )


def _inside_room(P, C, R, links, skip, margin=0.99):
    """Which points lie strictly inside the room: inside a hemisphere other than `skip`,
    or inside any link tube (radius interpolated between the ends)."""
    d = P[:, None, :] - C[None]
    inside = ((d**2).sum(-1) < (margin * R[None]) ** 2) & (d[..., 2] >= 0)
    inside[:, skip] = False
    out = inside.any(axis=1)
    if links is not None:
        A, B, RA, RB = links
        AB = B - A
        L2 = np.maximum((AB**2).sum(1), 1e-9)
        t = np.clip(
            ((P[:, None, :] - A[None]) * AB[None]).sum(-1) / L2[None], 0.0, 1.0
        )
        q = A[None] + t[..., None] * AB[None]
        rad = RA[None] + t * (RB - RA)[None]
        ok = (np.linalg.norm(P[:, None, :] - q, axis=-1) < margin * rad) & (
            P[:, None, 2] >= q[..., 2]
        )
        out |= ok.any(axis=1)
    return out


def _surface_door(i, ctx, rng, elevation, inward):
    """A point on sphere i's surface that is on the room's boundary, with its outward normal."""
    c, r = ctx.C[i], ctx.R[i]
    nb = list(ctx.adj[i])
    away = c - (
        ctx.C[nb].mean(0) if nb else ctx.centroid
    )  # away from the connecting passages
    away[2] = 0.0
    if np.linalg.norm(away) < 1e-6:
        away = c - ctx.centroid
        away[2] = 0.0
    if np.linalg.norm(away) < 1e-6:
        away = np.array([1.0, 0.0, 0.0])
    away = away / np.linalg.norm(away)

    el = math.radians(rng.uniform(*elevation))
    ang = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    U = np.stack(
        [
            math.cos(el) * np.cos(ang),
            math.cos(el) * np.sin(ang),
            np.full(72, math.sin(el)),
        ],
        axis=1,
    )  # outward normals of the dome
    P = c + r * U
    free = ~_inside_room(P, ctx.C, ctx.R, ctx.links, i)
    if not free.any():
        return None  # enclosed: no door possible here
    k = int(np.argmax(np.where(free, U @ away, -np.inf)))
    n = -U[k] if inward else U[k]
    yaw, pitch, roll = vector_to_ypr(n)
    return {
        "position": tuple(float(x) for x in P[k]),
        "normal": tuple(float(x) for x in n),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


def place_doors(
    room,
    rng=random,
    n_entrances=2,
    n_exits=2,
    entrance_mode="peripheral",
    gap=(1.0, 3.0),
    exit_spread=0.5,
    variety=0.8,
    climb_penalty=0.0,
    min_radius=350,
    elevation=(3, 15),
    inward=False,
):
    """
    Entrances close together, exits far away by walking distance; each door is a point on
    the room's surface plus an orientation pointing away from it.

    elevation: how far above the floor edge the door sits, in degrees (min, max)
    inward:    False = the vector points out of the room (away from its surface);
               True  = into the room
    (the other arguments are as before)
    Returns {"entrances": [...], "exits": [...], "report": [...], "connected_share": ...};
    each door has "position", "normal", "yaw", "pitch", "roll" (radians), "ypr_deg",
    plus the sphere it belongs to ("polyline", "sphere", "center", "radius").
    """
    adj, C, R, owner = room_graph(room, climb_penalty=climb_penalty)
    comp = _main_component(adj)
    main = np.array([i for i in comp if R[i] >= min_radius])
    if len(main) < n_entrances + n_exits:
        raise MissingEntranceOrExit_new(
            "the room's main connected piece is too small for that many doors"
        )
    ctx = SimpleNamespace(
        adj=adj, C=C, R=R, links=_links(room[0]), centroid=C[main].mean(0)
    )

    doors = {}

    def door_for(i):
        if i not in doors:
            doors[i] = _surface_door(i, ctx, rng, elevation, inward)
        return doors[i]

    def pick(idx, weights):
        """Weighted draw that skips spheres where no door fits; None if none do."""
        idx = [int(i) for i in idx]
        w = np.maximum(np.asarray(weights, dtype=float), 1e-9).tolist()
        while idx:
            k = rng.choices(range(len(idx)), weights=w)[0]
            if door_for(idx[k]) is not None:
                return idx[k]
            idx.pop(k)
            w.pop(k)
        return None

    # entrances
    if entrance_mode == "peripheral":
        d0 = walk_distance(adj, [int(rng.choice(list(main)))])
        far = main[d0[main] >= variety * d0[main].max()]
        a = pick(far, d0[far])
    else:
        a = None
    if a is None:
        a = pick(main, np.ones(len(main)))
    ents = [a]
    for _ in range(n_entrances - 1):
        dnear = walk_distance(adj, ents)
        lo, hi = gap[0] * (R[a] + R[main]), gap[1] * (R[a] + R[main])
        ok = (dnear[main] >= lo) & (dnear[main] <= hi) & ~np.isin(main, ents)
        b = pick(main[ok], np.ones(int(ok.sum()))) if ok.any() else None
        if b is None:  # nothing in range: the nearest valid sphere
            rest = main[~np.isin(main, ents)]
            b = pick(rest, 1.0 / (1.0 + dnear[rest]))
        ents.append(b)

    # exits
    dE = walk_distance(adj, ents)
    exits = []
    for _ in range(n_exits):
        score = dE.copy()
        if exits and exit_spread > 0:
            score = np.minimum(dE, walk_distance(adj, exits) / exit_spread)
        s = np.where(np.isin(main, ents + exits), -np.inf, score[main])
        e = pick(
            main[s >= variety * s.max()], score[main[s >= variety * s.max()]]
        )
        if e is None:
            rest = main[np.isfinite(s)]
            e = pick(rest, score[rest])
        if e is None:
            raise MissingEntranceOrExit_new("no valid exit position found")
        exits.append(e)

    def pack(i):
        d = door_for(i)
        gi, k = owner[i]
        return {
            **d,
            "index": i,
            "polyline": gi,
            "sphere": k,
            "center": tuple(float(x) for x in C[i]),
            "radius": float(R[i]),
        }

    report = [
        {
            "exit": i,
            "walk": float(dE[i]),
            "straight": float(min(np.linalg.norm(C[i] - C[e]) for e in ents)),
            "ratio": float(dE[i])
            / max(float(min(np.linalg.norm(C[i] - C[e]) for e in ents)), 1e-9),
        }
        for i in exits
    ]
    return {
        "entrances": [pack(i) for i in ents],
        "exits": [pack(i) for i in exits],
        "report": report,
        "connected_share": len(comp) / len(C),
    }
