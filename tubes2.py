import math, random, itertools
from collections import Counter
import numpy as np


def _tube(a, ra, b, rb):
    return [
        {"center": tuple(map(float, a)), "radius": float(ra)},
        {"center": tuple(map(float, b)), "radius": float(rb)},
    ]


def _fit(X, size, z_scale=1.0):
    """Center the points, scale the largest extent to `size`, then squash z."""
    X = np.asarray(X, dtype=float)
    X = X - (X.min(0) + X.max(0)) / 2
    X = X * (size / max(np.ptp(X, axis=0).max(), 1e-9))
    X[:, 2] *= z_scale
    return X


# ============================ percolation lattice ============================
_THRESH = {"cubic": 0.2488, "diamond": 0.3893}  # bond-percolation thresholds


def _lattice_edges(kind, n):
    """Edges between integer vertices; the diamond lattice uses quarter-cell units."""
    edges = []
    if kind == "cubic":
        for x, y, z in itertools.product(range(n), repeat=3):
            for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                v = (x + d[0], y + d[1], z + d[2])
                if max(v) < n:
                    edges.append(((x, y, z), v))
        return edges
    fcc = [(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)]
    bonds = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    for x, y, z in itertools.product(range(n), repeat=3):
        for f in fcc:
            u = (4 * x + f[0], 4 * y + f[1], 4 * z + f[2])
            for b in bonds:
                v = (u[0] + b[0], u[1] + b[1], u[2] + b[2])
                if all(0 <= c <= 4 * n for c in v):
                    edges.append((u, v))
    return edges


def _largest_component(edges):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        parent[find(u)] = find(v)
    groups = {}
    for e in edges:
        groups.setdefault(find(e[0]), []).append(e)
    return max(groups.values(), key=len) if groups else []


def lattice_room(
    rng=random,
    kind=None,
    cells=None,
    p=None,
    size=6000.0,
    radius=0.05,
    jitter=0.18,
    prune=2,
    z_scale=0.7,
    min_tubes=14,
):
    """
    Random subset of a cubic or diamond lattice's edges (bond percolation).
    kind:      "cubic" or "diamond" (None = random)
    cells:     lattice cells per side (default 4 for cubic, 2 for diamond)
    p:         chance each edge is kept (None = just above the percolation
               threshold; higher = denser, more loops)
    jitter:    random vertex displacement, as a fraction of the edge length
    prune:     passes that remove dead-end tubes (0 keeps all of them)
    z_scale:   <1 flattens the room vertically
    min_tubes: retry until the connected network has at least this many tubes
    """
    kind = kind or rng.choice(("cubic", "diamond"))
    cells = cells or (4 if kind == "cubic" else 2)
    base = _lattice_edges(kind, cells)
    p = _THRESH[kind] + rng.uniform(0.05, 0.15) if p is None else p

    best = []
    for _ in range(30):
        comp = _largest_component([e for e in base if rng.random() < p])
        if len(comp) > len(best):
            best = comp
        if len(best) >= min_tubes:
            break
    if not best:
        raise ValueError("no connected network; raise p")

    for _ in range(prune):  # drop edges ending in a dead end
        deg = Counter(x for e in best for x in e)
        pruned = [(u, v) for u, v in best if deg[u] > 1 and deg[v] > 1]
        if len(pruned) < 3:
            break
        best = pruned

    unit = 1.0 if kind == "cubic" else 0.25
    bond = unit if kind == "cubic" else unit * math.sqrt(3)
    verts = sorted({x for e in best for x in e})
    idx = {v: i for i, v in enumerate(verts)}
    X = np.array(
        [
            np.array(v, float) * unit
            + np.array([rng.uniform(-1, 1) for _ in range(3)]) * jitter * bond
            for v in verts
        ]
    )
    X = _fit(X, size, z_scale)

    deg = Counter(x for e in best for x in e)
    r0 = radius * size
    rv = {
        v: r0 * (0.75 + 0.15 * deg[v]) * rng.uniform(0.9, 1.1) for v in verts
    }  # bigger junctions
    tubes = [_tube(X[idx[u]], rv[u], X[idx[v]], rv[v]) for u, v in best]
    return tubes, [[0, 1] for _ in tubes]


# ========================== projected 4D polytopes ===========================
def _polytope(name):
    if name == "tesseract":
        V = list(itertools.product((-1, 1), repeat=4))
    elif name == "16-cell":
        V = [
            tuple(s if k == i else 0 for k in range(4))
            for i in range(4)
            for s in (-1, 1)
        ]
    elif name == "24-cell":
        V = []
        for i, j in itertools.combinations(range(4), 2):
            for si, sj in itertools.product((-1, 1), repeat=2):
                v = [0] * 4
                v[i], v[j] = si, sj
                V.append(tuple(v))
    else:
        raise ValueError(name)
    V = np.array(V, dtype=float)
    D = np.linalg.norm(V[:, None] - V[None], axis=-1)
    dmin = D[D > 1e-9].min()  # edges = the shortest vertex pairs
    edges = [
        (i, j)
        for i in range(len(V))
        for j in range(i + 1, len(V))
        if abs(D[i, j] - dmin) < 1e-6
    ]
    return V, edges


def _rot4(rng):
    R = np.eye(4)
    for i, j in itertools.combinations(
        range(4), 2
    ):  # one rotation per coordinate plane
        a = rng.uniform(-math.pi, math.pi)
        G = np.eye(4)
        G[i, i] = G[j, j] = math.cos(a)
        G[i, j], G[j, i] = -math.sin(a), math.sin(a)
        R = G @ R
    return R


def polytope_room(
    rng=random,
    name="16-cell",
    size=6000.0,
    radius=0.05,
    keep=0.7,
    z_scale=0.7,
    perspective=None,
    depth_taper=True,
):
    """
    Edges of a randomly rotated 4D polytope, perspective-projected into 3D.
    name:        "tesseract" (32 edges), "16-cell" (24) or "24-cell" (96); None = random
    keep:        fraction of the non-essential edges kept; a random spanning
                 tree is always kept, so the room stays connected
    perspective: camera distance in units of the polytope radius (None = random 2-3);
                 smaller = stronger perspective = bigger near/far contrast
    depth_taper: nearer vertices (in 4D) get wider tubes
    """
    name = name or rng.choice(("tesseract", "16-cell", "24-cell"))
    V, edges = _polytope(name)
    rng.shuffle(edges)

    parent = list(range(len(V)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kept = []
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            kept.append((i, j))  # spanning-tree edge: always kept
        elif rng.random() < keep:
            kept.append((i, j))

    W = V @ _rot4(rng).T
    d = np.linalg.norm(V, axis=1).max() * (
        rng.uniform(2.0, 3.0) if perspective is None else perspective
    )
    f = d / (d - W[:, 3])  # perspective factor per vertex
    X = _fit(W[:, :3] * f[:, None], size, z_scale)
    rv = radius * size * (f / f.mean() if depth_taper else np.ones(len(V)))
    tubes = [_tube(X[i], rv[i], X[j], rv[j]) for i, j in kept]
    return tubes, [[0, 1] for _ in tubes]
