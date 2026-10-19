import math, random
import numpy as np


def _axis_angle(axis, ang):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


PRESETS = {
    # winding main passage that sprouts side branches ending in chambers
    "spine": dict(
        axiom="FF",
        iterations=3,
        angle=0.9,
        decay=0.85,
        droop=0.05,
        rules={"F": [(0.5, "F[+!FC]F"), (0.3, "F[-!FC]F"), (0.2, "F&F")]},
    ),
    # tree-like network that fans out in 3D
    "dendrite": dict(
        axiom="F",
        iterations=3,
        angle=0.7,
        decay=0.8,
        droop=0.0,
        rules={"F": [(0.6, "F[+!F][-!F]"), (0.4, "F[/+!F]F")]},
    ),
    # zigzag that keeps dropping
    "switchback": dict(
        axiom="F",
        iterations=3,
        angle=0.6,
        decay=1.0,
        droop=0.1,
        rules={"F": [(0.5, "F+F&F"), (0.3, "F-F&F"), (0.2, "F[+F]F&F")]},
    ),
}

PRESETS["spine"] = dict(
    axiom="SSSSSSSS",
    iterations=1,  # 8 segments, each rewritten once
    angle=0.4,
    jitter=0.3,
    decay=0.65,
    droop=0.04,
    chamber=1.7,
    rules={
        "S": [
            (0.28, "F+F"),  # gentle bend left
            (0.28, "F-F"),  # gentle bend right
            (0.14, "FF"),  # straight
            (0.09, "F(++!FFFF--FF)F"),  # loop branch, left
            (0.09, "F(--!FFFF++FF)F"),  # loop branch, right
            (0.04, "F[++!FC]F"),  # rare dead-end alcove
            (0.04, "F&F"),  # slight drop
            # remaining weight is implicit: weights are normalized by rng.choices
        ]
    },
    loops=dict(min_gap=3, max_len=5.0),
)


def expand(axiom, rules, iterations, rng, max_len=1500):
    word = axiom
    for _ in range(iterations):
        out = []
        for ch in word:
            opts = rules.get(ch)
            if opts is None:
                out.append(ch)
            else:
                out.append(
                    rng.choices([s for _, s in opts], [p for p, _ in opts])[0]
                )
        new = "".join(out)
        if len(new) > max_len:  # stop before the string explodes
            break
        word = new
    return word


def interpret(
    word,
    rng,
    radius=800,
    stride=1.3,
    angle=0.7,
    jitter=0.25,
    decay=0.8,
    chamber=1.7,
    droop=0.0,
    max_spheres=150,
    radius_jitter=0.15,
    radius_drift=0.08,
):
    R = np.eye(3)
    pos = np.zeros(3)
    r = float(radius)
    w = 1.0  # slowly varying width multiplier
    polys = [[{"center": tuple(pos), "radius": r}]]
    meta = [None]
    cur, stack, count = 0, [], 1

    def turn(ang):
        return ang * (1 + rng.uniform(-jitter, jitter))

    for ch in word:
        if count >= max_spheres:
            break
        if ch == "F":
            pos = pos + R[:, 0] * r * stride
            # drift: random walk pulled back toward 1, clamped
            w = min(
                max(w + rng.gauss(0, radius_drift) + 0.3 * (1 - w), 0.6), 1.5
            )
            rad = r * w * (1 + rng.uniform(-radius_jitter, radius_jitter))
            polys[cur].append({"center": tuple(pos), "radius": rad})
            count += 1
            if droop:
                axis = np.cross(R[:, 0], (0, 0, -1))
                n = np.linalg.norm(axis)
                if n > 1e-9:
                    R = _axis_angle(axis, droop * math.asin(min(1.0, n))) @ R
        elif ch == "C":
            polys[cur][-1]["radius"] *= chamber
        elif ch in "+-":
            R = R @ _axis_angle(
                (0, 0, 1), (1 if ch == "+" else -1) * turn(angle)
            )
        elif ch in "&^":
            R = R @ _axis_angle(
                (0, 1, 0), (1 if ch == "&" else -1) * turn(angle)
            )
        elif ch == "/":
            R = R @ _axis_angle((1, 0, 0), turn(angle))
        elif ch == "!":
            r *= decay
        elif ch in "[(":
            stack.append((pos.copy(), R.copy(), r, cur, w))
            meta.append(
                {"parent": cur, "at": len(polys[cur]) - 1, "loop": ch == "("}
            )
            polys.append([{"center": tuple(pos), "radius": r}])
            cur = len(polys) - 1
            count += 1
        elif ch in "])" and stack:
            pos, R, r, cur, w = stack.pop()
    return polys, meta


def _lerp(a, b, t):
    return a + (b - a) * t


def close_loops(polys, meta, stride=1.3, min_gap=3, max_len=5.0):
    """Extends each loop branch to the nearest sphere of its parent polyline that is
    at least min_gap spheres away from where the branch left it. Loops that can't
    close (nothing suitable, or farther than max_len parent radii) are removed.
    """
    drop = set()
    for b, m in enumerate(meta):
        if m is None or not m["loop"]:
            continue
        branch, parent = polys[b], polys[m["parent"]]
        end = branch[-1]
        cands = [
            (math.dist(end["center"], s["center"]), j)
            for j, s in enumerate(parent)
            if abs(j - m["at"]) >= min_gap
        ]
        if len(branch) < 2 or not cands:
            drop.add(b)
            continue
        dist, j = min(cands)
        target = parent[j]
        if dist > max_len * target["radius"]:
            drop.add(b)
            continue
        # fill the gap with spheres spaced like the rest of the passage,
        # ending on a copy of the parent sphere
        k = max(0, math.ceil(dist / (end["radius"] * stride)) - 1)
        for i in range(1, k + 2):
            u = i / (k + 1)
            branch.append(
                {
                    "center": tuple(
                        _lerp(a, c, u)
                        for a, c in zip(end["center"], target["center"])
                    ),
                    "radius": _lerp(end["radius"], target["radius"], u),
                }
            )
    return [p for i, p in enumerate(polys) if i not in drop]


def scale_to_volume(polys, target):
    """Uniformly scales centers and radii so the sphere-sum volume equals target."""
    vol = sum(2 / 3 * math.pi * s["radius"] ** 3 for p in polys for s in p)
    k = (target / vol) ** (1 / 3)
    return [
        [
            {
                **s,
                "center": tuple(c * k for c in s["center"]),
                "radius": s["radius"] * k,
            }
            for s in p
        ]
        for p in polys
    ]


def generate_lsystem_room(
    rng=random,
    preset="spine",
    target_volume=None,
    radius=None,
    max_spheres=25,
    **overrides,
):
    cfg = dict(PRESETS[preset or rng.choice(list(PRESETS))])
    cfg.update(overrides)
    word = expand(cfg["axiom"], cfg["rules"], cfg["iterations"], rng)
    turtle = {
        k: cfg[k]
        for k in (
            "angle",
            "decay",
            "droop",
            "chamber",
            "stride",
            "jitter",
            "radius_jitter",
            "radius_drift",
        )
        if k in cfg
    }
    polys, meta = interpret(
        word,
        rng,
        radius=radius or cfg.get("radius", 800),
        max_spheres=max_spheres,
        **turtle,
    )
    polys = close_loops(
        polys, meta, stride=cfg.get("stride", 1.3), **cfg.get("loops", {})
    )
    polys = [p for p in polys if len(p) >= 2] or polys
    if target_volume:
        polys = scale_to_volume(polys, target_volume)

    paths = [list(range(len(p))) for p in polys]
    return polys, paths
