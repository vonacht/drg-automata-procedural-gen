import math, random
import numpy as np

_M = lambda x, y: ("M", x, y)
_L = lambda x, y: ("L", x, y)
_A = lambda cx, cy, rx, ry, a0, a1: ("A", cx, cy, rx, ry, a0, a1)

# each glyph = list of strokes in a unit box (y up); each stroke becomes one polyline
GLYPHS = {
    "A": [[_M(0, 0), _L(0.5, 1), _L(1, 0)], [_M(0.2, 0.4), _L(0.8, 0.4)]],
    "B": [
        [
            _M(0, 0),
            _L(0, 1),
            _L(0.55, 1),
            _A(0.55, 0.75, 0.4, 0.25, 90, -90),
            _L(0, 0.5),
        ],
        [
            _M(0, 0.5),
            _L(0.6, 0.5),
            _A(0.6, 0.25, 0.4, 0.25, 90, -90),
            _L(0, 0),
        ],
    ],
    "C": [[_A(0.5, 0.5, 0.5, 0.5, 45, 315)]],
    "D": [
        [
            _M(0, 0),
            _L(0, 1),
            _L(0.45, 1),
            _A(0.45, 0.5, 0.5, 0.5, 90, -90),
            _L(0, 0),
        ]
    ],
    "E": [
        [_M(1, 1), _L(0, 1), _L(0, 0), _L(1, 0)],
        [_M(0, 0.5), _L(0.8, 0.5)],
    ],
    "F": [[_M(1, 1), _L(0, 1), _L(0, 0)], [_M(0, 0.5), _L(0.8, 0.5)]],
    "G": [[_A(0.5, 0.5, 0.5, 0.5, 40, 360), _L(0.55, 0.5)]],
    "H": [
        [_M(0, 0), _L(0, 1)],
        [_M(1, 0), _L(1, 1)],
        [_M(0, 0.5), _L(1, 0.5)],
    ],
    "I": [
        [_M(0.5, 0), _L(0.5, 1)],
        [_M(0.2, 1), _L(0.8, 1)],
        [_M(0.2, 0), _L(0.8, 0)],
    ],
    "J": [[_M(1, 1), _L(1, 0.3), _A(0.5, 0.3, 0.5, 0.3, 0, -180)]],
    "K": [[_M(0, 0), _L(0, 1)], [_M(1, 1), _L(0, 0.5), _L(1, 0)]],
    "L": [[_M(0, 1), _L(0, 0), _L(1, 0)]],
    "M": [[_M(0, 0), _L(0, 1), _L(0.5, 0.4), _L(1, 1), _L(1, 0)]],
    "N": [[_M(0, 0), _L(0, 1), _L(1, 0), _L(1, 1)]],
    "O": [[_A(0.5, 0.5, 0.5, 0.5, 0, 360)]],
    "P": [
        [
            _M(0, 0),
            _L(0, 1),
            _L(0.55, 1),
            _A(0.55, 0.75, 0.4, 0.25, 90, -90),
            _L(0, 0.5),
        ]
    ],
    "Q": [[_A(0.5, 0.5, 0.5, 0.5, 0, 360)], [_M(0.55, 0.35), _L(1, 0)]],
    "R": [
        [
            _M(0, 0),
            _L(0, 1),
            _L(0.55, 1),
            _A(0.55, 0.75, 0.4, 0.25, 90, -90),
            _L(0, 0.5),
        ],
        [_M(0.45, 0.5), _L(1, 0)],
    ],
    "S": [
        [_A(0.5, 0.75, 0.5, 0.25, 20, 270), _A(0.5, 0.25, 0.5, 0.25, 90, -160)]
    ],
    "T": [[_M(0, 1), _L(1, 1)], [_M(0.5, 1), _L(0.5, 0)]],
    "U": [
        [_M(0, 1), _L(0, 0.35), _A(0.5, 0.35, 0.5, 0.35, 180, 360), _L(1, 1)]
    ],
    "V": [[_M(0, 1), _L(0.5, 0), _L(1, 1)]],
    "W": [[_M(0, 1), _L(0.25, 0), _L(0.5, 0.6), _L(0.75, 0), _L(1, 1)]],
    "X": [[_M(0, 1), _L(1, 0)], [_M(1, 1), _L(0, 0)]],
    "Y": [[_M(0, 1), _L(0.5, 0.5), _L(1, 1)], [_M(0.5, 0.5), _L(0.5, 0)]],
    "Z": [[_M(0, 1), _L(1, 1), _L(0, 0), _L(1, 0)]],
}

# alternative drawings of a letter, chosen at random: more variety for free
ALT = {
    "A": [
        [
            [
                _M(0, 0),
                _L(0.143, 0.803),
                _A(0.5, 0.7, 0.38, 0.3, 160, 20),
                _L(1, 0),
            ],
            [_M(0.07, 0.4), _L(0.93, 0.4)],
        ]
    ],  # rounded top
}


def _stroke_points(stroke, rng, arc_jitter, open_loops, step=0.01):
    """Dense points along one stroke, in glyph units."""
    pts, last = [], len(stroke) - 1
    for k, seg in enumerate(stroke):
        if seg[0] in ("M", "L"):
            q = np.array(seg[1:], dtype=float)
            if seg[0] == "L" and pts:
                p = pts[-1]
                n = max(2, int(np.linalg.norm(q - p) / step))
                pts.extend(p + (q - p) * t for t in np.linspace(0, 1, n)[1:])
            else:
                pts.append(q)
        else:
            _, cx, cy, rx, ry, a0, a1 = seg
            if abs(a1 - a0) >= 359.9:  # a full loop: sometimes leave a gap
                if rng.random() < open_loops:
                    a1 -= math.copysign(rng.uniform(25, 55), a1 - a0)
            else:  # only free ends may change
                if k == 0:
                    a0 += rng.uniform(-arc_jitter, arc_jitter)
                if k == last:
                    a1 += rng.uniform(-arc_jitter, arc_jitter)
            n = max(
                4, int(abs(a1 - a0) / 360 * 2 * math.pi * max(rx, ry) / step)
            )
            t = np.radians(np.linspace(a0, a1, n))
            arc = np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]
            pts.extend(arc[1:] if pts else arc)
    return np.array(pts)


def _smooth(P, window):
    """Moving average that rounds corners; end points stay put."""
    if window < 3 or len(P) < window:
        return P
    k = np.ones(window) / window
    Q = np.pad(P, ((window // 2, window // 2), (0, 0)), mode="edge")
    out = np.c_[
        np.convolve(Q[:, 0], k, "valid"), np.convolve(Q[:, 1], k, "valid")
    ]
    out[0], out[-1] = P[0], P[-1]
    return out


def _resample(P, spacing):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return P[:1]
    n = max(2, int(round(d[-1] / spacing)) + 1)
    s = np.linspace(0, d[-1], n)
    return np.c_[np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])]


def letter_room(
    ch,
    rng=random,
    height=4500.0,
    plane="xy",
    offset=0.0,
    slant=None,
    aspect=None,
    wobble=None,
    stroke=None,
    round_corners=None,
    spacing=0.9,
    radius_var=0.15,
    arc_jitter=15.0,
    open_loops=0.15,
):
    """
    Builds a cave room shaped like the letter `ch`.
    Returns (polylines, paths): each polyline is a list of {"center", "radius"}
    dicts and each path is [0, 1, ..., n-1] for its polyline.

    height:        letter height in world units
    plane:         "xy" = lying on the floor, "xz" = standing up
    offset:        z (for "xy") or y (for "xz") of the letter plane
    slant:         shear, positive leans right (None = random in +-0.25)
    aspect:        width multiplier (None = random 0.75-1.2)
    wobble:        smooth wiggle of the whole letter, fraction of height (None = random)
    stroke:        passage radius as a fraction of height (None = random)
    round_corners: corner rounding in glyph units (None = random 0-0.12)
    spacing:       sphere spacing as a fraction of the stroke radius
    radius_var:    smooth radius variation along each stroke
    arc_jitter:    degrees by which free arc ends may stretch or shrink
    open_loops:    chance that a closed loop (O, Q) is left slightly open
    """
    key = ch if ch in GLYPHS else ch.upper()
    if key not in GLYPHS:
        raise ValueError(f"no glyph for {ch!r}")
    glyph = rng.choice([GLYPHS[key]] + ALT.get(key, []))

    height = random.uniform(height, height * 1.4)
    slant = rng.uniform(-0.25, 0.25) if slant is None else slant
    aspect = rng.uniform(0.75, 1.2) if aspect is None else aspect
    wobble = rng.uniform(0.0, 0.035) if wobble is None else wobble
    stroke = rng.uniform(0.1, 0.3) if stroke is None else stroke
    round_corners = (
        rng.uniform(0.0, 0.12) if round_corners is None else round_corners
    )
    r0 = stroke * height
    f1, f2, p1, p2 = (
        rng.uniform(0.8, 1.6),
        rng.uniform(0.8, 1.6),
        rng.random(),
        rng.random(),
    )

    def world(
        P,
    ):  # glyph units -> plane coordinates; the same warp for every stroke
        x, y = P[:, 0] - 0.5, P[:, 1] - 0.5
        X = (x * aspect + slant * y) * height
        Y = y * height
        X = X + wobble * height * np.sin(2 * np.pi * (f1 * P[:, 1] + p1))
        Y = Y + wobble * height * np.sin(2 * np.pi * (f2 * P[:, 0] + p2))
        return np.c_[X, Y]

    window = int(round(round_corners / 0.01)) | 1  # odd window length
    polys = []
    for strk in glyph:
        P = _smooth(_stroke_points(strk, rng, arc_jitter, open_loops), window)
        P = _resample(world(P), spacing * r0)
        s = np.linspace(0, 1, len(P))
        radii = r0 * (
            1
            + radius_var
            * np.sin(
                2 * np.pi * rng.uniform(1, 3) * s + rng.uniform(0, 2 * math.pi)
            )
        )
        radii = radii * np.array([1 + 0.05 * rng.uniform(-1, 1) for _ in s])
        polys.append(
            [
                {
                    "center": (
                        (float(x), float(y), offset)
                        if plane == "xy"
                        else (float(x), offset, float(y))
                    ),
                    "radius": float(r),
                }
                for (x, y), r in zip(P, radii)
            ]
        )

    return polys, [list(range(len(p))) for p in polys]


def shift_room_multiple(spheres, point):
    # shifted_room = []
    # for cluster in spheres:
    # shifted_room.append(shift_room(cluster, point))
    shift_vector = np.array(point) - np.array(spheres[0][0]["center"])
    shifted_room = []
    for cluster in spheres:
        shifted_room.append(
            [
                {
                    "center": (np.array(c["center"] + shift_vector)),
                    "radius": c["radius"],
                }
                for c in cluster
            ]
        )
    return shifted_room


def generate_letter_room():
    room, path = letter_room(random.choice(list(GLYPHS.keys())))
    return shift_room_multiple(room, (0, 0, 0)), path
