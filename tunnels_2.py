import numpy as np


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _basis(direction):
    """Return two vectors perpendicular to direction."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    # Pick a vector that isn't parallel to direction
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    side = np.cross(direction, tmp)
    side /= np.linalg.norm(side)

    up = np.cross(direction, side)
    up /= np.linalg.norm(up)

    return direction, side, up


def _interpolate(start, end, t):
    return start[None, :] + t[:, None] * (end - start)[None, :]


# ------------------------------------------------------------
# 1. Straight
# ------------------------------------------------------------


def straight(start, end, points=100):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    return np.linspace(start, end, points)


# ------------------------------------------------------------
# 2. Sine wave
# ------------------------------------------------------------


def sine(start, end, amplitude=10, waves=1.5, points=100):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction, side, _ = _basis(end - start)

    t = np.linspace(0, 1, points)
    path = _interpolate(start, end, t)

    offset = amplitude * np.sin(t * waves * 2 * np.pi)

    path += offset[:, None] * side

    return path


# ------------------------------------------------------------
# 3. Spiral
# ------------------------------------------------------------


def spiral(
    center,
    start_radius=20,
    end_radius=5,
    height=20,
    turns=2,
    points=200,
):
    center = np.asarray(center, dtype=float)

    t = np.linspace(0, 1, points)

    angle = t * turns * 2 * np.pi
    radius = start_radius + (end_radius - start_radius) * t

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = height * t

    return center + np.column_stack((x, y, z))


# ------------------------------------------------------------
# 4. Helix
# ------------------------------------------------------------


def helix(
    center,
    radius=15,
    height=30,
    turns=2,
    points=200,
):
    center = np.asarray(center, dtype=float)

    t = np.linspace(0, 1, points)
    angle = t * turns * 2 * np.pi

    return center + np.column_stack(
        (
            radius * np.cos(angle),
            radius * np.sin(angle),
            height * t,
        )
    )


# ------------------------------------------------------------
# 5. Expanding spiral
# ------------------------------------------------------------


def expanding_spiral(
    center,
    start_radius=2,
    end_radius=30,
    height=10,
    turns=1.5,
    points=200,
):
    center = np.asarray(center, dtype=float)

    t = np.linspace(0, 1, points)

    angle = turns * 2 * np.pi * t
    radius = start_radius + (end_radius - start_radius) * t

    return center + np.column_stack(
        (
            radius * np.cos(angle),
            radius * np.sin(angle),
            height * t,
        )
    )


# ------------------------------------------------------------
# 6. Circle / loop
# ------------------------------------------------------------


def loop(
    center,
    radius=20,
    height=0,
    points=150,
):
    center = np.asarray(center, dtype=float)

    t = np.linspace(0, 2 * np.pi, points)

    return center + np.column_stack(
        (
            radius * np.cos(t),
            radius * np.sin(t),
            height * np.sin(t * 2),
        )
    )


# ------------------------------------------------------------
# 7. Figure eight
# ------------------------------------------------------------


def figure_eight(
    center,
    size=20,
    height=0,
    points=200,
):
    center = np.asarray(center, dtype=float)

    t = np.linspace(0, 2 * np.pi, points)

    # Lemniscate-like curve
    x = size * np.sin(t)
    y = size * np.sin(t) * np.cos(t)
    z = height * np.sin(t * 2)

    return center + np.column_stack((x, y, z))


# ------------------------------------------------------------
# 8. Lissajous
# ------------------------------------------------------------


def lissajous(
    center,
    size=(20, 20, 10),
    frequencies=(2, 3, 1),
    phase=np.pi / 2,
    points=300,
):
    center = np.asarray(center, dtype=float)

    sx, sy, sz = size
    fx, fy, fz = frequencies

    t = np.linspace(0, 2 * np.pi, points)

    x = sx * np.sin(fx * t + phase)
    y = sy * np.sin(fy * t)
    z = sz * np.sin(fz * t)

    return center + np.column_stack((x, y, z))


# ------------------------------------------------------------
# 9. Random worm
# ------------------------------------------------------------


def random_worm(
    start,
    direction,
    length=100,
    step=1,
    turn=0.2,
    vertical_bias=1.0,
):
    position = np.asarray(start, dtype=float).copy()
    direction = np.asarray(direction, dtype=float).copy()

    direction /= np.linalg.norm(direction)

    result = [position.copy()]

    for _ in range(int(length / step)):
        random_turn = np.random.normal(0, turn, 3)

        # Control how much the tunnel can move vertically
        random_turn[2] *= vertical_bias

        direction += random_turn
        direction /= np.linalg.norm(direction)

        position += direction * step

        result.append(position.copy())

    return np.asarray(result)


# ------------------------------------------------------------
# 10. Smooth random worm
# ------------------------------------------------------------


def smooth_worm(
    start,
    direction,
    length=100,
    step=1,
    turn=0.15,
    vertical_bias=0.5,
    smoothing=0.9,
):
    position = np.asarray(start, dtype=float).copy()
    direction = np.asarray(direction, dtype=float).copy()
    direction /= np.linalg.norm(direction)

    result = [position.copy()]

    for _ in range(int(length / step)):
        random_direction = np.random.normal(0, 1, 3)
        random_direction[2] *= vertical_bias

        new_direction = direction + random_direction * turn

        direction = smoothing * direction + (1 - smoothing) * new_direction

        direction /= np.linalg.norm(direction)

        position += direction * step
        result.append(position.copy())

    return np.asarray(result)


# ------------------------------------------------------------
# 11. Random spline-like path
# ------------------------------------------------------------


def random_spline(
    start,
    end,
    amplitude=20,
    waves=3,
    points=200,
):
    """
    Smooth random-ish path between start and end.
    Uses several sine waves with random amplitudes.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction, side, up = _basis(end - start)

    t = np.linspace(0, 1, points)

    path = _interpolate(start, end, t)

    # Random low-frequency components
    rng = np.random.default_rng()

    offset_side = np.sin(t * 2 * np.pi * waves) * rng.uniform(0.5, 1.0)

    offset_up = np.sin(t * 2 * np.pi * (waves - 0.5)) * rng.uniform(0.2, 0.8)

    path += amplitude * offset_side[:, None] * side
    path += amplitude * 0.5 * offset_up[:, None] * up

    return path


# ------------------------------------------------------------
# 12. Corkscrew
# ------------------------------------------------------------


def corkscrew(
    start,
    end,
    radius=10,
    turns=3,
    points=250,
):
    """
    A spiral wrapped around the straight line from start to end.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction, side, up = _basis(end - start)

    t = np.linspace(0, 1, points)
    angle = t * turns * 2 * np.pi

    centerline = _interpolate(start, end, t)

    offset = np.cos(angle)[:, None] * side + np.sin(angle)[:, None] * up

    return centerline + radius * offset


def generate_tunnel(
    start,
    direction,
    length=100,
    step=1.0,
    max_segments=3,
    branch_probability=0.25,
    seed=None,
):
    """
    Generate a procedural 3D tunnel centerline.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing tunnel center coordinates.
    """

    rng = np.random.default_rng(seed)

    start = np.asarray(start, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    # --------------------------------------------------------
    # Available tunnel generators
    # --------------------------------------------------------

    def make_segment(kind, start, direction, length):
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)

        if kind == "straight":
            end = start + direction * length
            return straight(start, end, max(2, int(length / step)))

        elif kind == "sine":
            end = start + direction * length

            return sine(
                start,
                end,
                amplitude=rng.uniform(200, 450),
                waves=rng.uniform(0.5, 2.0),
                points=max(10, int(length / step)),
            )

        elif kind == "worm":
            return random_worm(
                start,
                direction,
                length=length,
                step=step,
                turn=rng.uniform(0.05, 0.20),
                vertical_bias=rng.uniform(0.2, 0.8),
            )

        elif kind == "smooth_worm":
            return smooth_worm(
                start,
                direction,
                length=length,
                step=step,
                turn=rng.uniform(0.05, 0.15),
                vertical_bias=rng.uniform(0.2, 0.7),
                smoothing=rng.uniform(0.8, 0.95),
            )

        elif kind == "spline":
            end = start + direction * length

            return random_spline(
                start,
                end,
                amplitude=rng.uniform(200, 450),
                waves=rng.uniform(1, 2),
                points=max(20, int(length / step)),
            )

        elif kind == "spiral":
            # Spiral needs a local center.
            _, side, up = _basis(direction)

            center = start + direction * (length * 0.5)

            return spiral(
                center=center,
                start_radius=rng.uniform(50, 150),
                end_radius=rng.uniform(50, 150),
                height=rng.uniform(length * 0.6, length),
                turns=rng.uniform(0.75, 2.5),
                points=max(50, int(length / step)),
            )

        elif kind == "helix":
            center = start

            return helix(
                center=center,
                radius=rng.uniform(50, 300),
                height=rng.uniform(length * 0.5, length),
                turns=rng.uniform(1, 3),
                points=max(50, int(length / step)),
            )

        elif kind == "loop":
            return loop(
                center=start,
                radius=rng.uniform(200, 250),
                height=rng.uniform(50, 200),
                points=max(50, int(length / step)),
            )

        elif kind == "figure_eight":
            return figure_eight(
                center=start,
                size=rng.uniform(300, 400),
                height=rng.uniform(50, 200),
                points=max(50, int(length / step)),
            )

        elif kind == "lissajous":
            return lissajous(
                center=start,
                size=(
                    rng.uniform(150, 200),
                    rng.uniform(150, 200),
                    rng.uniform(150, 200),
                ),
                frequencies=(
                    rng.integers(1, 4),
                    rng.integers(1, 4),
                    rng.integers(1, 3),
                ),
                points=max(100, int(length / step)),
            )

        elif kind == "corkscrew":
            end = start + direction * length

            return corkscrew(
                start,
                end,
                radius=rng.uniform(150, 200),
                turns=rng.uniform(1, 2),
                points=max(50, int(length / step)),
            )

        raise ValueError(f"Unknown tunnel type: {kind}")

    # --------------------------------------------------------
    # Shape selection
    # --------------------------------------------------------

    types = [
        "sine",
        "spline",
        "spiral",
        "helix",
        "loop",
        "figure_eight",
        "lissajous",
        "corkscrew",
    ]

    weights = [
        1,
        1,
        1,
        1,
        1,
        1,
        0.1,
        0.1,
    ]

    # --------------------------------------------------------
    # Generate segments
    # --------------------------------------------------------

    segments = []

    current_start = start.copy()
    current_direction = direction.copy()

    for segment_index in range(max_segments):

        kind = rng.choice(types, p=np.asarray(weights) / sum(weights))
        print(kind)

        # Don't make every segment crazy.
        segment_length = length / max_segments

        segment = make_segment(
            kind,
            current_start,
            current_direction,
            segment_length,
        )

        # Avoid duplicate point at segment boundaries.
        if segments:
            segment = segment[1:]

        segments.append(segment)

        # New start
        current_start = segment[-1]

        # Estimate direction from final two points.
        if len(segment) >= 2:
            current_direction = segment[-1] - segment[-2]

            norm = np.linalg.norm(current_direction)

            if norm > 0:
                current_direction /= norm

        # ----------------------------------------------------
        # Branching
        # ----------------------------------------------------

        if (
            segment_index < max_segments - 1
            and rng.random() < branch_probability
        ):
            # Pick a point somewhere in the existing segment.
            branch_index = rng.integers(
                max(1, len(segment) // 4),
                len(segment) - 1,
            )

            branch_start = segment[branch_index]

            # Create a direction perpendicular-ish to main path.
            random_dir = rng.normal(size=3)

            branch_direction = current_direction + random_dir * rng.uniform(
                0.5, 1.5
            )

            branch_direction /= np.linalg.norm(branch_direction)

            branch_length = length * rng.uniform(0.2, 0.5)

            branch_type = rng.choice(
                [
                    "worm",
                    "smooth_worm",
                    "sine",
                    "spline",
                    "spiral",
                ],
                p=[
                    0.35,
                    0.30,
                    0.15,
                    0.15,
                    0.05,
                ],
            )

            branch = make_segment(
                branch_type,
                branch_start,
                branch_direction,
                branch_length,
            )

            segments.append(branch)

    return np.concatenate(segments)
