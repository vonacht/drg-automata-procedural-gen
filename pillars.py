import numpy as np


def random_unit_vector():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)


def perpendicular_direction(direction):
    v = random_unit_vector()
    v -= v.dot(direction) * direction

    if np.linalg.norm(v) < 1e-5:
        v = np.cross(direction, [1, 0, 0])

        if np.linalg.norm(v) < 1e-5:
            v = np.cross(direction, [0, 1, 0])

    return v / np.linalg.norm(v)


def generate_formation(
    origin,
    length=5.0,
    radius=0.3,
    n_segments=6,
):
    """
    Generate a random cave formation.

    Returns:
        list[np.ndarray]: points describing the formation.
    """

    origin = np.asarray(origin, dtype=float)

    formation_type = np.random.choice(
        [
            "straight",
            "bent",
            "curved",
            "s_curve",
            "angled",
            "wavy",
            "cluster",
            "branching",
        ]
    )

    # direction_sign = np.random.choice([-1.0, 1.0])
    direction_sign = np.random.choice([1.0])

    tilt = np.random.uniform(0.0, 0.35)

    direction = np.array(
        [
            np.random.normal(0, tilt),
            np.random.normal(0, tilt),
            direction_sign,
        ]
    )
    direction /= np.linalg.norm(direction)

    # ---------------------------------------------------------
    # Cluster
    # ---------------------------------------------------------
    if formation_type == "cluster":
        points = []

        count = np.random.randint(3, 8)

        for _ in range(count):
            start = origin + np.random.normal(
                scale=radius * 0.7,
                size=3,
            )

            spike_length = length * np.random.uniform(0.4, 1.0)

            spike_dir = np.array(
                [
                    np.random.normal(0, 0.15),
                    np.random.normal(0, 0.15),
                    direction_sign,
                ]
            )
            spike_dir /= np.linalg.norm(spike_dir)

            side = perpendicular_direction(spike_dir)

            end = (
                start
                + spike_dir * spike_length
                + side * np.random.uniform(-radius, radius)
            )

            points.extend([start, end])

        return points

    # ---------------------------------------------------------
    # Branching
    # ---------------------------------------------------------
    if formation_type == "branching":
        points = []

        # Main stem
        stem_fraction = np.random.uniform(0.55, 0.7)
        stem_length = length * stem_fraction
        stem_segments = max(2, int(n_segments * stem_fraction))

        side = perpendicular_direction(direction)

        bend = np.random.uniform(-1.0, 1.0) * radius * 2.5

        for i in range(stem_segments + 1):
            t = i / stem_segments

            p = origin + direction * stem_length * t + side * bend * t**2

            p += np.random.normal(
                scale=radius * 0.05 * t,
                size=3,
            )

            points.append(p)

        # Branching starts at the end of the main stem.
        branch_origin = points[-1]

        n_branches = np.random.choice([2, 2, 2, 3])

        for branch_index in range(n_branches):
            branch_length = length * np.random.uniform(0.2, 0.45)

            branch_dir = perpendicular_direction(direction)

            # Give branches some outward spread and some
            # continuation in the original direction.
            branch_dir += direction * np.random.uniform(0.15, 0.4)
            branch_dir /= np.linalg.norm(branch_dir)

            branch_side = perpendicular_direction(branch_dir)

            branch_segments = max(
                2,
                int(n_segments * np.random.uniform(0.3, 0.5)),
            )

            bend = np.random.uniform(-1.0, 1.0) * radius * 2.0

            for i in range(1, branch_segments + 1):
                t = i / branch_segments

                p = (
                    branch_origin
                    + branch_dir * branch_length * t
                    + branch_side * bend * t**2
                )

                p += np.random.normal(
                    scale=radius * 0.05 * t,
                    size=3,
                )

                points.append(p)

        return points

    # ---------------------------------------------------------
    # Everything else
    # ---------------------------------------------------------
    points = [origin.copy()]

    side = perpendicular_direction(direction)

    for i in range(1, n_segments + 1):
        t = i / n_segments

        p = origin + direction * length * t

        if formation_type == "straight":
            offset = 0

        elif formation_type == "bent":
            offset = radius * 3.0 * t**2

        elif formation_type == "curved":
            offset = radius * 4.0 * t**2

        elif formation_type == "s_curve":
            offset = radius * 5.0 * np.sin(t * np.pi * 1.5)

        elif formation_type == "wavy":
            offset = radius * 2.5 * np.sin(t * np.pi * 3)

        elif formation_type == "angled":
            offset = radius * 5.0 * t

        p += side * offset

        p += np.random.normal(
            scale=radius * 0.08 * t,
            size=3,
        )

        points.append(p)

    return points
