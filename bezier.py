import numpy as np
import random


def random_control_points(A, B, num_controls=2, spread=0.3):
    """
    Generate random control points between A and B.

    spread: controls how far points can deviate from the straight line
    """
    A = np.array(A)
    B = np.array(B)
    direction = B - A
    length = np.linalg.norm(direction)

    if length == 0:
        return [A.copy() for _ in range(num_controls)]

    direction_unit = direction / length

    # Perpendicular vector
    perp = np.array([-direction_unit[1], direction_unit[0]])

    controls = []
    for i in range(num_controls):
        t = random.uniform(0, 1)

        # Base point along the line
        base = A + t * direction

        # Offset perpendicular to the line
        offset_mag = random.uniform(-spread, spread) * length
        offset = perp * offset_mag

        controls.append(base + offset)

    return controls


def bezier_curve(points, num_samples=100):
    """
    Evaluate a Bézier curve for arbitrary number of control points.
    Uses De Casteljau's algorithm.
    """
    points = np.array(points)
    n = len(points)

    t_values = np.linspace(0, 1, num_samples)
    curve = []

    for t in t_values:
        temp = points.copy()
        for r in range(1, n):
            temp[: n - r] = (1 - t) * temp[: n - r] + t * temp[1 : n - r + 1]
        curve.append(temp[0])

    return np.array(curve)


def random_bezier_path(A, B, num_controls=2, num_samples=100, spread=0.3):
    """
    Generate a randomized Bézier path from A to B.

    num_controls:
        1 → quadratic Bézier
        2 → cubic Bézier
        3 → quartic (less common but works)
    """
    controls = random_control_points(A, B, num_controls, spread)

    # Full list of points defining the curve
    points = [np.array(A)] + controls + [np.array(B)]

    curve = bezier_curve(points, num_samples)

    return curve, points


def midpoint_displacement(A, B, depth=5, roughness=0.5, initial_disp=None):
    """
    Generate a fractal path between A and B using midpoint displacement.

    depth: recursion depth (more = more detail)
    roughness: how quickly randomness decreases (0 < r < 1)
    initial_disp: initial displacement magnitude (defaults to segment length)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    if initial_disp is None:
        initial_disp = np.linalg.norm(B - A)

    def recurse(p1, p2, disp, depth):
        if depth == 0:
            return [p1, p2]

        # Midpoint
        mid = (p1 + p2) / 2

        # მიმართულება (perpendicular)
        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length == 0:
            offset = np.zeros(2)
        else:
            dir_unit = direction / length
            perp = np.array([-dir_unit[1], dir_unit[0]])

            offset_mag = random.uniform(-disp, disp)
            offset = perp * offset_mag

        mid = mid + offset

        # Recurse on both halves
        left = recurse(p1, mid, disp * roughness, depth - 1)
        right = recurse(mid, p2, disp * roughness, depth - 1)

        # Merge (avoid duplicating midpoint)
        return left[:-1] + right

    return np.array(recurse(A, B, initial_disp, depth))
