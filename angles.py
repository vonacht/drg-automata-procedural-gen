import math
import numpy as np

def most_free_angle(circles, origin=(0.0, 0.0)):
    """
    circles: list of (x, y, r)
    origin: (ox, oy)

    returns:
        max_free_angle (radians),
        best_direction (radians, angle from +x axis)
    """

    ox, oy = origin
    intervals = []

    for x, y, r in circles:
        dx = x - ox
        dy = y - oy
        d = math.hypot(dx, dy)

        # Center is fully covered
        if d <= r:
            return 0.0, None

        theta = math.atan2(dy, dx)
        alpha = math.asin(min(1.0, r / d))

        start = theta - alpha
        end = theta + alpha

        # Normalize to [0, 2π)
        start = start % (2 * math.pi)
        end = end % (2 * math.pi)

        if start <= end:
            intervals.append((start, end))
        else:
            # Wrap-around interval
            intervals.append((start, 2 * math.pi))
            intervals.append((0.0, end))

    if not intervals:
        # No blockers at all
        return 2 * math.pi, 0.0

    # Sort and merge intervals
    intervals.sort()
    merged = [intervals[0]]

    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    # Find largest gap
    max_gap = 0.0
    best_angle = None

    # Gaps between intervals
    for i in range(len(merged) - 1):
        gap = merged[i + 1][0] - merged[i][1]
        if gap > max_gap:
            max_gap = gap
            best_angle = (merged[i][1] + merged[i + 1][0]) / 2

    # Circular gap (end → start + 2π)
    wrap_gap = (merged[0][0] + 2 * math.pi) - merged[-1][1]
    if wrap_gap > max_gap:
        max_gap = wrap_gap
        best_angle = (merged[-1][1] + merged[0][0] + 2 * math.pi) / 2
        best_angle %= 2 * math.pi

    return best_angle, max_gap

center = (5, 3)
neighbors = [(6, 5, 1), (2, 7, 1), (2,2,1)]
angle, value = most_free_angle(neighbors, origin=center)
print(np.degrees(angle), value)


