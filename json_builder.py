import random
import numpy as np
import math


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


def generate_PE_features(circles):
    # For PE we choose a few circles at random and put the minehead there.
    number_of_mineheads = random.randrange(1, 4) if len(circles) > 4 else 1
    chosen_circles = random.choices(range(len(circles)), k=number_of_mineheads)
    return [
        (
            circles[circle]["center"][0],
            circles[circle]["center"][1],
            circles[circle]["center"][2] + 200,
        )
        for circle in chosen_circles
    ]


def generate_entrances(circles, graph):

    spheres_w_entrance = [0]
    # First decide the entrances:
    circles_w_one_neighbor = [
        k for k in graph if len(graph[k]) == 1 and k != 0 and 0 not in graph[k]
    ]
    if not circles_w_one_neighbor:
        spheres_w_exit = [random.randrange(1, len(circles))]
    else:
        spheres_w_exit = circles_w_one_neighbor

    entrances = []
    for ii, t in [(spheres_w_entrance, "Entrance"), (spheres_w_exit, "Exit")]:
        for sphere_idx in ii:
            sphere_coords = (
                circles[sphere_idx]["center"][0],
                circles[sphere_idx]["center"][1],
            )
            neighbors = graph[sphere_idx]
            neighbor_coords = [
                (
                    circles[circle]["center"][0],
                    circles[circle]["center"][1],
                    circles[circle]["radius"],
                )
                for circle in neighbors
            ]
            angle, _ = most_free_angle(neighbor_coords, origin=sphere_coords)
            angle = 0 if angle is None else angle
            if t == "Entrance":
                vector_angle = np.degrees((angle + np.pi) % (2 * np.pi))
            else:
                vector_angle = np.degrees(angle)
            entrances.append(
                (
                    circles[sphere_idx]["center"][0]
                    + circles[sphere_idx]["radius"] * math.cos(angle),
                    circles[sphere_idx]["center"][1]
                    + circles[sphere_idx]["radius"] * math.sin(angle),
                    circles[sphere_idx]["center"][2]
                    + circles[sphere_idx]["radius"] * 0.2,
                    0,
                    0,
                    vector_angle,
                    t,
                )
            )

    return entrances


def build_json(
    traversal_list, circles, entrances, room_num, key, settings, pe_features
) -> dict:

    result = {"FloodFillLines": {}}
    lines = {"Points": []}
    for ii in traversal_list:
        circle = circles[ii]
        lines["Points"].append(
            {
                "Location": {
                    "X": circle["center"][0],
                    "Y": circle["center"][1],
                    "Z": circle["center"][2],
                },
                "HRange": circle["radius"],
                "VRange": circle["radius"],
            }
        )
    result["FloodFillLines"]["FFill_0"] = lines

    result["Entrances"] = {}
    for ii, entrance in enumerate(entrances):
        result["Entrances"].update(
            {
                f"Entrance_{ii}": {
                    "Location": {
                        "X": entrance[0],
                        "Y": entrance[1],
                        "Z": entrance[2],
                    },
                    "Type": entrance[-1],
                    "Direction": {
                        "Roll": entrance[3],
                        "Pitch": entrance[4],
                        "Yaw": entrance[5],
                    },
                }
            }
        )

    if pe_features:
        result["PE_MiningHead"] = {}
        for ii, feature in enumerate(pe_features):
            result["PE_MiningHead"].update(
                {
                    f"MineHead_{ii}": {
                        "Location": {"X": feature[0], "Y": feature[1], "Z": feature[2]}
                    }
                }
            )

    result["Name"] = f"RMA_ICGEN_{key}{room_num}"
    result["Bounds"] = settings["bounds"]
    result["Tags"] = settings["tags"]

    return result
