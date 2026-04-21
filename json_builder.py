import random
import numpy as np
import math


def most_free_angle(main_circle, surrounding_circles):
    mx, my, mR = main_circle
    blocked_intervals = []

    for sx, sy, sr in surrounding_circles:
        dx = sx - mx
        dy = sy - my
        d = math.sqrt(dx**2 + dy**2)

        if d == 0:
            if sr >= mR:
                return None, 0, "Main circle is entirely covered."
            continue

        # Law of Cosines
        cos_alpha = (mR**2 + d**2 - sr**2) / (2 * mR * d)

        if cos_alpha <= -1:
            return None, 0, "Main circle is entirely covered."
        if cos_alpha >= 1:
            continue

        alpha = math.acos(cos_alpha)
        phi = math.atan2(dy, dx)

        # Normalize to [0, 2pi]
        start = (phi - alpha) % (2 * math.pi)
        end = (phi + alpha) % (2 * math.pi)

        if start > end:
            blocked_intervals.append((start, 2 * math.pi))
            blocked_intervals.append((0, end))
        else:
            blocked_intervals.append((start, end))

    if not blocked_intervals:
        return 0.0, 2 * math.pi, "Entire perimeter is free. Midpoint set to 0."

    # Merge intervals
    blocked_intervals.sort()
    merged = []
    curr_s, curr_e = blocked_intervals[0]
    for next_s, next_e in blocked_intervals[1:]:
        if next_s <= curr_e:
            curr_e = max(curr_e, next_e)
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = next_s, next_e
    merged.append((curr_s, curr_e))

    # If the first and last intervals touch the 2pi boundary, merge them
    if len(merged) > 1 and merged[0][0] == 0 and merged[-1][1] == 2 * math.pi:
        new_start = merged[-1][0]
        new_end = merged[0][1]
        merged.pop()
        merged[0] = (new_start - 2 * math.pi, new_end)

    # Find the largest gap
    max_gap = -1
    best_mid = 0

    for i in range(len(merged)):
        next_idx = (i + 1) % len(merged)
        gap_start = merged[i][1]
        gap_end = merged[next_idx][0]

        if next_idx == 0:
            gap_end += 2 * math.pi

        gap_size = gap_end - gap_start
        if gap_size > max_gap:
            max_gap = gap_size
            # Calculate midpoint
            best_mid = (gap_start + gap_end) / 2

    # Normalize midpoint back to [0, 2pi]
    best_mid = best_mid % (2 * math.pi)

    return (
        best_mid,
        max_gap,
        f"Largest gap: {math.degrees(max_gap):.1f}°. Mid-angle: {math.degrees(best_mid):.1f}°.",
    )


def is_point_in_disk(px, py, cx, cy, r):
    """
    Check if point (px, py) is inside or on a disk
    centered at (cx, cy) with radius r.
    """
    return (px - cx) ** 2 + (py - cy) ** 2 <= r**2


def generate_PE_features(circles):
    # For PE we choose a few circles at random and put the minehead there.
    number_of_mineheads = random.randrange(3, 6) if len(circles) > 4 else 1
    # chosen_circles = random.choices(range(len(circles)), k=number_of_mineheads)
    chosen_circles = sorted(circles, key=lambda x: x["radius"], reverse=True)[
        :number_of_mineheads
    ]
    return [
        (
            circle["center"][0],
            circle["center"][1],
            circle["center"][2] + 200,
        )
        for circle in chosen_circles
        if not is_point_in_disk(
            0, 0, circle["center"][0], circle["center"][1], circle["radius"]
        )
    ]


def top_n_indices(arr, n):
    arr = np.asarray(arr)
    return np.argpartition(arr, -n)[-n:]


def ray_circle_intersection(P, v, C, r):
    """
    P: ray origin (tuple)
    v: ray direction (tuple, doesn't need to be normalized)
    C: circle center (tuple)
    r: radius (float)

    Returns the smallest positive t where intersection occurs, or None.
    """
    px, py = P
    vx, vy = v
    cx, cy = C

    # Vector from circle center to ray origin
    dx = px - cx
    dy = py - cy

    # Quadratic coefficients
    a = vx * vx + vy * vy
    b = 2 * (vx * dx + vy * dy)
    c = dx * dx + dy * dy - r * r

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return None  # no intersection

    sqrt_disc = math.sqrt(discriminant)

    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # We only care about intersections in front of the ray
    ts = [t for t in (t1, t2) if t >= 0]

    if not ts:
        return None

    return min(ts)  # closest intersection


def find_intersections(disks, P, v):
    """
    disks: list of (cx, cy, r)
    P: ray origin
    v: ray direction

    Returns:
        intersected_disks: list of disks that intersect
        farthest_disk: disk with largest intersection distance
    """
    intersections = []

    for disk in disks:
        (cx, cy, cz), r = disk["center"], disk["radius"]
        t = ray_circle_intersection(P, v, (cx, cy), r)

        if t is not None:
            intersections.append((disk, t))

    if not intersections:
        return [], None

    # Extract disks only
    intersected_disks = [disk for disk, _ in intersections]

    # Find farthest (max t)
    farthest_disk, max_t = max(intersections, key=lambda x: x[1])

    return intersected_disks, farthest_disk


def generate_exit_w_vector(circles, point, vector):
    _, farthest_disk = find_intersections(circles, point, vector)
    angle = math.asin(
        farthest_disk["center"][1]
        / np.sqrt(
            farthest_disk["center"][0] ** 2 + farthest_disk["center"][1] ** 2
        )
    )
    return [
        (
            farthest_disk["center"][0] + farthest_disk["radius"] * 0.8,
            farthest_disk["center"][1] + farthest_disk["radius"] * 0.8,
            farthest_disk["center"][2] + 150,
            0,
            0,
            np.rad2deg(angle),
            "Exit",
        )
    ]


def generate_entrances(circles, graph):

    spheres_w_entrance = [0]
    # First decide the entrances:
    circles_w_one_neighbor = [
        k for k in graph if len(graph[k]) == 1 and k != 0 and 0 not in graph[k]
    ]
    if not circles_w_one_neighbor:
        # spheres_w_exit = [random.randrange(1, len(circles))]
        coords_y = [c["center"][1] for c in circles]
        spheres_w_exit = [np.argmax(coords_y), np.argmin(coords_y)]
    else:
        spheres_w_exit = circles_w_one_neighbor + [
            x
            for x in random.choices(
                range(len(circles)), k=random.randrange(1, 3)
            )
            if x not in circles_w_one_neighbor and x != 0 and 0 not in graph[x]
        ]

    entrances = []
    for ii, t in [(spheres_w_entrance, "Entrance"), (spheres_w_exit, "Exit")]:
        for sphere_idx in ii:
            sphere_coords = (
                circles[sphere_idx]["center"][0],
                circles[sphere_idx]["center"][1],
                circles[sphere_idx]["radius"],
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
            angle, _, _ = most_free_angle(sphere_coords, neighbor_coords)
            angle = 0 if angle is None else angle
            vector_angle = np.degrees(angle)
            entrances.append(
                (
                    circles[sphere_idx]["center"][0]
                    + circles[sphere_idx]["radius"] * math.cos(angle),
                    circles[sphere_idx]["center"][1]
                    + circles[sphere_idx]["radius"] * math.sin(angle),
                    circles[sphere_idx]["center"][2]
                    + circles[sphere_idx].get("floor_depth", 0) * 0.8
                    + circles[sphere_idx]["radius"] * 0.2,
                    0,
                    0,
                    vector_angle,
                    t,
                )
            )

    return entrances


def generate_pillars_balls(circles, method=None):
    """
    Here we generate some random pillars, not too many, just for added decoration.
    """
    pillar_method = (
        random.choice(
            [
                "stalactite",
                "stalagmite",
                "arch",
                "triangle",
                "straight",
                "balls",
            ]
        )
        if method is None
        else method
    )
    pillar_method = "balls"
    pillars = []
    # number_of_pillars = (
    #     random.randrange(5) if len(circles) > 4 else len(circles) - 1
    # )
    suitable_circles = sorted(
        circles, key=lambda x: x["radius"], reverse=True
    )[:3]

    match pillar_method:
        case "balls":
            BALL_RADIUS = 500
            for c in suitable_circles:
                x = c["center"][0]
                y = c["center"][1]
                z = c["center"][2]
                pillars.append(
                    [
                        {
                            "Location": (x, y, z + 300),
                            "Range": BALL_RADIUS,
                            "NoiseRange": 100,
                        },
                        {
                            "Location": (x, y + 250, z + 300),
                            "Range": BALL_RADIUS,
                            "NoiseRange": 100,
                        },
                    ]
                )
    return pillars


def sample_point_in_disk(disk):
    cx, cy, z = disk["center"]
    r = disk["radius"] * 0.7

    # Random angle
    theta = random.uniform(0, 2 * math.pi)

    # Correct radius scaling for uniform distribution
    radius = r * math.sqrt(random.uniform(0, 1))

    # Convert to Cartesian coordinates
    x = cx + radius * math.cos(theta)
    y = cy + radius * math.sin(theta)

    return (x, y, z)


def generate_pillars(circles, method=None, height_mult=1):
    """
    Here we generate some random pillars, not too many, just for added decoration.
    """
    pillar_method = (
        random.choice(
            [
                "stalactite",
                "stalagmite",
                "straight",
                "balls",
                "none",
            ]
        )
        if method is None
        else method
    )
    print(pillar_method)
    pillars = []
    # number_of_pillars = (
    #     random.randrange(5) if len(circles) > 4 else len(circles) - 1
    # )
    suitable_circles = sorted(
        circles, key=lambda x: x["radius"], reverse=True
    )[:3]

    match pillar_method:
        case "balls":
            for c in suitable_circles:
                BALL_RADIUS = min(500, random.uniform(300, c["radius"] * 0.5))
                print(BALL_RADIUS)
                x, y, z = sample_point_in_disk(c)
                pillars.append(
                    [
                        {
                            "Location": (x, y, z + 300),
                            "Range": BALL_RADIUS,
                            "NoiseRange": 100,
                        },
                        {
                            "Location": (x, y + 250, z + 300),
                            "Range": BALL_RADIUS,
                            "NoiseRange": 100,
                        },
                    ]
                )
        case "straight":
            for c in suitable_circles:
                if c["radius"] < 400:
                    continue
                x, y, _ = sample_point_in_disk(c)
                pillars.append(
                    [
                        {
                            "Location": (
                                x,
                                y,
                                c["center"][2] - c["radius"] * 0.3,
                            ),
                            "Range": random.randrange(150, 200),
                            "NoiseRange": random.randrange(100, 150),
                        },
                        {
                            "Location": (
                                x,
                                y,
                                c["center"][2]
                                + c["radius"] * 1.3 * height_mult,
                            ),
                            "Range": random.randrange(150, 200),
                            "NoiseRange": random.randrange(100, 150),
                        },
                    ]
                )
        case "none":
            pass
        case "stalagmite":
            for c in suitable_circles:
                if c["radius"] < 400:
                    continue
                x, y, _ = sample_point_in_disk(c)
                heights = np.linspace(
                    c["radius"] * 1.2, c["radius"] * 0.4, num=6
                )
                thickness = np.linspace(200, 50, num=6)
                pillars.append(
                    [
                        {
                            "Location": (
                                x,
                                y,
                                c["center"][2] + h,
                            ),
                            "Range": t,
                            "NoiseRange": random.randrange(100),
                        }
                        for h, t in zip(heights, thickness)
                    ]
                )
        case "stalactite":
            for c in suitable_circles:
                if c["radius"] < 400:
                    continue
                x, y, _ = sample_point_in_disk(c)
                heights = np.linspace(
                    c["radius"] * -0.2, c["radius"] * 0.6, num=6
                )
                thickness = np.linspace(200, 50, num=6)
                pillars.append(
                    [
                        {
                            "Location": (
                                x,
                                y,
                                c["center"][2] + h,
                            ),
                            "Range": t,
                            "NoiseRange": random.randrange(100),
                        }
                        for h, t in zip(heights, thickness)
                    ]
                )
    return pillars


def build_json_multiple_ffill(
    traversal_list,
    circles,
    entrances,
    pillars,
    room_num,
    key,
    settings,
    pe_features,
) -> dict:
    result = {"FloodFillLines": {}}
    for jj, (ffill, traversal) in enumerate(zip(circles, traversal_list)):
        lines = {"Points": []}
        for ii in traversal:
            circle = ffill[ii]
            if circle.get("floor_depth"):
                lines["Points"].append(
                    {
                        "Location": {
                            "X": circle["center"][0],
                            "Y": circle["center"][1],
                            "Z": circle["center"][2],
                        },
                        "HRange": circle["radius"],
                        "VRange": circle.get("height", circle["radius"]),
                        "CeilingHeight": circle.get(
                            "height", circle["radius"]
                        ),
                        "FloorDepth": circle.get("floor_depth", 0),
                    }
                )
            elif circle.get("floor_angle"):
                lines["Points"].append(
                    {
                        "Location": {
                            "X": circle["center"][0],
                            "Y": circle["center"][1],
                            "Z": circle["center"][2],
                        },
                        "HRange": circle["radius"],
                        "VRange": circle.get("height", circle["radius"]),
                        "CeilingHeight": circle.get(
                            "height", circle["radius"]
                        ),
                        "FloorAngle": circle.get("floor_angle", 0),
                    }
                )
            else:
                lines["Points"].append(
                    {
                        "Location": {
                            "X": circle["center"][0],
                            "Y": circle["center"][1],
                            "Z": circle["center"][2],
                        },
                        "HRange": circle["radius"],
                        "VRange": circle.get("height", circle["radius"]),
                        "CeilingHeight": circle.get(
                            "height", circle["radius"]
                        ),
                    }
                )
        result["FloodFillLines"][f"FFill_{jj}"] = lines

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

    # if list(pillars):
    #     result["FloodFillPillars"] = {}
    #     for ii, p in enumerate(pillars):
    #         result["FloodFillPillars"].update(
    #             {
    #                 f"FloodFillPillar_{ii}": {
    #                     "Points": [
    #                         {
    #                             "Location": {
    #                                 "X": p["Location_1"][0],
    #                                 "Y": p["Location_1"][1],
    #                                 "Z": p["Location_1"][2],
    #                             },
    #                             "Range": {
    #                                 "Min": p["Range"],
    #                                 "Max": p["Range"],
    #                             },
    #                             "NoiseRange": {
    #                                 "Min": p["NoiseRange"],
    #                                 "Max": p["NoiseRange"],
    #                             },
    #                         },
    #                         {
    #                             "Location": {
    #                                 "X": p["Location_2"][0],
    #                                 "Y": p["Location_2"][1],
    #                                 "Z": p["Location_2"][2],
    #                             },
    #                             "Range": {
    #                                 "Min": p["Range"],
    #                                 "Max": p["Range"],
    #                             },
    #                             "NoiseRange": {
    #                                 "Min": p["NoiseRange"],
    #                                 "Max": p["NoiseRange"],
    #                             },
    #                         },
    #                     ]
    #                 }
    #             }
    #         )
    if list(pillars):
        result["FloodFillPillars"] = {}
        for ii, p in enumerate(pillars):
            result["FloodFillPillars"].update(
                {
                    f"FloodFillPillar_{ii}": {
                        "Points": [
                            {
                                "Location": {
                                    "X": z["Location"][0],
                                    "Y": z["Location"][1],
                                    "Z": z["Location"][2],
                                },
                                "Range": {
                                    "Min": z["Range"],
                                    "Max": z["Range"],
                                },
                                "NoiseRange": {
                                    "Min": z["NoiseRange"],
                                    "Max": z["NoiseRange"],
                                },
                            }
                            for z in p
                        ]
                    }
                }
            )

    if pe_features:
        result["PE_MiningHead"] = {}
        for ii, feature in enumerate(pe_features):
            result["PE_MiningHead"].update(
                {
                    f"MineHead_{ii}": {
                        "Location": {
                            "X": feature[0],
                            "Y": feature[1],
                            "Z": feature[2],
                        }
                    }
                }
            )

    result["Name"] = f"RMA_ICGEN_{key}{room_num}"
    result["Bounds"] = settings["bounds"]
    result["Tags"] = settings["tags"]

    return result


def build_json(
    traversal_list,
    circles,
    entrances,
    pillars,
    room_num,
    key,
    settings,
    pe_features,
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
                "CeilingHeight": circle["radius"],
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

    # if pillars:
    #     result["FloodFillPillars"] = {}
    #     for ii, p in enumerate(pillars):
    #         result["FloodFillPillars"].update(
    #             {
    #                 f"FloodFillPillar_{ii}": {
    #                     "Points": [
    #                         {
    #                             "Location": {
    #                                 "X": p["Location_1"][0],
    #                                 "Y": p["Location_1"][1],
    #                                 "Z": p["Location_1"][2],
    #                             },
    #                             "Range": {
    #                                 "Min": p["Range"],
    #                                 "Max": p["Range"],
    #                             },
    #                             "NoiseRange": {
    #                                 "Min": p["NoiseRange"],
    #                                 "Max": p["NoiseRange"],
    #                             },
    #                         },
    #                         {
    #                             "Location": {
    #                                 "X": p["Location_2"][0],
    #                                 "Y": p["Location_2"][1],
    #                                 "Z": p["Location_2"][2],
    #                             },
    #                             "Range": {
    #                                 "Min": p["Range"],
    #                                 "Max": p["Range"],
    #                             },
    #                             "NoiseRange": {
    #                                 "Min": p["NoiseRange"],
    #                                 "Max": p["NoiseRange"],
    #                             },
    #                         },
    #                     ]
    #                 }
    #             }
    #         )
    if list(pillars):
        result["FloodFillPillars"] = {}
        for ii, p in enumerate(pillars):
            result["FloodFillPillars"].update(
                {
                    f"FloodFillPillar_{ii}": {
                        "Points": [
                            {
                                "Location": {
                                    "X": z["Location"][0],
                                    "Y": z["Location"][1],
                                    "Z": z["Location"][2],
                                },
                                "Range": {
                                    "Min": z["Range"],
                                    "Max": z["Range"],
                                },
                                "NoiseRange": {
                                    "Min": z["NoiseRange"],
                                    "Max": z["NoiseRange"],
                                },
                            }
                            for z in p
                        ]
                    }
                }
            )

    if pe_features:
        result["PE_MiningHead"] = {}
        for ii, feature in enumerate(pe_features):
            result["PE_MiningHead"].update(
                {
                    f"MineHead_{ii}": {
                        "Location": {
                            "X": feature[0],
                            "Y": feature[1],
                            "Z": feature[2],
                        }
                    }
                }
            )

    result["Name"] = f"RMA_ICGEN_{key}{room_num}"
    result["Bounds"] = settings["bounds"]
    result["Tags"] = settings["tags"]

    return result
