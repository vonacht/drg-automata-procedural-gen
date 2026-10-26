import random
import numpy as np
import math
from pillars import generate_formation
from more_itertools import flatten


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
    return [
        (
            circle["center"][0],
            circle["center"][1],
            circle["center"][2] + 200,
        )
        for circle in circles
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


# def generate_entrances(circles, graph):

#     spheres_w_entrance = [0]
#     # First decide the entrances:
#     circles_w_one_neighbor = [
#         k for k in graph if len(graph[k]) == 1 and k != 0 and 0 not in graph[k]
#     ]
#     if not circles_w_one_neighbor:
#         # spheres_w_exit = [random.randrange(1, len(circles))]
#         coords_y = [c["center"][1] for c in circles]
#         spheres_w_exit = [np.argmax(coords_y), np.argmin(coords_y)]
#     else:
#         spheres_w_exit = circles_w_one_neighbor + [
#             x
#             for x in random.choices(
#                 range(len(circles)), k=random.randrange(1, 3)
#             )
#             if x not in circles_w_one_neighbor and x != 0 and 0 not in graph[x]
#         ]

#     entrances = []
#     for ii, t in [(spheres_w_entrance, "Entrance"), (spheres_w_exit, "Exit")]:
#         for sphere_idx in ii:
#             sphere_coords = (
#                 circles[sphere_idx]["center"][0],
#                 circles[sphere_idx]["center"][1],
#                 circles[sphere_idx]["radius"],
#             )
#             neighbors = graph[sphere_idx]
#             neighbor_coords = [
#                 (
#                     circles[circle]["center"][0],
#                     circles[circle]["center"][1],
#                     circles[circle]["radius"],
#                 )
#                 for circle in neighbors
#             ]
#             angle, _, _ = most_free_angle(sphere_coords, neighbor_coords)
#             angle = 0 if angle is None else angle
#             vector_angle = np.degrees(angle)
#             entrances.append(
#                 (
#                     circles[sphere_idx]["center"][0]
#                     + circles[sphere_idx]["radius"] * math.cos(angle),
#                     circles[sphere_idx]["center"][1]
#                     + circles[sphere_idx]["radius"] * math.sin(angle),
#                     circles[sphere_idx]["center"][2]
#                     + circles[sphere_idx].get("floor_depth", 0) * 0.8
#                     + circles[sphere_idx]["radius"] * 0.2,
#                     0,
#                     0,
#                     vector_angle,
#                     t,
#                 )
#             )

#     return entrances


def xy_center(spheres):
    """
    Returns the geometric center of a group of spheres
    ignoring the Z coordinate.

    Output: (x_center, y_center)
    """
    if not spheres:
        raise ValueError("Sphere group is empty")

    n = len(spheres)

    x = sum(s["center"][0] for s in spheres) / n
    y = sum(s["center"][1] for s in spheres) / n

    return (x, y)


def center_of_mass(disks):
    """
    disks: dict like {key: {"center": (x, y), "radius": r}, ...}
    returns: (x_com, y_com)
    """
    total_mass = 0.0
    x_sum = 0.0
    y_sum = 0.0

    for disk in disks:
        x, y, _ = disk["center"]
        r = disk["radius"]
        mass = r**2  # proportional to area (π cancels out)

        total_mass += mass
        x_sum += mass * x
        y_sum += mass * y

    if total_mass == 0:
        raise ValueError("Total mass is zero")

    return (x_sum / total_mass, y_sum / total_mass)


def ray_circle_intersections(cx, cy, dx, dy, disk):
    (x0, y0, _) = disk["center"]
    r = disk["radius"]

    ox = cx - x0
    oy = cy - y0

    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r

    disc = b * b - 4 * a * c
    if disc < 0:
        return []

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    return [t for t in (t1, t2) if t >= 0]


def last_disk_intersections(disks, center, angle_step):
    """
    Returns for each ray:
        (ray_angle, disk_key, intersection_point, angle_from_origin)
    """
    cx, cy = center
    results = []

    angle = 0.0
    while angle < 2 * math.pi:
        dx = math.cos(angle)
        dy = math.sin(angle)

        best_t = -1
        best_disk = None

        for ii, disk in enumerate(disks):
            ts = ray_circle_intersections(cx, cy, dx, dy, disk)
            if ts:
                t_far = max(ts)
                if t_far > best_t:
                    best_t = t_far
                    best_disk = ii

        if best_disk is not None:
            # intersection point
            ix = cx + best_t * dx
            iy = cy + best_t * dy

            # angle relative to (0,0)
            theta_origin = math.atan2(iy, ix)

            results.append(
                (
                    np.rad2deg(angle),
                    disks[best_disk],
                    (ix, iy, disks[best_disk]["center"][2]),
                    np.rad2deg(theta_origin),
                )
            )
        else:
            results.append((np.rad2deg(angle), None, None, None))

        angle += angle_step

    return results


def furthest_spheres(spheres, point, n=3):
    """
    Return the n spheres furthest from a 3D point.

    Each sphere should be a dict like:
        {"center": (x, y, z), "radius": r}

    Distance is measured from the point to the sphere's surface.
    """

    def distance_to_sphere(sphere):
        center = sphere["center"]
        radius = sphere["radius"]

        dx = point[0] - center[0]
        dy = point[1] - center[1]
        dz = point[2] - center[2]

        center_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return center_distance - radius

    return sorted(spheres, key=distance_to_sphere, reverse=True)[:n]


def normalize(v):
    x, y = v
    length = math.sqrt(x**2 + y**2)
    return np.array((x / length, y / length))


def angle_degrees(v):
    x, y = v
    return math.degrees(math.atan2(y, x))


def generate_exits_furthest(circles, entrances):

    room_center = xy_center(circles)
    exits = []
    for entrance in entrances:
        entrance_location = (entrance[0], entrance[1], entrance[2])
        candidates = furthest_spheres(circles, entrance_location, n=1)
        for candidate in candidates:
            C = np.array((candidate["center"][0], candidate["center"][1]))
            P = np.array(room_center)
            A = C - P
            B = C + candidate["radius"] * normalize(A)

            exit_point = (B[0], B[1])
            exits.append(
                (
                    exit_point[0],
                    exit_point[1],
                    candidate["center"][2]
                    + candidate["radius"] * 0.2
                    + candidate.get("floor_depth", 0) * 0.8,
                    0,
                    0,
                    angle_degrees(A),
                    "Exit",
                )
            )
    return exits


def generate_entrances(
    circles,
    entrance_interval=(200, 360),
    exit_interval=(0, 180),
    theta=30,
    return_type=None,
):

    entrances = []
    exits = []
    room_center = xy_center(circles)
    valid_entry_points = [
        e
        for e in last_disk_intersections(
            circles, room_center, np.deg2rad(theta)
        )
        if e[1] is not None
    ]
    for entrance in [
        e
        for e in valid_entry_points
        if entrance_interval[0] < e[0] < entrance_interval[1]
    ]:
        entrances.append(
            (
                entrance[2][0],
                entrance[2][1],
                entrance[1]["center"][2]
                + entrance[1]["radius"] * 0.2
                + entrance[1].get("floor_depth", 0) * 0.8,
                0,
                0,
                entrance[3],
                "Entrance",
            )
        )
    for exit in [
        e
        for e in valid_entry_points
        if exit_interval[0] < e[0] < exit_interval[1]
    ]:
        exits.append(
            (
                exit[2][0],
                exit[2][1],
                exit[1]["center"][2]
                + exit[1]["radius"] * 0.2
                + exit[1].get("floor_depth", 0) * 0.8,
                0,
                0,
                exit[3],
                "Exit",
            )
        )
    match return_type:
        case "Exits":
            return exits
        case "Entrances":
            return entrances
        case _:
            return entrances + exits


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


def is_point_inside_any_hemisphere(point, hemispheres):
    """Checks if a 3D point is inside any dome-up hemisphere."""
    x, y, z = point
    for h in hemispheres:
        cx, cy, cz = h["center"]
        r = h["radius"]
        # Point must be above the flat cap (z >= cz) and inside the spherical shell radius
        if (
            z >= cz
            and (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < (r * 0.95) ** 2
        ):
            return True
    return False


def get_floor_and_pillar_height(x, y, hemispheres, min_pillar_height=1.0):
    """
    Finds the floor elevation and calculates total pillar height up to the ceiling.
    Returns (z_floor, pillar_height) or None if point is invalid or ceiling is too low.
    """
    candidate_floors = []

    # 1. Identify all hemispheres that contain (x, y) horizontally
    for h in hemispheres:
        cx, cy, cz = h["center"]
        r = h["radius"]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2

        if (
            dist_sq < (r * 0.98) ** 2
        ):  # Slight inset to keep pillars off extreme edge boundaries
            # Flat floor level for dome-up orientation
            candidate_floors.append(cz)

    if not candidate_floors:
        return None

    # Cave floor at (x, y) is the lowest overlapping flat base
    z_floor = min(candidate_floors)

    # 2. Raycast upward to find highest ceiling point within overlapping domes
    ceiling_z_candidates = []
    for h in hemispheres:
        cx, cy, cz = h["center"]
        r = h["radius"]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2

        if dist_sq < r**2:
            dz = math.sqrt(r**2 - dist_sq)
            z_ceiling = cz + dz

            # Ceiling point must not be hollowed out inside a higher overlapping dome
            if not is_point_inside_any_hemisphere(
                (x, y, z_ceiling + 0.1), hemispheres
            ):
                ceiling_z_candidates.append(z_ceiling)

    if not ceiling_z_candidates:
        return None

    z_ceiling = max(ceiling_z_candidates)
    pillar_height = z_ceiling - z_floor

    # Filter out tight crawlspaces where pillars wouldn't fit
    if pillar_height < min_pillar_height:
        return None

    return z_floor, pillar_height


def sample_pillars(
    hemisphere_groups, min_dist=5.0, min_height=1.5, candidates_per_step=30
):
    """
    Samples Poisson-distributed pillars with floor position and total vertical height.

    :param hemisphere_groups: List of lists of dicts [{'center': (x,y,z), 'radius': r}]
    :param min_dist: Minimum horizontal spacing between pillars
    :param min_height: Minimum required ceiling clearance for a pillar to generate
    :return: List of dicts [{'position': (x, y, z_floor), 'height': height}]
    """
    hemispheres = [h for group in hemisphere_groups for h in group]
    if not hemispheres:
        return []

    # 1. Bounding box calculation
    min_x = min(h["center"][0] - h["radius"] for h in hemispheres)
    max_x = max(h["center"][0] + h["radius"] for h in hemispheres)
    min_y = min(h["center"][1] - h["radius"] for h in hemispheres)
    max_y = max(h["center"][1] + h["radius"] for h in hemispheres)

    # 2. Grid for spatial hashing
    cell_size = min_dist / math.sqrt(2)
    grid = {}

    def get_grid_cell(p):
        return (
            int((p[0] - min_x) / cell_size),
            int((p[1] - min_y) / cell_size),
        )

    # 3. Seed initial point
    pillars = []
    active_list = []

    for _ in range(500):
        rx = random.uniform(min_x, max_x)
        ry = random.uniform(min_y, max_y)
        res = get_floor_and_pillar_height(rx, ry, hemispheres, min_height)
        if res is not None:
            z_floor, height = res
            init_pillar = {"position": (rx, ry, z_floor), "height": height}
            pillars.append(init_pillar)
            active_list.append(init_pillar)
            grid[get_grid_cell((rx, ry))] = init_pillar
            break

    # 4. Poisson Disk Sampling
    while active_list:
        idx = random.randint(0, len(active_list) - 1)
        base = active_list[idx]
        bx, by, _ = base["position"]
        found_valid = False

        for _ in range(candidates_per_step):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(min_dist, 2 * min_dist)
            cx = bx + dist * math.cos(angle)
            cy = by + dist * math.sin(angle)

            res = get_floor_and_pillar_height(cx, cy, hemispheres, min_height)
            if res is None:
                continue

            z_floor, height = res
            cell = get_grid_cell((cx, cy))

            # Spacing constraint check
            is_valid = True
            cell_x, cell_y = cell
            for gx in range(cell_x - 2, cell_x + 3):
                for gy in range(cell_y - 2, cell_y + 3):
                    neighbor = grid.get((gx, gy))
                    if neighbor:
                        nx, ny, _ = neighbor["position"]
                        if (cx - nx) ** 2 + (cy - ny) ** 2 < min_dist**2:
                            is_valid = False
                            break
                if not is_valid:
                    break

            if is_valid:
                new_pillar = {"position": (cx, cy, z_floor), "height": height}
                pillars.append(new_pillar)
                active_list.append(new_pillar)
                grid[cell] = new_pillar
                found_valid = True
                break

        if not found_valid:
            active_list.pop(idx)

    return pillars


def select_pillars(spheres, n=3):
    if not spheres:
        return []

    selected_indices = [
        max(range(len(spheres)), key=lambda i: spheres[i]["radius"])
    ]

    while len(selected_indices) < min(n, len(spheres)):
        remaining_indices = [
            i for i in range(len(spheres)) if i not in selected_indices
        ]

        chosen = max(
            remaining_indices,
            key=lambda i: min(
                clearance(spheres[i], spheres[j]) for j in selected_indices
            ),
        )

        selected_indices.append(chosen)

    return [spheres[i] for i in selected_indices]


def clearance(a, b):
    center_distance = np.linalg.norm(
        np.asarray(a["center"]) - np.asarray(b["center"])
    )

    return center_distance - a["radius"] - b["radius"]


def distance_3d(p1, p2):
    """Return the Euclidean distance between two 3D points."""
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def generate_pillars(circles, method=None, height_mult=1):
    """
    Here we generate some random pillars, not too many, just for added decoration.
    """
    pillars = []
    # number_of_pillars = (
    #     random.randrange(5) if len(circles) > 4 else len(circles) - 1
    # )
    # suitable_circles = sorted(
    #     circles, key=lambda x: x["radius"], reverse=True
    # )[: math.ceil(len(circles) * 0.15)]
    suitable_circles = sample_pillars(circles, min_dist=1500)
    # suitable_circles = []

    for c in suitable_circles:
        pillar_method = (
            random.choice(
                [
                    # "stalactite",
                    # "stalagmite",
                    "straight",
                    # "balls",
                    "formation",
                ]
            )
            if method is None
            else method
        )
        match pillar_method:
            case "formation":
                formation = generate_formation(
                    origin=c["position"],
                    length=c["height"],
                    radius=random.randint(150, 250),
                    n_segments=random.randint(1, 5),
                )
                pillars.append(
                    [
                        {
                            "Location": p,
                            "Range": random.randrange(100, 250),
                            "NoiseRange": random.randrange(100, 250),
                        }
                        for p in formation
                    ]
                )
            case "balls":
                if c["radius"] < 400:
                    continue
                BALL_RADIUS = min(500, random.uniform(300, c["radius"] * 0.3))
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
            case "asymmetric_ball":
                if c["radius"] < 400:
                    continue
                BALL_RADIUS = min(500, random.uniform(300, c["radius"] * 0.3))
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
                        {
                            "Location": (
                                x,
                                y + 250,
                                z + 300 + random.randint(100, 300),
                            ),
                            "Range": BALL_RADIUS,
                            "NoiseRange": 100,
                        },
                    ]
                )
            case "straight":
                # if c["radius"] < 400:
                #     continue
                # x, y, _ = sample_point_in_disk(c)
                # pillars.append(
                #     [
                #         {
                #             "Location": (
                #                 x,
                #                 y,
                #                 c["center"][2] - c["radius"] * 0.3,
                #             ),
                #             "Range": random.randrange(150, 200),
                #             "NoiseRange": random.randrange(100, 150),
                #         },
                #         {
                #             "Location": (
                #                 x,
                #                 y,
                #                 c["center"][2]
                #                 + c["radius"] * 1.3 * height_mult,
                #             ),
                #             "Range": random.randrange(150, 200),
                #             "NoiseRange": random.randrange(100, 150),
                #         },
                #     ]
                # )
                x, y, z = c["position"]
                pillars.append(
                    [
                        {
                            "Location": (x, y, z),
                            "Range": random.randrange(150, 200),
                            "NoiseRange": random.randrange(100, 150),
                        },
                        {
                            "Location": (x, y, z + c["height"]),
                            "Range": random.randrange(150, 200),
                            "NoiseRange": random.randrange(100, 150),
                        },
                    ]
                )
            case "none":
                pass
            case "stalagmite":
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
    for ii, entrance in enumerate(entrances["entrances"]):
        result["Entrances"].update(
            {
                f"Entrance_{ii}": {
                    "Location": {
                        "X": entrance["position"][0],
                        "Y": entrance["position"][1],
                        "Z": entrance["position"][2],
                    },
                    "Type": "Entrance",
                    "Direction": {
                        "Roll": entrance["roll"],
                        "Pitch": entrance["pitch"],
                        "Yaw": entrance["yaw"],
                    },
                }
            }
        )
    for ii, entrance in enumerate(
        entrances["exits"], len(entrances["entrances"])
    ):
        result["Entrances"].update(
            {
                f"Entrance_{ii}": {
                    "Location": {
                        "X": entrance["position"][0],
                        "Y": entrance["position"][1],
                        "Z": entrance["position"][2],
                    },
                    "Type": "Exit",
                    "Direction": {
                        "Roll": entrance["roll"],
                        "Pitch": entrance["pitch"],
                        "Yaw": entrance["yaw"],
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
