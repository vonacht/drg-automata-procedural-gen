import random
import numpy as np
import json
import math
import logging
import argparse
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from itertools import product
from more_itertools import flatten

from automata import CellularAutomaton

from heightmap import generate_heightmap
from utils import (
    compute_room_scale_factor,
    scale_spheres,
    build_graph,
    greedy_dfs_min_backtrack,
    scale_spheres_multiple_ffill,
)
from json_builder import (
    generate_entrances,
    generate_exit_w_vector,
    generate_PE_features,
    build_json,
    build_json_multiple_ffill,
    generate_pillars,
)
from delaunay import generate_complete_road_traversal
from bezier import random_bezier_path, midpoint_displacement

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class RoomTooSmall(Exception):
    pass


def game_of_life_rule(state, neighbors):
    BIRTH_THRESHOLD = [5, 6, 7, 8]
    SURVIVAL_THRESHOLD = [4, 5, 6, 7, 8]
    alive_neighbors = sum(neighbors)
    if state == 0:
        return 1 if alive_neighbors in BIRTH_THRESHOLD else 0
    else:
        return 1 if alive_neighbors in SURVIVAL_THRESHOLD else 0


def run_automata(
    inverted=False,
    randomize_size=False,
    size=15,
    iterations=8,
    remove_pockets=True,
):
    M = (
        random.choices([15, size], k=1, weights=[0.5, 0.5])[0]
        if randomize_size
        else size
    )
    ca = CellularAutomaton(
        M=M,
        N=M,
        rule=game_of_life_rule,
        p_alive=0.45,  # 45% initially alive
        boundary="wrap",
    )

    # We run it for N iteraions:
    ca.run(iterations, remove_pockets=remove_pockets)
    if inverted:
        ca.invert()

    return ca.grid


def gradient_grid(
    tilemap,
    row_direction="increasing",
    col_direction="increasing",
    row_weight=1.0,
    col_weight=1.0,
):
    """
    Create an MxN grid where values increase or decrease
    depending on row (x) and column (y) positions.

    Parameters:
        M (int): number of rows
        N (int): number of columns
        row_direction (str): "increasing" or "decreasing"
        col_direction (str): "increasing" or "decreasing"
        row_weight (float): scaling factor for row contribution
        col_weight (float): scaling factor for column contribution

    Returns:
        numpy.ndarray: MxN array
    """
    M, N = tilemap.shape
    rows = np.arange(M, dtype=float).reshape(M, 1)
    cols = np.arange(N, dtype=float).reshape(1, N)

    if row_direction == "decreasing":
        rows = M - 1 - rows

    if col_direction == "decreasing":
        cols = N - 1 - cols

    grid = row_weight * rows + col_weight * cols

    return grid


def create_room(
    tilemap,
    height_scale=20,
    max_cube_vol=1500,
    scale_coeff=1,
    verticality=None,
    cliff=None,
    pe=False,
):

    match verticality:
        case "heightmap":
            # We get a heightmap with a randomized seed:
            heightmap = generate_heightmap(
                tilemap,
                height_scale=height_scale,
                seed=random.randrange(10000),
            )
        case "up":
            heightmap = gradient_grid(tilemap, row_weight=0.3, col_weight=0.3)
        case "down":
            heightmap = gradient_grid(
                tilemap,
                col_direction="decreasing",
                row_direction="decreasing",
                row_weight=0.3,
                col_weight=0.3,
            )
        case "flat":
            heightmap = np.zeros((tilemap.shape))
        case _:
            heightmap = np.zeros((tilemap.shape))
    # We skeletonize the tilemap to get the maximmaly inscribed circles (which will later be converted in FloodFillLines):
    skeleton = skeletonize(tilemap)
    distance_map = distance_transform_edt(tilemap)
    # Get skeleton pixels with radius:
    ys, xs = np.where(skeleton)
    skeleton_points = [
        (x, y, distance_map[y, x])
        for y, x in zip(ys, xs)
        if distance_map[y, x] > 0
    ]
    # Sort by radius descending:
    skeleton_points.sort(key=lambda c: c[2], reverse=True)
    # Filter overlapping circles
    selected_circles = []
    for x, y, r in skeleton_points:
        overlap = False
        for sx, sy, sr in selected_circles:
            # Distance between centers
            d = np.hypot(x - sx, y - sy)
            if d < 0.4 * (
                r + sr
            ):  # allow some small overlap. If the value is too big there will be gaps
                overlap = True
                break
        if not overlap:
            selected_circles.append((x, y, r))

    # We skip empty rooms and rooms with 1 circle (they are too simple even for the small rooms):
    if len(selected_circles) <= 1 and not pe:
        raise RoomTooSmall

    if verticality in ["up", "down"]:
        normalized_heightmap = (
            heightmap
            - heightmap[selected_circles[0][1], selected_circles[0][0]]
        )
    else:
        normalized_heightmap = heightmap

    # We attach the height value from the heightmap to each one of the final circles:
    if verticality in ["down"] and cliff:
        num_circles = len(selected_circles) // 2
        spheres = [
            {
                "center": (
                    c[0],
                    c[1],
                    normalized_heightmap[c[1], c[0]]
                    + cliff_factor(ii, num_circles, verticality),
                ),
                "radius": c[2],
            }
            for ii, c in enumerate(selected_circles)
        ]
    else:
        spheres = [
            {
                "center": (c[0], c[1], normalized_heightmap[c[1], c[0]]),
                "radius": c[2],
            }
            for c in selected_circles
        ]
    center_point = (0, 0, 0) if len(spheres) < 1 else spheres[0]["center"]
    # We scale the room based on the max_cube_vol value in the settings:
    scale_factor = compute_room_scale_factor(
        spheres, optional_size=max_cube_vol, scale_coeff=scale_coeff
    )
    scaled_spheres = scale_spheres(
        spheres, scale_factor, center_point=center_point
    )
    """
    if not pe:
        for sphere in scaled_spheres:
            if sphere["radius"] < 300:
                raise RoomTooSmall
    """
    scaled_spheres = [
        {"center": s["center"], "radius": 500} if s["radius"] <= 300 else s
        for s in scaled_spheres
    ]
    # We create an adjacency graph of all the circles:
    graph = build_graph(scaled_spheres)
    # And we get the traversal path to be able to create our FloodFillLines:
    traversal_path = greedy_dfs_min_backtrack(graph)

    return (
        traversal_path,
        scaled_spheres,
        graph,
        heightmap,
        tilemap,
        selected_circles,
    )


def cliff_factor(ii, num_circles, verticality):
    result = 0
    match verticality:
        case "up":
            result = 2 if ii > num_circles else 0
        case "down":
            result = -2 if ii > num_circles else 0
    return result


def upscale2x(arr):
    """
    Upscale a 2D numpy array by 2x using nearest-neighbor (pixel doubling).
    """
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


def create_DLA_tilemap(M, N, target_fill, stick_prob, max_steps=10000):
    """
    Perform Diffusion-Limited Aggregation (DLA) on an MxN grid.

    Parameters:
        M, N (int): Grid size
        target_fill (float): Fraction of grid to fill (0 < target_fill <= 1)
        max_steps (int): Max steps per particle to avoid infinite wandering

    Returns:
        grid (np.ndarray): MxN array with 1s as occupied cells
    """

    grid = np.zeros((M, N), dtype=np.int8)

    # Start with a seed in the center
    center = (M // 2, N // 2)
    grid[center] = 1

    occupied_count = 1
    target_count = int(M * N * target_fill)

    # Neighbor directions (4-connectivity)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_adjacent_to_cluster(x, y):
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < M and 0 <= ny < N:
                if grid[nx, ny] == 1:
                    return True
        return False

    while occupied_count < target_count:

        # Spawn walker at random boundary
        x, y = random.choice(
            [
                (0, random.randrange(N)),
                (M - 1, random.randrange(N)),
                (random.randrange(M), 0),
                (random.randrange(M), N - 1),
            ]
        )

        for _ in range(max_steps):
            # Random walk step
            dx, dy = random.choice(directions)
            x += dx
            y += dy

            # Keep inside bounds
            if x <= 0 or x >= M - 1 or y <= 0 or y >= N - 1:
                break  # discard walker if it escapes

            # Stick if adjacent to cluster
            if is_adjacent_to_cluster(x, y):
                if random.random() < stick_prob:
                    if grid[x, y] == 0:
                        grid[x, y] = 1
                        occupied_count += 1
                    break

    return upscale2x(grid)


def create_random_walk(M, N, steps, start=None):
    """
    Perform a random walk on an MxN grid.

    Parameters:
        M (int): number of rows
        N (int): number of columns
        steps (int): number of steps in the walk
        start (tuple): optional starting position (row, col)

    Returns:
        grid (numpy array): MxN grid with visited cells marked as 1
    """

    grid = np.zeros((M, N), dtype=int)

    # Starting position
    if start is None:
        row = random.randint(0, M - 1)
        col = random.randint(0, N - 1)
    else:
        row, col = start

    grid[row, col] = 1

    # Possible moves: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(steps):
        dr, dc = random.choice(moves)
        new_row = row + dr
        new_col = col + dc

        # Check boundaries
        if 0 <= new_row < M and 0 <= new_col < N:
            row, col = new_row, new_col
            grid[row, col] = 1

    return grid


def rotation_matrix_axis_angle(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis

    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c

    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rotate_spheres(spheres, center_point, rotation_matrix):
    cx, cy, cz = center_point
    rotated = []

    intra_array = [
        (c["center"][0], c["center"][1], c["center"][2], c["radius"])
        for c in spheres
    ]
    for x, y, z, r in intra_array:
        # Translate to origin
        point = np.array([x - cx, y - cy, z - cz])

        # Rotate
        rotated_point = rotation_matrix @ point

        # Translate back
        rotated_point += np.array([cx, cy, cz])

        rotated.append((*rotated_point, r))

    return [{"center": (s[0], s[1], s[2]), "radius": s[3]} for s in rotated]


def compute_side_tunnel(c1, c2, type):
    direction_vector = (np.array(c2["center"]) - np.array(c1["center"])) / 2
    mid_point = direction_vector + np.array(c1["center"])
    match type:
        case "-":
            perpendicular_vector = (
                -direction_vector[1],
                direction_vector[0],
                0,
            )
        case "+":
            perpendicular_vector = (
                direction_vector[1],
                -direction_vector[0],
                0,
            )
    return mid_point + perpendicular_vector


def sample_helix(radius, height, turns=1, n_points=100):
    """
    Sample points along a 3D helix.

    Args:
        radius (float): radius of the helix
        height (float): total height of the helix
        turns (float): number of full rotations
        n_points (int): number of sample points

    Returns:
        list of (x, y, z) tuples
    """
    points = []
    max_t = 2 * math.pi * turns

    for i in range(n_points):
        t = max_t * i / (n_points - 1)
        x = radius * math.cos(t)
        y = radius * math.sin(t)
        z = height * (t / max_t)
        points.append((x, y, z))

    return points


def run_pitrooms(mixed):
    tags = {"small": 2000, "medium": 3000, "big": 4000}
    for i in range(10):
        for k, v in tags.items():
            generate_pitroom(k, i, v, mixed)


def generate_pitroom(tag, room_num, size, mixed):

    direction = random.choice([-1])
    radius = random.randrange(500, 1200)
    height = size + random.uniform(-500, 500)
    turns = {"small": 1, "medium": 1, "big": 1}
    main_helix = sample_helix(radius * 1.3, direction * height, turns[tag], 40)
    main_helix = [{"center": h, "radius": 500} for h in main_helix]
    main_shaft = [
        {"center": (0, 0, 0), "radius": radius * 1.4},
        {"center": (0, 0, 0), "radius": radius},
        {"center": (0, 0, direction * height), "radius": radius},
        {"center": (0, 0, direction * height), "radius": radius * 1.4},
    ]
    new_room = [main_helix] + [main_shaft]
    traversal_paths = [list(range(len(main_helix)))] + [[0, 1, 2, 3]]
    first_point = main_shaft[0]
    second_point = main_shaft[-1]
    # Entrances:
    first_entrances = [
        (
            first_point["center"][0],
            first_point["center"][1] - first_point["radius"],
            first_point["center"][2] + 250,
            0,
            0,
            -90,
            "Entrance",
        )
    ]
    second_entrances = [
        (
            second_point["center"][0],
            second_point["center"][1] + second_point["radius"],
            second_point["center"][2] + 250,
            0,
            0,
            90,
            "Exit",
        )
    ]
    pillars = []
    pe = []
    name = f"RMA_ICGEN_{tag.upper()}_PITROOM{room_num}.json"
    room_name = f"{tag.upper()}_PITROOM"
    if not mixed:
        tags = {
            "small": ["Rooms.Linear.CustomSmall"],
            "medium": ["Rooms.Linear.CustomMedium"],
            "big": ["Rooms.Linear.CustomBig"],
        }
    else:
        tags = random.choice(
            [
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Flat"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Flat"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Flat"],
                },
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Vertical"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Vertical"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Vertical"],
                },
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Slope"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Slope"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Slope"],
                },
            ]
        )
    room_json = build_json_multiple_ffill(
        traversal_paths,
        new_room,
        first_entrances + second_entrances,
        pillars,
        room_num,
        room_name,
        {"bounds": size + 800, "tags": tags[tag]},
        pe,
    )
    try:
        path = Path("generated")
        path.mkdir(parents=True, exist_ok=True)
        with open(f"generated/{name}", "w") as f:
            json.dump(room_json, f, indent=4)
        logging.info(f"Written {name} to generated/{name}, method: pitroom")
    except Exception as e:
        logging.error(f"Couldn't write {name}: {e}")


def circular_arc_points_by_length(
    radius, start_angle, end_angle, step_length, center=(0, 0)
):
    """
    Generate points along a 2D circular arc with approximately uniform arc-length spacing.

    Parameters:
        radius (float): Radius of the circle
        start_angle (float): Start angle in radians
        end_angle (float): End angle in radians
        step_length (float): Desired distance between consecutive points along the arc
        center (tuple): (x, y) center of the circle

    Returns:
        np.ndarray: Array of shape (N, 2) with (x, y) points
    """
    arc_length = abs(end_angle - start_angle) * radius

    # Number of segments (at least 1 to avoid division issues)
    num_segments = max(1, int(np.ceil(arc_length / step_length)))

    # Recompute exact angle step so spacing is uniform
    angles = np.linspace(start_angle, end_angle, num_segments + 1)

    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    return np.column_stack((x, y))


def centroid_of_disk_centers(disks):
    """
    Compute the centroid (average position) of disk centers.

    Parameters:
        disks: iterable of (x, y, r) — radius is ignored

    Returns:
        (cx, cy): centroid of the centers
    """
    if not disks:
        raise ValueError("Empty disk list.")

    sx = 0.0
    sy = 0.0

    for disk in disks:
        x, y, _ = disk["center"]
        sx += x
        sy += y

    n = len(disks)
    return sx / n, sy / n


def closest_point_3d(target, points):
    """
    Find the closest point to 'target' in a list of 3D points.

    Parameters:
        target: tuple (x, y, z)
        points: iterable of (x, y, z)

    Returns:
        closest point (x, y, z)
    """
    if not points:
        raise ValueError("Point list is empty.")

    tx, ty, tz = target["center"]

    closest = None
    min_dist_sq = float("inf")

    for p in points:
        x, y, z = p["center"]
        dx = x - tx
        dy = y - ty
        dz = z - tz
        dist_sq = dx * dx + dy * dy + dz * dz  # squared distance

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest = (x, y, z)

    return closest


def quartile_elements(seq):
    n = len(seq)
    indices = [0, n // 3, (2 * n) // 3]
    return [seq[i] for i in indices]


def shift_room(room, point):
    shift_vector = np.array(point) - np.array(room[0]["center"])
    return [
        {
            "center": (np.array(c["center"] + shift_vector)),
            "radius": c["radius"],
        }
        for c in room
    ]


def circle_points(n, r, c=(0, 0), h=None):
    """
    Generate n points on a circle of radius r centered at c.

    Args:
        n (int): Number of points
        r (float): Radius of the circle
        c (tuple): Center of the circle (cx, cy)

    Returns:
        list of tuples: [(x1, y1), (x2, y2), ...]
    """
    cx, cy = c
    points = []

    for i in range(n):
        theta = 2 * math.pi * i / n
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        points.append((x, y))

    if h is None:
        return points
    else:
        return [(j[0], j[1], h) for j in points]


def generate_torus_room(name, size, room_num):

    def make_one_room():
        method = random.choice(["random_walk", "automaton"])
        # print(method)
        method = "automaton"
        # automaton_size = random.choice([15, 22, 30])
        automaton_size = 15
        while True:
            if method == "automaton":
                tilemap = run_automata(
                    inverted=False,
                    randomize_size=False,
                    remove_pockets=True,
                    size=automaton_size,
                )
            else:
                tilemap = create_random_walk(50, 50, 100)
            try:
                (
                    traversal_path,
                    scaled_circles,
                    graph,
                    heightmap,
                    ca_grid,
                    selected_circles,
                ) = create_room(
                    tilemap,
                    height_scale=10,
                    max_cube_vol=size,
                    scale_coeff=1,
                    verticality=random.choice(
                        ["flat", "up", "down", "heightmap"]
                    ),
                    cliff=False,
                )
                return (
                    traversal_path,
                    scaled_circles,
                    graph,
                    heightmap,
                    ca_grid,
                    selected_circles,
                )
            except RoomTooSmall:
                continue

    a1, a2, a3, *_ = make_one_room()
    b1, b2, b3, *_ = make_one_room()
    c1, c2, c3, *_ = make_one_room()
    d1, d2, d3, *_ = make_one_room()

    main_torus = circle_points(20, 4000)
    main_torus.append(main_torus[0])
    # Shift the generated rooms to the points across the torus:
    shifted_rooms = [
        shift_room(room, (torus_point[0], torus_point[1], 0))
        for torus_point, room in zip(
            quartile_elements(main_torus), [a2, b2, c2]
        )
    ]

    shifted_rooms.append(shift_room(d2, (0, 0, 0)))

    main_torus = [
        {"center": (c[0], c[1], 0), "radius": 500} for c in main_torus
    ]
    drop_pod_tunnel = [
        {"center": (0, 0, 0), "radius": 500},
        {"center": random.choice(main_torus)["center"], "radius": 500},
    ]
    # Append room and traversal paths:
    new_room = shifted_rooms + [main_torus] + [drop_pod_tunnel]
    traversal_paths = (
        [a1] + [b1] + [c1] + [d1] + [list(range(len(main_torus)))] + [[0, 1]]
    )
    # Entrances:
    first_entrances = [
        b
        for b in generate_entrances(shifted_rooms[1], b3)
        if b[-1] == "Entrance"
    ]
    second_entrances = [
        b for b in generate_entrances(shifted_rooms[2], c3) if b[-1] == "Exit"
    ]
    pillars = []
    pe = flatten([generate_PE_features(c) for c in shifted_rooms])
    name = f"RMA_ICGEN_PE{room_num}.json"
    room_name = f"PE"
    # Create the room:
    room_json = build_json_multiple_ffill(
        traversal_paths,
        new_room,
        first_entrances + second_entrances,
        pillars,
        room_num,
        room_name,
        {"bounds": size + 800, "tags": ["Rooms.Small.CustomPE"]},
        pe,
    )

    try:
        path = Path("generated")
        path.mkdir(parents=True, exist_ok=True)
        with open(f"generated/{name}", "w") as f:
            json.dump(room_json, f, indent=4)
        logging.info(f"Written {name} to generated/{name}, method: torus")
    except Exception as e:
        logging.error(f"Couldn't write {name}: {e}")


def get_quarter_positions(arr):
    if not arr:
        return None  # or raise an error if preferred

    n = len(arr)

    first = arr[0]
    quarter = arr[n // 4]
    middle = arr[n // 2]
    three_quarter = arr[(3 * n) // 4]

    return first, quarter, middle, three_quarter


def generate_maze_with_loops(n, m, loop_factor=0.1):
    # Ensure odd dimensions
    n = n if n % 2 == 1 else n + 1
    m = m if m % 2 == 1 else m + 1

    maze = [[1 for _ in range(m)] for _ in range(n)]

    def carve(x, y):
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < n and 0 < ny < m and maze[nx][ny] == 1:
                maze[nx][ny] = 0
                maze[x + dx // 2][y + dy // 2] = 0
                carve(nx, ny)

    # Step 1: generate perfect maze
    maze[1][1] = 0
    carve(1, 1)

    # Step 2: add loops by removing random walls
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            if maze[i][j] == 1:
                # Only consider walls between two paths
                neighbors = 0
                if maze[i + 1][j] == 0:
                    neighbors += 1
                if maze[i - 1][j] == 0:
                    neighbors += 1
                if maze[i][j + 1] == 0:
                    neighbors += 1
                if maze[i][j - 1] == 0:
                    neighbors += 1

                # If it's separating paths, maybe remove it
                if neighbors >= 2 and random.random() < loop_factor:
                    maze[i][j] = 0

    return maze


def traverse_maze(maze, start=(1, 1)):
    n, m = len(maze), len(maze[0])
    visited = [[False] * m for _ in range(n)]
    path = []

    def dfs(x, y):
        visited[x][y] = True
        path.append((x, y))

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m:
                if maze[nx][ny] == 0 and not visited[nx][ny]:
                    dfs(nx, ny)

    dfs(*start)
    return path


def traverse_maze_1(maze, start=(1, 1)):
    n = len(maze)
    visited = [[False] * n for _ in range(n)]
    path = []

    def dfs(r, c):
        # bounds + visited check
        if r < 0 or r >= n or c < 0 or c >= n or visited[r][c]:
            return
        if maze[r][c] == 1:
            return

        visited[r][c] = True
        path.append((r, c))

        # explore neighbors (up, right, down, left)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                dfs(nr, nc)
                # append current cell again to keep path continuous when backtracking
                path.append((r, c))

    dfs(start[0], start[1])
    return path


def run_doublerooms(mixed):
    tags = {"small": 2000, "medium": 2500, "big": 3000}
    # for i in range(25):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "dive")
    # for i in range(60):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "overlap")
    # for i in range(25):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "side")
    # for i in range(10):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "ring")
    # for i in range(20):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "upwards_pit")
    # for i in range(20):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "concave")
    # # for i in range(20):
    # #     for k, v in tags.items():
    # #         generate_doubleroom(k, i, v, mixed, "slope")
    # for i in range(20):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "maze")
    # for i in range(20):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "delaunay")
    # for i in range(20):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "patterned")
    # for i in range(20):
    #     for k, v in tags.items():
    # #         generate_doubleroom(k, i, v, mixed, "ramp")
    # for i in range(20):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "trench")
    for i in range(20):
        for k, v in tags.items():
            generate_doubleroom(k, i, v, mixed, "tunnel")
    # for i in range(20):
    #     for k, v in tags.items():
    #         generate_doubleroom(k, i, v, mixed, "doublefloor")


def generate_doubleroom(tag, room_num, size, mixed, room_type):

    def make_one_room():
        method = random.choice(["random_walk", "automaton"])
        # print(method)
        method = "automaton"
        verticality = (
            "flat"
            if room_type == "ramp"
            else random.choice(["up", "down", "flat", "heightmap"])
        )
        # automaton_size = random.choice([15, 22, 30])
        automaton_size = 15
        while True:
            if method == "automaton":
                tilemap = run_automata(
                    inverted=False,
                    randomize_size=False,
                    remove_pockets=True,
                    size=automaton_size,
                )
            else:
                tilemap = create_random_walk(50, 50, 100)
            try:
                (
                    traversal_path,
                    scaled_circles,
                    graph,
                    heightmap,
                    ca_grid,
                    selected_circles,
                ) = create_room(
                    tilemap,
                    height_scale=10,
                    max_cube_vol=size,
                    scale_coeff=1,
                    verticality=verticality,
                    cliff=False,
                )
                return (
                    traversal_path,
                    scaled_circles,
                    graph,
                    heightmap,
                    ca_grid,
                    selected_circles,
                )
            except RoomTooSmall:
                continue

    a1, a2, a3, *_ = make_one_room()
    b1, b2, b3, *_ = make_one_room()

    match room_type:
        case "dive":
            # Decide if angled or straight down:
            angle = random.choice(["vertical", "angle"])
            match angle:
                case "vertical":
                    xshift = 0
                    yshift = 0
                case "angle":
                    xshift = random.randrange(1000, 1500)
                    yshift = random.randrange(1000, 1500)
            # Get coordinates of last point of first room:
            pit_center = random.choice([0, -1])
            # Modify the coordinates of the second room:
            shift_vector = np.array(a2[pit_center]["center"]) - np.array(
                b2[0]["center"]
            )
            b2 = [
                {
                    "center": (
                        c["center"][0] + shift_vector[0] + xshift,
                        c["center"][1] + shift_vector[1] + yshift,
                        c["center"][2] - random.randrange(1500, 2000),
                    ),
                    "radius": c["radius"],
                }
                for c in b2
            ]
            shaft = [
                {
                    "center": a2[pit_center]["center"],
                    "radius": a2[pit_center]["radius"] * 0.7,
                },
                {"center": b2[0]["center"], "radius": b2[0]["radius"] * 0.7},
            ]
            # Append room and traversal paths:
            new_room = [a2] + [b2] + [shaft]
            traversal_paths = [a1] + [b1] + [[0, 1]]
            # Entrances:
            first_entrances = [
                b for b in generate_entrances(a2, a3) if b[-1] == "Entrance"
            ]
            second_entrances = [
                b for b in generate_entrances(b2, b3) if b[-1] == "Exit"
            ]
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_DOUBLEROOM{room_num}.json"
            room_name = f"{tag.upper()}_DOUBLEROOM"

        case "overlap":
            direction_vector = np.array(a2[-1]["center"]) - np.array(
                a2[0]["center"]
            )
            perpendicular_vector = perpendicular_2d_normalized(
                direction_vector
            )
            depth_method = random.choices(
                ["depth", "shift"], k=1, weights=[0.3, 0.7]
            )[0]
            print(depth_method)
            if depth_method == "depth":
                a_copy = [
                    {
                        "center": (
                            c["center"][0]
                            + perpendicular_vector[0]
                            + a2[0]["radius"] * random.uniform(0.7, 1.2),
                            c["center"][1]
                            - perpendicular_vector[1]
                            + a2[0]["radius"] * random.uniform(0.7, 1.2),
                            c["center"][2],
                        ),
                        "radius": c["radius"],
                        "floor_depth": -random.randrange(500, 1500),
                    }
                    for c in a2
                ]
                new_room = [a2] + [a_copy]
                traversal_paths = [a1] + [a1]
                # Entrances:
                first_entrances = [
                    b
                    for b in generate_entrances(a2, a3)
                    if b[-1] == "Entrance"
                ]
                second_entrances = [
                    b
                    for b in generate_entrances(a_copy, a3)
                    if b[-1] == "Exit"
                ]
            else:
                axis = (0, 0, 0)
                while axis == (0, 0, 0):
                    axis = (
                        random.choice([0, 1]),
                        random.choice([0, 1]),
                        random.choice([0, 1]),
                    )
                theta = np.radians(random.uniform(50, 120))
                R = rotation_matrix_axis_angle(axis, theta)
                center_of_rotation = direction_vector / 2
                a_copy = rotate_spheres(b2, center_of_rotation, R)
                new_room = [a2] + [a_copy]
                traversal_paths = [a1] + [b1]
                # Entrances:
                first_entrances = [
                    b
                    for b in generate_entrances(a2, a3)
                    if b[-1] == "Entrance"
                ]
                second_entrances = [
                    b
                    for b in generate_entrances(a_copy, b3)
                    if b[-1] == "Exit"
                ]
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_OVERLAPROOM{room_num}.json"
            room_name = f"{tag.upper()}_OVERLAPROOM"
        case "side":
            direction = random.choice(["-y", "y", "x"])
            direction = "x"
            match direction:
                case "-y":
                    shift = min(
                        [c["center"][1] - c["radius"] for c in a2]
                    ) - random.randrange(1500, 2000)
                case "y":
                    shift = max(
                        [c["center"][1] + c["radius"] for c in a2]
                    ) + random.randrange(1500, 2000)
                case "x":
                    shift = max(
                        [c["center"][0] + c["radius"] for c in a2]
                    ) + random.randrange(1500, 2000)
            b2 = [
                {
                    "center": (
                        c["center"][0] + shift,
                        c["center"][1] + shift,
                        c["center"][2] + random.randrange(-1200, 1200),
                    ),
                    "radius": c["radius"],
                }
                for c in b2
            ]
            match direction:
                case "-y" | "y":
                    min_first_circle = min(
                        a2, key=lambda x: x["center"][1] - x["radius"]
                    )
                    max_first_circle = max(
                        a2, key=lambda x: x["center"][1] + x["radius"]
                    )
                    min_second_circle = min(
                        b2, key=lambda x: x["center"][1] - x["radius"]
                    )
                    max_second_circle = max(
                        b2, key=lambda x: x["center"][1] + x["radius"]
                    )
                case "x":
                    min_first_circle = min(
                        a2, key=lambda x: x["center"][0] - x["radius"]
                    )
                    max_first_circle = max(
                        a2, key=lambda x: x["center"][0] + x["radius"]
                    )
                    min_second_circle = min(
                        b2, key=lambda x: x["center"][0] - x["radius"]
                    )
                    max_second_circle = max(
                        b2, key=lambda x: x["center"][0] + x["radius"]
                    )
            tunnel_middle_room_c1 = compute_side_tunnel(
                min_first_circle, min_second_circle, "-"
            )
            tunnel_middle_room_c2 = compute_side_tunnel(
                max_first_circle, max_second_circle, "+"
            )
            c1 = [
                min_first_circle,
                {"center": tunnel_middle_room_c1, "radius": 450},
                min_second_circle,
            ]
            c1 = sample_quadratic_bezier(
                *[c["center"] for c in c1], num_points=10
            )
            c1 = [{"center": p, "radius": 450} for p in c1]
            c2 = [
                max_first_circle,
                {"center": tunnel_middle_room_c2, "radius": 450},
                max_second_circle,
            ]
            c2 = sample_quadratic_bezier(
                *[c["center"] for c in c2], num_points=10
            )
            c2 = [{"center": p, "radius": 450} for p in c2]
            new_room = [a2] + [b2] + [c1] + [c2]
            traversal_paths = (
                [a1] + [b1] + [list(range(len(c1)))] + [list(range(len(c2)))]
            )
            # Entrances:
            # first_entrances = [b for b in generate_entrances(a2, a3) if b[-1] == "Entrance"]
            # second_entrances = [b for b in generate_entrances(b2, b3) if b[-1] == "Exit"]
            first_point = min(a2, key=lambda x: x["center"][1] - x["radius"])
            second_point = max(b2, key=lambda x: x["center"][1] + x["radius"])
            first_entrances = [
                (
                    first_point["center"][0],
                    first_point["center"][1] - first_point["radius"],
                    first_point["center"][2],
                    0,
                    0,
                    -90,
                    "Entrance",
                )
            ]
            second_entrances = [
                (
                    second_point["center"][0],
                    second_point["center"][1] + second_point["radius"],
                    second_point["center"][2],
                    0,
                    0,
                    90,
                    "Exit",
                )
            ]
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_SIDEROOM{room_num}.json"
            room_name = f"{tag.upper()}_SIDEROOM"
        case "ring":
            max_x = max([c["center"][0] + c["radius"] for c in a2])
            min_x = min([c["center"][0] - c["radius"] for c in a2])
            max_y = max([c["center"][1] + c["radius"] for c in a2])
            min_y = min([c["center"][1] - c["radius"] for c in a2])
            room_span = max(max_x - min_x, max_y - min_y)
            room_type = random.choice(["one", "two"])
            # The first ring:
            start_angle = np.deg2rad(random.uniform(0, 360))
            end_angle = np.deg2rad(random.uniform(40, 169)) + start_angle
            center = centroid_of_disk_centers(a2)
            ring_points = circular_arc_points_by_length(
                room_span * 0.9, start_angle, end_angle, 1000, center=center
            )
            ring_points = [
                {"center": (d[0], d[1], 0), "radius": 450} for d in ring_points
            ]
            # Tunnels:
            tunnels = [
                [r, {"center": closest_point_3d(r, a2), "radius": 450}]
                for r in ring_points[::2]
            ]
            new_room = [a2] + [ring_points] + tunnels
            traversal_paths = (
                [a1]
                + [list(range(len(ring_points)))]
                + [[0, 1]] * len(tunnels)
            )
            # Entrances:
            # first_entrances = [b for b in generate_entrances(a2, a3) if b[-1] == "Entrance"]
            # second_entrances = [b for b in generate_entrances(b2, b3) if b[-1] == "Exit"]
            second_point = min(a2, key=lambda x: x["center"][1] - x["radius"])
            first_point = max(
                ring_points, key=lambda x: x["center"][1] + x["radius"]
            )
            first_entrances = [
                (
                    math.cos((start_angle + end_angle) / 2) * room_span * 1.05
                    + center[0],
                    math.sin((start_angle + end_angle) / 2) * room_span * 1.05
                    + center[1],
                    first_point["center"][2] + 150,
                    0,
                    0,
                    np.rad2deg((start_angle + end_angle) / 2),
                    "Entrance",
                )
            ]
            point = (
                math.cos((start_angle + end_angle) / 2) * room_span * 1.05
                + center[0],
                math.sin((start_angle + end_angle) / 2) * room_span * 1.05
                + center[1],
            )
            second_entrances = generate_exit_w_vector(
                a2, point, np.array(center) - np.array(point)
            )
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_RINGROOM{room_num}.json"
            room_name = f"{tag.upper()}_RINGROOM"
        case "upwards_pit":
            radius = random.randrange(800, 850)
            height = 2000 + random.uniform(-1000, 1000)
            main_helix = sample_helix(radius * 1.3, height, 1, 40)
            main_helix = [{"center": h, "radius": 500} for h in main_helix]
            main_shaft = [
                {"center": (0, 0, 0), "radius": radius * 1.4},
                {"center": (0, 0, 0), "radius": radius},
                {"center": (0, 0, height), "radius": radius},
                {"center": (0, 0, height), "radius": radius * 1.4},
            ]
            direction_vector = np.array(main_helix[-1]["center"])
            second_room = shift_room(
                a2,
                direction_vector + np.array([1200, 1200, 0]),
            )
            secondary_shaft = [
                {"center": main_shaft[-1]["center"], "radius": 450},
                {"center": second_room[0]["center"], "radius": 450},
            ]
            new_room = (
                [main_helix] + [main_shaft] + [second_room] + [secondary_shaft]
            )
            traversal_paths = (
                [list(range(len(main_helix)))]
                + [[0, 1, 2, 3]]
                + [a3]
                + [[0, 1]]
            )
            first_point = main_shaft[0]
            second_point = second_room[-1]
            # Entrances:
            first_entrances = [
                (
                    first_point["center"][0],
                    first_point["center"][1] - first_point["radius"],
                    first_point["center"][2] + 250,
                    0,
                    0,
                    -90,
                    "Entrance",
                )
            ]
            second_entrances = [
                e
                for e in generate_entrances(second_room, a3)
                if e[-1] != "Entrance"
            ]
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_UPWARDSPIT{room_num}.json"
            room_name = f"{tag.upper()}_UPWARDSPIT"
        case "concave":
            pit_room = random.choices([False, True], weights=[0.5, 0.5], k=1)[
                0
            ]
            outer_radius = random.randrange(1500, 3000)
            rings = [
                circle_points(10, outer_radius - 200 * i, h=-100 * i)
                for i in range(12)
            ]
            rings = [
                [{"center": (c[0], c[1], c[2]), "radius": 600} for c in p]
                for p in rings
            ]
            rings = [p + [p[0]] for p in rings]
            if pit_room:
                second_room = shift_room(a2, (0, 0, -1800))
            else:
                second_room = shift_room(a2, rings[0][4]["center"])
            new_room = rings + [second_room]
            traversal_paths = [list(range(11))] * len(rings) + [a1]
            first_point = rings[0][0]
            second_point = min(
                second_room, key=lambda x: x["center"][1] + x["center"][0]
            )
            # Entrances:
            first_entrances = [
                (
                    first_point["center"][0] + first_point["radius"],
                    first_point["center"][1],
                    first_point["center"][2] + 250,
                    0,
                    0,
                    -0,
                    "Entrance",
                )
            ]
            if pit_room:
                second_entrances = [
                    e
                    for e in generate_entrances(second_room, a3)
                    if e[-1] != "Entrance"
                ]
            else:
                second_entrances = [
                    (
                        second_point["center"][0] - second_point["radius"],
                        second_point["center"][1],
                        second_point["center"][2] + 250,
                        0,
                        0,
                        180,
                        "Exit",
                    )
                ]
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_CONCAVEROOM{room_num}.json"
            room_name = f"{tag.upper()}_CONCAVEROOM"
        case "slope":
            angle = random.uniform(25, 45)
            height = random.uniform(3000, 4000)
            direction_vector = np.array(
                [0, math.sin(np.deg2rad(angle + 180)) * height, -height]
            )
            main_slope = [
                {"center": (0, 0, 0), "radius": random.uniform(600, 800)},
                {
                    "center": (direction_vector),
                    "radius": random.uniform(600, 800),
                },
            ]
            main_room = shift_room(a2, direction_vector / 2)
            new_room = [main_slope] + [main_room]
            traversal_paths = [[0, 1]] + [a3]
            first_point = main_slope[0]
            second_point = main_slope[-1]
            # Entrances:
            first_entrances = [
                (
                    first_point["center"][0],
                    first_point["center"][1] + first_point["radius"],
                    first_point["center"][2] + 250,
                    0,
                    0,
                    90,
                    "Entrance",
                )
            ]
            second_entrances = [
                (
                    second_point["center"][0] + second_point["radius"],
                    second_point["center"][1],
                    second_point["center"][2] + 250,
                    0,
                    0,
                    -0,
                    "Exit",
                )
            ] + [
                e
                for e in generate_entrances(main_room, a3)
                if e[-1] != "Entrance"
            ]
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_SLOPEROOM{room_num}.json"
            room_name = f"{tag.upper()}_SLOPEROOM"
        case "maze" | "delaunay":
            # maze_type = random.choice(["normal", "delaunay"])
            maze_type = "normal" if room_type == "maze" else "delaunay"
            match maze_type:
                case "normal":
                    match tag:
                        case "small":
                            maze = generate_maze_with_loops(
                                4, 4, loop_factor=0.15
                            )
                        case "medium":
                            maze = generate_maze_with_loops(
                                5, 5, loop_factor=0.15
                            )
                        case _:
                            maze = generate_maze_with_loops(
                                6, 6, loop_factor=0.15
                            )
                    height = random.choices([1, 1.5], weights=[0.8, 0.2], k=1)[
                        0
                    ]
                    room = []
                    for ii in range(len(maze)):
                        for jj in range(len(maze[0])):
                            if maze[ii][jj] == 0:
                                line = []
                                line.append(
                                    {"center": (ii, jj, 0), "radius": 0.6}
                                )
                                line.append(
                                    {"center": (ii, jj, height), "radius": 0.6}
                                )
                                room.append(line)
                    room = scale_spheres_multiple_ffill(room, 1000)
                    new_room = room
                    traversal_paths = [[0, 1]] * len(room)
                    second_point = room[0][0]
                    first_point = room[-1][0]
                    # Entrances:
                    first_entrances = [
                        (
                            first_point["center"][0],
                            first_point["center"][1] + second_point["radius"],
                            first_point["center"][2] + 250,
                            0,
                            0,
                            90,
                            "Entrance",
                        )
                    ]
                    second_entrances = [
                        (
                            second_point["center"][0],
                            second_point["center"][1] - second_point["radius"],
                            second_point["center"][2] + 250,
                            0,
                            0,
                            -90,
                            "Exit",
                        )
                    ]
                    name = f"RMA_ICGEN_{tag.upper()}_MAZEROOM{room_num}.json"
                    room_name = f"{tag.upper()}_MAZEROOM"
                case "delaunay":
                    tower_stack = 4
                    points, _, path = generate_complete_road_traversal(
                        6, 0.5, width=2, height=2
                    )
                    room = []
                    delaunay_points = [
                        {"center": (m[0], m[1], 0), "radius": 0.13}
                        for m in points
                    ]
                    delaunay_points = shift_room(delaunay_points, (0, 0, 0))
                    for ii in range(1, tower_stack + 1):
                        delaunay_w_z_offset = [
                            {
                                "center": (
                                    m["center"][0],
                                    m["center"][1],
                                    ii * 0.08,
                                ),
                                "radius": 0.1,
                            }
                            for m in delaunay_points
                        ]
                        room.append(delaunay_w_z_offset)
                    room.append(delaunay_points)
                    room = scale_spheres_multiple_ffill(room, 3200)
                    new_room = room
                    traversal_paths = [path] * tower_stack
                    first_point = max(
                        room[0], key=lambda x: x["center"][0] + x["center"][1]
                    )
                    second_point = min(
                        room[0], key=lambda x: x["center"][0] + x["center"][1]
                    )
                    first_entrances = [
                        (
                            first_point["center"][0],
                            first_point["center"][1] + second_point["radius"],
                            first_point["center"][2] + 250,
                            0,
                            0,
                            90,
                            "Entrance",
                        )
                    ]
                    second_entrances = [
                        (
                            second_point["center"][0],
                            second_point["center"][1] - second_point["radius"],
                            second_point["center"][2] + 250,
                            0,
                            0,
                            -90,
                            "Exit",
                        )
                    ]
                    name = (
                        f"RMA_ICGEN_{tag.upper()}_DELAUNAYROOM{room_num}.json"
                    )
                    room_name = f"{tag.upper()}_DELAUNAYROOM"
            pillars = []
            pe = []
        case "patterned":
            room_method = random.choice(["pit", "wall"])
            shaft_type = random.choice(["stripe", "cross"])
            shaft_type = "stripe"
            match shaft_type:
                case "stripe":
                    number_of_walls = random.randrange(1, 4)
                    number_of_occupied_walls = random.randrange(
                        1, number_of_walls + 1
                    )
                    occupied_walls = random.sample(
                        list(range(1, number_of_walls + 1)),
                        k=number_of_occupied_walls,
                    )
                    room_length = 3000
                    initial_shape = [
                        {"center": (0, 0, 0), "radius": 2000},
                        {"center": (0, room_length, 0), "radius": 2000},
                    ]
                    pillars = []
                    match room_method:
                        case "wall":
                            for wall in occupied_walls:
                                points, _, path = (
                                    generate_complete_road_traversal(
                                        40, 800, 5000, 5000
                                    )
                                )
                                pillars.append(
                                    [
                                        {
                                            "Location": (
                                                points[p][0] - 2600,
                                                (
                                                    (2 * room_length)
                                                    / (number_of_walls + 1)
                                                    * wall
                                                )
                                                - room_length / 2,
                                                points[p][1] - 1000,
                                            ),
                                            "Range": 95,
                                            "NoiseRange": 10,
                                        }
                                        for p in path
                                    ]
                                )
                            room = [initial_shape]
                            new_room = room
                            traversal_paths = [[0, 1]]
                        case "pit":
                            initial_shape = [
                                {"center": (0, 0, 0), "radius": 1200},
                                {
                                    "center": (0, room_length, 0),
                                    "radius": 1200,
                                },
                            ]
                            pit = [
                                {"center": (0, 1500, 0), "radius": 1500},
                                {"center": (0, 1500, -1500), "radius": 1500},
                            ]
                            ramp_A_points = sample_quadratic_bezier(
                                (-250, -900),
                                (-1400, 700),
                                (-1200, 1200),
                                num_points=8,
                            )
                            ramp_A = [
                                {
                                    "center": (p[0], p[1], -ii * 225),
                                    "radius": 550,
                                }
                                for ii, p in enumerate(ramp_A_points)
                            ]
                            ramp_B_points = sample_quadratic_bezier(
                                (250, -900),
                                (1400, 700),
                                (1200, 1200),
                                num_points=8,
                            )
                            ramp_B = [
                                {
                                    "center": (p[0], p[1], -ii * 225),
                                    "radius": 550,
                                }
                                for ii, p in enumerate(ramp_B_points)
                            ]
                            ramp_C_points = sample_quadratic_bezier(
                                (-250, 3800),
                                (-1400, 2500),
                                (-1200, 1800),
                                num_points=8,
                            )
                            ramp_C = [
                                {
                                    "center": (p[0], p[1], -ii * 225),
                                    "radius": 550,
                                }
                                for ii, p in enumerate(ramp_C_points)
                            ]
                            ramp_D_points = sample_quadratic_bezier(
                                (250, 3800),
                                (1400, 2500),
                                (1200, 1800),
                                num_points=8,
                            )
                            ramp_D = [
                                {
                                    "center": (p[0], p[1], -ii * 225),
                                    "radius": 550,
                                }
                                for ii, p in enumerate(ramp_D_points)
                            ]
                            points, _, path = generate_complete_road_traversal(
                                40, 800, 5000, 5000
                            )
                            pillars.append(
                                [
                                    {
                                        "Location": (
                                            points[p][0] - 2400,
                                            points[p][1] - 1000,
                                            -100,
                                        ),
                                        "Range": 30,
                                        "NoiseRange": 1,
                                    }
                                    for p in path
                                ]
                            )
                            room = (
                                [initial_shape]
                                + [pit]
                                + [ramp_A]
                                + [ramp_B]
                                + [ramp_C]
                                + [ramp_D]
                            )
                            new_room = room
                            traversal_paths = (
                                [[0, 1]]
                                + [[0, 1]]
                                + [[0, 1, 2, 3, 4, 5, 6, 7]] * 4
                            )
                    print(room_method)
                    first_point = initial_shape[0]
                    second_point = initial_shape[1]
                    first_entrances = (
                        [
                            (
                                first_point["center"][0],
                                first_point["center"][1]
                                - first_point["radius"],
                                first_point["center"][2] + 250,
                                0,
                                0,
                                -90,
                                "Entrance",
                            )
                        ]
                        + [
                            (
                                first_point["center"][0]
                                - first_point["radius"],
                                first_point["center"][1],
                                first_point["center"][2] + 250,
                                0,
                                0,
                                180,
                                "Entrance",
                            )
                        ]
                        + [
                            (
                                first_point["center"][0]
                                + first_point["radius"],
                                first_point["center"][1],
                                first_point["center"][2] + 250,
                                0,
                                0,
                                0,
                                "Entrance",
                            )
                        ]
                    )
                    second_entrances = (
                        [
                            (
                                second_point["center"][0],
                                second_point["center"][1]
                                + second_point["radius"],
                                second_point["center"][2] + 250,
                                0,
                                0,
                                90,
                                "Exit",
                            )
                        ]
                        + [
                            (
                                second_point["center"][0]
                                + second_point["radius"],
                                second_point["center"][1],
                                second_point["center"][2] + 250,
                                0,
                                0,
                                0,
                                "Exit",
                            )
                        ]
                        + [
                            (
                                second_point["center"][0]
                                - second_point["radius"],
                                second_point["center"][1],
                                second_point["center"][2] + 250,
                                0,
                                0,
                                180,
                                "Exit",
                            )
                        ]
                    )
                    if room_method == "pit":
                        second_point = pit[-1]
                        second_entrances += [
                            (
                                second_point["center"][0],
                                second_point["center"][1]
                                + second_point["radius"],
                                second_point["center"][2] + 250,
                                0,
                                0,
                                90,
                                "Exit",
                            )
                        ]
                        second_entrances += [
                            (
                                second_point["center"][0]
                                + second_point["radius"],
                                second_point["center"][1],
                                second_point["center"][2] + 250,
                                0,
                                0,
                                0,
                                "Exit",
                            )
                        ]
                        second_entrances += [
                            (
                                second_point["center"][0]
                                - second_point["radius"],
                                second_point["center"][1],
                                second_point["center"][2] + 250,
                                0,
                                0,
                                180,
                                "Exit",
                            )
                        ]
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_PATTERNEDROOM{room_num}.json"
            room_name = f"{tag.upper()}_PATTERNEDROOM"
        case "ramp":
            ramp_type = random.choice(["up", "down", "flat"])
            match ramp_type:
                case "up":
                    height = 1500
                case "down":
                    height = -1500
                case "flat":
                    height = 0
            ramp_A = [
                {"center": (-300, -600, -50), "radius": 800},
                {"center": (-300, 1500, height), "radius": 800},
            ]
            ramp_B = [
                {"center": (300, -600, -50), "radius": 800},
                {"center": (300, 1500, height), "radius": 800},
            ]
            ramp_C = [
                {"center": (-300, 3500, -50), "radius": 800},
                {"center": (-300, 1500, height), "radius": 800},
            ]
            ramp_D = [
                {"center": (300, 3500, -50), "radius": 800},
                {"center": (300, 1500, height), "radius": 800},
            ]
            separation_net, _, path = generate_complete_road_traversal(
                140, 800, 6000, 3000
            )
            pillars = []
            pe = []
            z_offset = -2000 if ramp_type == "down" else 0
            pillars.append(
                [
                    {
                        "Location": (
                            0,
                            separation_net[p][0] - 1800,
                            separation_net[p][1] + z_offset,
                        ),
                        "Range": 80,
                        "NoiseRange": 10,
                    }
                    for p in path
                ]
            )
            shifted_room = shift_room(a2, (0, 1500, height))
            room = [ramp_A] + [ramp_B] + [ramp_C] + [ramp_D] + [shifted_room]
            new_room = room
            traversal_paths = [[0, 1]] * 4 + [a1]
            first_entrances = [
                (
                    ramp_A[0]["center"][0],
                    ramp_A[0]["center"][1] - ramp_A[0]["radius"],
                    ramp_A[0]["center"][2] + 250,
                    0,
                    0,
                    -90,
                    "Entrance",
                )
            ] + [
                (
                    ramp_B[0]["center"][0],
                    ramp_B[0]["center"][1] - ramp_B[0]["radius"],
                    ramp_B[0]["center"][2] + 250,
                    0,
                    0,
                    -90,
                    "Entrance",
                )
            ]
            second_entrances = [
                (
                    ramp_C[0]["center"][0],
                    ramp_C[0]["center"][1] + ramp_C[0]["radius"],
                    ramp_C[0]["center"][2] + 250,
                    0,
                    0,
                    90,
                    "Exit",
                )
            ] + [
                e
                for e in generate_entrances(shifted_room, a3)
                if e[-1] != "Entrance"
            ]
            name = f"RMA_ICGEN_{tag.upper()}_RAMPROOM{room_num}.json"
            room_name = f"{tag.upper()}_RAMPROOM"
        case "trench":
            trench_start, trench_end = random.sample(a2, k=2)
            angle = 30
            bezier_middle_point = compute_side_tunnel(
                trench_start, trench_end, "+"
            )
            trench = [
                {
                    "center": (a[0], a[1], 100),
                    "radius": random.randrange(500, 800),
                }
                for a in sample_quadratic_bezier(
                    trench_start["center"],
                    bezier_middle_point,
                    trench_end["center"],
                    num_points=5,
                )
            ]
            new_room = [a2] + [trench]
            for ii in range(8):
                sign = 1 if ii <= 3 else -1
                trench_a = [
                    {
                        "center": (
                            p["center"][0],
                            p["center"][1],
                            trench_start["center"][2] + 200 - 200 * ii,
                        ),
                        "radius": 500,
                    }
                    for p in trench
                ]
                new_room += [trench_a]
            traversal_paths = [a1] + [[0, 1, 2, 3, 4]] * 8

            first_point = max(
                new_room[0], key=lambda x: x["center"][0] + x["center"][1]
            )
            second_point = new_room[-1][-1]
            first_entrances = [
                e for e in generate_entrances(a2, a3) if e[6] != "Exit"
            ]
            direction_vector = (
                np.array(trench_end["center"]) - bezier_middle_point
            )
            second_entrances = generate_exit_w_vector(
                trench_a,
                (bezier_middle_point[0], bezier_middle_point[1]),
                (direction_vector[0], direction_vector[1]),
            )
            # second_entrances = [
            #     (
            #         second_point["center"][0],
            #         second_point["center"][1] + second_point["radius"],
            #         second_point["center"][2] + 250,
            #         0,
            #         0,
            #         90,
            #         "Exit",
            #     )
            # ]
            name = f"RMA_ICGEN_{tag.upper()}_TRENCHROOM{room_num}.json"
            room_name = f"{tag.upper()}_TRENCHROOM"
            pillars = []
            pe = []
        case "tunnel":
            A = (0, 0)
            B = random.choice([(10, 5), (-10, -5)])

            tunnel_method = random.choice(["bezier", "fractal"])
            tunnel_method = "fractal"
            match tunnel_method:
                case "bezier":
                    curve, _ = random_bezier_path(
                        A, B, num_controls=3, spread=0.8, num_samples=20
                    )
                case "fractal":
                    curve = midpoint_displacement(
                        A, B, depth=6, roughness=0.35
                    )
            height_function = random.choice([-1, 1])
            print(height_function)
            curve = points_2d_to_shape(
                curve,
                2,
                lambda i, x, y: i
                * height_function
                * random.uniform(0.15, 0.2),
            )
            curve = scale_spheres(curve, 400)
            room = curve
            new_room = [room]
            traversal_paths = [list(range(len(curve)))]
            first_point = room[0]
            second_point = room[-1]
            if B == (-10, -5):
                first_entrances = [
                    (
                        first_point["center"][0] - first_point["radius"],
                        first_point["center"][1],
                        first_point["center"][2] + 250,
                        0,
                        0,
                        180,
                        "Entrance",
                    )
                ]
                second_entrances = [
                    (
                        second_point["center"][0],
                        second_point["center"][1] - second_point["radius"],
                        second_point["center"][2] + 250,
                        0,
                        0,
                        180,
                        "Exit",
                    )
                ]
            else:
                first_entrances = [
                    (
                        first_point["center"][0] - first_point["radius"],
                        first_point["center"][1],
                        first_point["center"][2] + 250,
                        0,
                        0,
                        180,
                        "Entrance",
                    )
                ]
                second_entrances = [
                    (
                        second_point["center"][0],
                        second_point["center"][1] + second_point["radius"],
                        second_point["center"][2] + 250,
                        0,
                        0,
                        90,
                        "Exit",
                    )
                ]
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_TUNNELROOM{room_num}.json"
            room_name = f"{tag.upper()}_TUNNELROOM"
        case "doublefloor":
            x, y, z, _, _, angle, _ = generate_entrances(a2, a3)[-1]
            # angle = (angle + 180) % 360
            second_shaft = (
                x + 2000 * math.cos(np.deg2rad(angle)),
                y + 2000 * math.sin(np.deg2rad(angle)),
                2000,
            )
            shaft = [
                {"center": (x, y, z), "radius": 600},
                {
                    "center": second_shaft,
                    "radius": 600,
                },
            ]
            upper_floor = shift_room(b2, second_shaft)
            room = [a2] + [shaft] + [upper_floor]
            new_room = room
            traversal_paths = [a1] + [[0, 1]] + [b1]
            first_point = room[0][0]
            second_point = room[-1][-1]
            first_entrances = [
                e for e in generate_entrances(a2, a3) if e[6] != "Exit"
            ]
            second_entrances = [
                e
                for e in generate_entrances(upper_floor, b3)
                if e[6] != "Entrance"
            ]
            pillars = []
            pe = []
            name = f"RMA_ICGEN_{tag.upper()}_DOUBLEFLOORROOM{room_num}.json"
            room_name = f"{tag.upper()}_DOUBLEFLOORROOM"

    if not mixed:
        tags = {
            "small": ["Rooms.Linear.CustomSmall"],
            "medium": ["Rooms.Linear.CustomMedium"],
            "big": ["Rooms.Linear.CustomBig"],
        }
    else:
        tags = random.choice(
            [
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Flat"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Flat"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Flat"],
                },
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Vertical"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Vertical"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Vertical"],
                },
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Slope"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Slope"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Slope"],
                },
            ]
        )
    match room_type:
        case "maze":
            bounds = 4500
        case "patterned":
            bounds = 5000
        case _:
            bounds = size + 800
    # Create the room:
    room_json = build_json_multiple_ffill(
        traversal_paths,
        new_room,
        first_entrances + second_entrances,
        pillars,
        room_num,
        room_name,
        {"bounds": bounds, "tags": tags[tag]},
        pe,
    )

    try:
        path = Path("generated")
        path.mkdir(parents=True, exist_ok=True)
        with open(f"generated/{name}", "w") as f:
            json.dump(room_json, f, indent=4)
        logging.info(f"Written {name} to generated/{name}, method: doubleroom")
    except Exception as e:
        logging.error(f"Couldn't write {name}: {e}")


def points_2d_to_shape(arr, radius, f):
    return [
        {"center": (p[0], p[1], f(ii, p[0], p[1])), "radius": radius}
        for ii, p in enumerate(arr)
    ]


def perpendicular_2d_normalized(v):
    x, y, _ = v
    px, py = -y, x

    magnitude = math.hypot(px, py)
    if magnitude == 0:
        raise ValueError("Zero vector has no defined perpendicular direction")

    return (px / magnitude, py / magnitude)


def generate_lindenmayer_base():
    base = [[(0, i)] for i in range(random.randrange(3, 5))]
    for _ in range(random.randrange(5, 8)):
        node = random.randrange(len(base))
        angle = random.choice([180, 135, 45, 0])
        new_node = (0, 0)
        match angle:
            case 180:
                new_node = (base[node][-1][0] - 1, base[node][-1][1])
            case 135:
                new_node = (base[node][-1][0] - 1, base[node][-1][1] + 1)
            case 90:
                new_node = (base[node][-1][0], base[node][-1][1] + 1)
            case 45:
                new_node = (base[node][-1][0] + 1, base[node][-1][1] + 1)
            case 0:
                new_node = (base[node][-1][0] + 1, base[node][-1][1])
        base[node].append(new_node)
    base.append([(0, i) for i in range(len(base))])
    return base


def generate_l_rooms(mixed):

    for size, size_tag in [(3000, "small"), (3000, "medium"), (3000, "big")]:
        for i in range(16):
            l_base = generate_lindenmayer_base()
            base = [
                [
                    {"center": (c[0], c[1], 0), "radius": 0.4, "height": 1000}
                    for c in b
                ]
                for b in l_base
            ]
            scale_factor = compute_room_scale_factor(
                list(flatten(base)), 1, optional_size=size
            )
            scaled_spheres = scale_spheres_multiple_ffill(base, scale_factor)
            scaled_spheres = [
                [
                    {
                        "center": c["center"],
                        "radius": c["radius"],
                        "height": 850,
                    }
                    for c in b
                ]
                for b in scaled_spheres
            ]
            pit_index = random.randrange(1, len(scaled_spheres[-1]))
            scaled_spheres[-1][pit_index]["floor_depth"] = -500
            first_point, last_point = (
                scaled_spheres[0][0],
                scaled_spheres[-1][-1],
            )
            entrance = (
                first_point["center"][0],
                first_point["center"][1] - first_point["radius"],
                first_point["center"][2],
                0,
                0,
                -90,
                "Entrance",
            )
            exit = (
                last_point["center"][0],
                last_point["center"][1] + last_point["radius"],
                last_point["center"][2],
                0,
                0,
                90,
                "Exit",
            )
            pillars = flatten([generate_pillars(p) for p in scaled_spheres])
            pe_features = []
            if not mixed:
                tags = {
                    "small": ["Rooms.Linear.CustomSmall"],
                    "medium": ["Rooms.Linear.CustomMedium"],
                    "big": ["Rooms.Linear.CustomBig"],
                }
            else:
                tags = random.choice(
                    [
                        {
                            "small": [
                                "Rooms.Linear.Small",
                                "Rooms.Linear.Flat",
                            ],
                            "medium": [
                                "Rooms.Linear.Medium",
                                "Rooms.Linear.Flat",
                            ],
                            "big": ["Rooms.Linear.Big", "Rooms.Linear.Flat"],
                        },
                        {
                            "small": [
                                "Rooms.Linear.Small",
                                "Rooms.Linear.Vertical",
                            ],
                            "medium": [
                                "Rooms.Linear.Medium",
                                "Rooms.Linear.Vertical",
                            ],
                            "big": [
                                "Rooms.Linear.Big",
                                "Rooms.Linear.Vertical",
                            ],
                        },
                        {
                            "small": [
                                "Rooms.Linear.Small",
                                "Rooms.Linear.Slope",
                            ],
                            "medium": [
                                "Rooms.Linear.Medium",
                                "Rooms.Linear.Slope",
                            ],
                            "big": ["Rooms.Linear.Big", "Rooms.Linear.Slope"],
                        },
                    ]
                )
            room_json = build_json_multiple_ffill(
                [[i for i in range(len(a))] for a in scaled_spheres],
                scaled_spheres,
                [entrance, exit],
                pillars,
                i,
                f"{size_tag.upper()}_LINDEN",
                {"bounds": size, "tags": tags[size_tag]},
                pe_features,
            )

            name = f"RMA_ICGEN_{size_tag.upper()}_LINDEN{i}.json"

            try:
                path = Path("generated")
                path.mkdir(parents=True, exist_ok=True)
                with open(f"generated/{name}", "w") as f:
                    json.dump(room_json, f, indent=4)
                logging.info(
                    f"Written {name} to generated/{name}, method: linden"
                )
            except Exception as e:
                logging.error(f"Couldn't write {name}: {e}")


def generate_tunnel_rooms(mixed):
    tunnels = {
        "L": [(0, 0, 0), (0, 1, 0), (1, 1, 0)],
        "U": [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
        "arcR": [(0, 0, 0), (0, 1.5, 0)],
        "arcL": [(0, 0, 0), (0, 1.5, 0)],
    }
    sizes = {"small": 2500, "medium": 2500, "big": 4000}
    for tunnel, verticality, size in product(
        tunnels, ["up", "down", "flat"], sizes
    ):
        generate_tunnel_room(
            tunnel, tunnels[tunnel], verticality, size, sizes[size], mixed
        )


def generate_tunnel_room(
    tunnel_type, tunnel_base, verticality, size_tag, size, mixed
):
    match tunnel_type:
        case "L" | "U":
            base = tunnel_base
        case "arcL":
            base = sample_quadratic_bezier(
                tunnel_base[0], (2, 0.75, 0), tunnel_base[1], num_points=10
            )
        case "arcR":
            base = sample_quadratic_bezier(
                tunnel_base[0], (-2, 0.75, 0), tunnel_base[1], num_points=10
            )

    match verticality:
        case "flat":
            base = [{"center": b, "radius": 0.4} for b in base]
            number = 0
        case "up":
            if tunnel_type == "U":
                base = [
                    {"center": (b[0], b[1], ii / 2.5), "radius": 0.4}
                    for ii, b in enumerate(base)
                ]
            else:
                base = [
                    {"center": (b[0], b[1], b[1] * 0.8), "radius": 0.4}
                    for b in base
                ]
            number = 1
        case "down":
            if tunnel_type == "U":
                base = [
                    {"center": (b[0], b[1], -ii / 2.5), "radius": 0.4}
                    for ii, b in enumerate(base)
                ]
            else:
                base = [
                    {"center": (b[0], b[1], -b[1] * 0.8), "radius": 0.4}
                    for b in base
                ]
            number = 2
    scale_factor = compute_room_scale_factor(base, 1, optional_size=size)
    scaled_spheres = scale_spheres(base, scale_factor)
    first_point, last_point = scaled_spheres[0], scaled_spheres[-1]
    entrance = (
        first_point["center"][0],
        first_point["center"][1] - first_point["radius"],
        first_point["center"][2],
        0,
        0,
        -90,
        "Entrance",
    )
    exit = (
        last_point["center"][0],
        last_point["center"][1] + last_point["radius"],
        last_point["center"][2],
        0,
        0,
        0,
        "Exit",
    )
    pillars = []
    pe_features = []
    if not mixed:
        tags = {
            "small": ["Rooms.Linear.CustomSmall"],
            "medium": ["Rooms.Linear.CustomMedium"],
            "big": ["Rooms.Linear.CustomBig"],
        }
    else:
        tags = random.choice(
            [
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Flat"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Flat"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Flat"],
                },
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Vertical"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Vertical"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Vertical"],
                },
                {
                    "small": ["Rooms.Linear.Small", "Rooms.Linear.Slope"],
                    "medium": ["Rooms.Linear.Medium", "Rooms.Linear.Slope"],
                    "big": ["Rooms.Linear.Big", "Rooms.Linear.Slope"],
                },
            ]
        )
    room_json = build_json(
        list(range(len(scaled_spheres))),
        scaled_spheres,
        [entrance, exit],
        pillars,
        number,
        f"tunnel_{size_tag.upper()}{tunnel_type.upper()}",
        {"bounds": size, "tags": tags[size_tag]},
        pe_features,
    )

    name = f"RMA_ICGEN_TUNNEL_{size_tag.upper()}{tunnel_type.upper()}{number}.json"

    try:
        path = Path("generated")
        path.mkdir(parents=True, exist_ok=True)
        with open(f"generated/{name}", "w") as f:
            json.dump(room_json, f, indent=4)
        logging.info(f"Written {name} to generated/{name}, method: tunnel")
    except Exception as e:
        logging.error(f"Couldn't write {name}: {e}")


def sample_quadratic_bezier(p0, p1, p2, num_points=100):
    """
    Sample points along a quadratic Bézier curve.

    Parameters:
        p0, p1, p2 : tuple or list of (x, y)
            Start point, control point, and end point.
        num_points : int
            Number of samples along the curve.

    Returns:
        numpy.ndarray of shape (num_points, 2)
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    t = np.linspace(0, 1, num_points)
    points = (
        ((1 - t) ** 2)[:, None] * p0
        + (2 * (1 - t) * t)[:, None] * p1
        + (t**2)[:, None] * p2
    )

    return points


def main():

    parser = argparse.ArgumentParser(
        description="Procedural generation of DRG rooms."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        nargs="?",
        default="settings/settings.json",
        help="Path to the user configuration file. If not defined defaults to config/config.json.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        nargs="?",
        default="generated",
        help="Path where the room JSONs will be written to. If not specified defaults to generated/.",
    )
    parser.add_argument(
        "-k",
        "--keys",
        nargs="+",
        default=None,
        help="List of keys from the settings file that will be used for the generation.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="If present, the program will print the images of the caves in the generated/debug directory.",
    )
    parser.add_argument(
        "-t",
        "--tunnels",
        action="store_true",
        help="If present, we will generate the fixed tunnel rooms.",
    )
    parser.add_argument(
        "-r",
        "--rdouble",
        action="store_true",
        help="If present, we will generate the fixed double rooms.",
    )
    parser.add_argument(
        "-m",
        "--mixed",
        action="store_true",
        help="Mixed tags for the mixed mod.",
    )
    parser.add_argument(
        "-l",
        "--linden",
        action="store_true",
        help="Lindenmayer rooms",
    )
    parser.add_argument(
        "-p",
        "--pitrooms",
        action="store_true",
        help="Pit rooms",
    )

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=20,
        help="Number of rooms for each key, defaults to 20.",
    )

    parser.add_argument(
        "-s",
        "--starting-number",
        type=int,
        default=0,
        help="Where to start the numbering of the rooms, defaults to 0.",
    )
    args = parser.parse_args()

    if args.tunnels:
        generate_tunnel_rooms(args.mixed)
        return 0

    if args.rdouble:
        run_doublerooms(args.mixed)
        return 0

    if args.linden:
        generate_l_rooms(args.mixed)
        return 0

    if args.pitrooms:
        run_pitrooms(args.mixed)
        return 0

    try:
        with open(args.config_path, "r") as f:
            user_settings = json.load(f)
    except Exception as e:
        logging.error(
            f"Error opening configuration file {args.config_path}: {e}"
        )
        return 1

    if args.keys is not None:
        keys = args.keys
    else:
        keys = list(user_settings.keys())

    rooms_count = {k: 0 for k in keys}

    for tag in keys:
        while rooms_count[tag] < args.number:
            name = (
                f"RMA_ICGEN_{tag}{rooms_count[tag]+args.starting_number}.json"
            )
            # Create tilemap:
            tilemap_method = random.choice(["automaton", "random_walk"])
            verticality = user_settings[tag].get("verticality")
            if verticality is None:
                verticality = random.choice(
                    ["up", "down", "flat", "heightmap"]
                )
            elif isinstance(verticality, list):
                verticality = random.choice(verticality)
            cliff = random.choice([True, False])
            cliff = False
            tries = 1
            if user_settings[tag]["PE"]:
                tilemap_method = random.choice(["automaton", "DLA", "torus"])
                match tilemap_method:
                    case "automaton":
                        tilemap = run_automata(
                            inverted=user_settings[tag]["inverted"],
                            randomize_size=user_settings[tag][
                                "randomize_size"
                            ],
                            remove_pockets=user_settings[tag][
                                "remove_pockets"
                            ],
                            size=user_settings[tag]["size"],
                        )
                        scale_coeff = 1
                    case "random_walk":
                        tilemap = create_random_walk(100, 100, 500)
                        scale_coeff = 1
                    case "DLA":
                        tilemap = create_DLA_tilemap(20, 20, 0.05, 0.4)
                        scale_coeff = 1.2
                    case "torus":
                        generate_torus_room(
                            name,
                            3000,
                            rooms_count[tag],
                        )
                        rooms_count[tag] += 1
                        continue
                (
                    traversal_path,
                    scaled_circles,
                    graph,
                    heightmap,
                    ca_grid,
                    selected_circles,
                ) = create_room(
                    tilemap,
                    height_scale=10,
                    max_cube_vol=user_settings[tag].get("max_cube_vol"),
                    scale_coeff=scale_coeff,
                    verticality=verticality,
                    cliff=cliff,
                    pe=True,
                )
            else:
                while True:
                    match tilemap_method:
                        case "automaton":
                            tilemap = run_automata(
                                inverted=user_settings[tag]["inverted"],
                                randomize_size=user_settings[tag][
                                    "randomize_size"
                                ],
                                remove_pockets=user_settings[tag][
                                    "remove_pockets"
                                ],
                                size=user_settings[tag]["size"],
                            )
                            scale_coeff = 1
                        case "random_walk":
                            tilemap = create_random_walk(20, 20, 100)
                            scale_coeff = 1
                        case "DLA":
                            # tilemap = create_DLA_tilemap(20, 20, random.uniform(0.1, 0.2), random.uniform(0.1, 0.5))
                            tilemap = create_DLA_tilemap(20, 20, 0.05, 0.4)
                            scale_coeff = 1.1
                    try:
                        (
                            traversal_path,
                            scaled_circles,
                            graph,
                            heightmap,
                            ca_grid,
                            selected_circles,
                        ) = create_room(
                            tilemap,
                            height_scale=10,
                            max_cube_vol=user_settings[tag].get(
                                "max_cube_vol"
                            ),
                            scale_coeff=scale_coeff,
                            verticality=verticality,
                            cliff=cliff,
                        )
                        break
                    except RoomTooSmall:
                        tries += 1
                        continue

            if args.debug:
                path = Path("generated") / "debug"
                path.mkdir(parents=True, exist_ok=True)
                plt.cla()
                plt.imshow(heightmap, cmap="terrain")
                plt.savefig(f"{path / name}.png")
                plt.cla()
                plt.imshow(ca_grid, cmap="gray")
                for x, y, r in selected_circles:
                    circle = plt.Circle(
                        (x, y), r, color="red", fill=False, linewidth=1
                    )
                    plt.gca().add_patch(circle)
                plt.gca().set_aspect("equal")
                plt.savefig(f"{path}/circles_{name}.png")

            # Build the JSON:
            if tilemap_method != "DLA":
                pillars = generate_pillars(scaled_circles)
            else:
                pillars = []
            # pillars = []
            entrances = generate_entrances(scaled_circles, graph)
            if user_settings[tag]["PE"]:
                pe_features = generate_PE_features(scaled_circles)
            else:
                pe_features = []
            room_json = build_json(
                traversal_path,
                scaled_circles,
                entrances,
                pillars,
                rooms_count[tag],
                tag,
                user_settings[tag],
                pe_features,
            )

            try:
                path = Path("generated")
                path.mkdir(parents=True, exist_ok=True)
                with open(f"generated/{name}", "w") as f:
                    json.dump(room_json, f, indent=4)
                logging.info(
                    f"Written {name} to {args.output_path}/{name}, method: {tilemap_method}, vert: {verticality}, cliff: {cliff}, tries: {tries}"
                )
                rooms_count[tag] += 1
            except Exception as e:
                logging.error(f"Couldn't write {name}: {e}")


if __name__ == "__main__":
    main()
