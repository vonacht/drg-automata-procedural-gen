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

from automata import CellularAutomaton

from heightmap import generate_heightmap
from utils import (
    compute_room_scale_factor,
    scale_spheres,
    build_graph,
    greedy_dfs_min_backtrack,
)
from json_builder import (
    generate_entrances,
    generate_PE_features,
    build_json,
    generate_pillars,
)

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

def run_automata(inverted=False, randomize_size=False, size=15, iterations=8, remove_pockets=True):
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

def gradient_grid(tilemap, 
                  row_direction="increasing", 
                  col_direction="increasing",
                  row_weight=1.0,
                  col_weight=1.0):
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
    verticality=None
):
    
    verticality = random.choice(["up", "down", "flat", "heightmap"]) if verticality is None else verticality
    cliff = random.choice([True, False])
    match verticality:
        case "heightmap":
            # We get a heightmap with a randomized seed:
            heightmap = generate_heightmap(
                tilemap, height_scale=height_scale, seed=random.randrange(10000)
            )
        case "up":
            heightmap = gradient_grid(tilemap, row_weight=0.3, col_weight=0.3)
        case "down":
            heightmap = gradient_grid(tilemap, col_direction="decreasing", row_direction="decreasing", row_weight=0.3, col_weight=0.3)
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
        (x, y, distance_map[y, x]) for y, x in zip(ys, xs) if distance_map[y, x] > 0
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
    if len(selected_circles) <= 1:
        raise RoomTooSmall

    if verticality in ["up", "down"]:
        normalized_heightmap = heightmap - heightmap[selected_circles[0][1], selected_circles[0][0]]
    else:
        normalized_heightmap = heightmap


    # We attach the height value from the heightmap to each one of the final circles:
    if verticality in ["down"] and cliff:
        print(verticality, cliff)
        num_circles = len(selected_circles) // 2
        spheres = [
                {
                    "center": (c[0], c[1],
                               normalized_heightmap[c[1], c[0]] + cliff_factor(ii, num_circles, verticality)),
                    "radius": c[2]
                    }
                for ii, c in enumerate(selected_circles)
                ]
    else:
        spheres = [
            {"center": (c[0], c[1], normalized_heightmap[c[1], c[0]]), "radius": c[2]}
            for c in selected_circles
        ]
    center_point = (0, 0, 0) if len(spheres) < 1 else spheres[0]["center"]
    # We scale the room based on the max_cube_vol value in the settings:
    scale_factor = compute_room_scale_factor(spheres, optional_size=max_cube_vol, scale_coeff=scale_coeff)
    scaled_spheres = scale_spheres(spheres, scale_factor, center_point=center_point)
    for sphere in scaled_spheres:
        if sphere["radius"] < 300:
            raise RoomTooSmall
    # We create an adjacency graph of all the circles:
    graph = build_graph(scaled_spheres)
    # And we get the traversal path to be able to create our FloodFillLines:
    traversal_path = greedy_dfs_min_backtrack(graph)

    return traversal_path, scaled_spheres, graph, heightmap, tilemap, selected_circles

def cliff_factor(ii, num_circles, verticality):
    result = 0
    match verticality:
        case "up":
            result = 2 if ii > num_circles else 0 
        case "down":
            result = -2 if ii > num_circles else 0 
    return result

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

def generate_tunnel_rooms():
    tunnels = {"L": [(0,0,0), (0,1,0), (1,1,0)],
               "U": [(0,0,0), (0,1,0), (1,1,0), (1,0,0)],
               "arcR": [(0,0,0), (0, 3.5, 0)],
               "arcL": [(0,0,0), (0, 3.5, 0)]}
    sizes = {"small": 1500, "medium": 2500, "big": 4000}
    for tunnel, verticality, size in product(tunnels, ["up", "down", "flat"], sizes):
        generate_tunnel_room(tunnel, tunnels[tunnel], verticality, size, sizes[size])

def generate_tunnel_room(tunnel_type, tunnel_base, verticality, size_tag, size):
    match tunnel_type:
        case "L" | "U":
            base = tunnel_base
        case "arcL": 
            base = sample_quadratic_bezier(tunnel_base[0], (5, 1.5, 0), tunnel_base[1], num_points=10)
        case "arcR": 
            base = sample_quadratic_bezier(tunnel_base[0], (-5, 1.5, 0), tunnel_base[1], num_points=10)

    match verticality:
        case "flat":
            base = [{"center": b, "radius": 0.4} for b in base]
            number = 0
        case "up":
            base = [{"center": (b[0], b[1], b[1]), "radius": 0.5} for b in base]
            number = 1
        case "down":
            base = [{"center": (b[0], b[1], -b[1]), "radius": 0.5} for b in base]
            number = 2
    scale_factor = compute_room_scale_factor(base, 1, optional_size=size)
    scaled_spheres = scale_spheres(base, scale_factor)
    first_point, last_point = scaled_spheres[0], scaled_spheres[-1]
    entrance = (
            first_point["center"][0], first_point["center"][1] - first_point["radius"], first_point["center"][2],
            0, 0, -90, "Entrance"
            )
    exit = (
            last_point["center"][0], last_point["center"][1] + last_point["radius"], last_point["center"][2],
            0, 0, 0, "Exit"
            )
    pillars = []
    pe_features = []
    tags = {"small": ["Rooms.Linear.CustomSmall"],
            "medium": ["Rooms.Linear.CustomMedium"],
            "big": ["Rooms.Linear.CustomBig"]}
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
    points = ((1 - t)**2)[:, None] * p0 \
           + (2 * (1 - t) * t)[:, None] * p1 \
           + (t**2)[:, None] * p2

    return points

def main():

    parser = argparse.ArgumentParser(description="Procedural generation of DRG rooms.")
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
        generate_tunnel_rooms()
        return 0

    try:
        with open(args.config_path, "r") as f:
            user_settings = json.load(f)
    except Exception as e:
        logging.error(f"Error opening configuration file {args.config_path}: {e}")
        return 1

    if args.keys is not None:
        keys = args.keys
    else:
        keys = list(user_settings.keys())

    rooms_count = {k: 0 for k in keys}

    for tag in keys:
        while rooms_count[tag] < args.number:
            name = f"RMA_ICGEN_{tag}{rooms_count[tag]+args.starting_number}.json"
            # Create tilemap:
            tilemap_method = random.choice(["automaton", "random_walk"])
            #tilemap_method = "random_walk"
            match tilemap_method:
                case "automaton":
                    tilemap = run_automata(
                        inverted=user_settings[tag]["inverted"],
                        randomize_size=user_settings[tag]["randomize_size"],
                        remove_pockets=user_settings[tag]["remove_pockets"],
                        size=user_settings[tag]["size"],
                    )
                    scale_coeff = 1
                case "random_walk":
                    tilemap = create_random_walk(50, 50, 100)
                    scale_coeff = 2
                case _:
                    tilemap = create_random_walk(50, 50, 100)
                    scale_coeff = 2
            try:
                traversal_path, scaled_circles, graph, heightmap, ca_grid, selected_circles = (
                    create_room(
                        tilemap,
                        height_scale=10,  # 2000 seems to be a good value gameplay-wise, could be randomized
                        max_cube_vol=user_settings[tag].get("max_cube_vol"),
                        scale_coeff=scale_coeff,
                        verticality=user_settings[tag].get("verticality")
                    )
                )
            except RoomTooSmall:
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
                    circle = plt.Circle((x, y), r, color="red", fill=False, linewidth=1)
                    plt.gca().add_patch(circle)
                plt.gca().set_aspect("equal")
                plt.savefig(f"{path}/circles_{name}.png")

            # Build the JSON:
            pillars = generate_pillars(scaled_circles)
            #pillars = []
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
                logging.info(f"Written {name} to {args.output_path}/{name}, method: {tilemap_method}")
                rooms_count[tag] += 1
            except Exception as e:
                logging.error(f"Couldn't write {name}: {e}")


if __name__ == "__main__":
    main()
