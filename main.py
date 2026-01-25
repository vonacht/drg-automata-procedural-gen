from opensimplex import OpenSimplex
from scipy.ndimage import label
import random
import numpy as np
import json
from collections import deque
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from more_itertools import flatten
import math
from mpl_toolkits.mplot3d import Axes3D


class CellularAutomaton:
    def __init__(self, M, N, rule, p_alive=0.5, boundary="wrap"):
        self.M = M
        self.N = N
        self.rule = rule
        self.boundary = boundary

        self.grid = np.random.choice([0, 1], size=(M, N), p=[1 - p_alive, p_alive])

        #self.grid[0, :] = 0
        #self.grid[-1, :] = 0
        #self.grid[:, 0] = 0
        #self.grid[:, -1] = 0

    def invert(self):
        for i in range(self.M):
            for j in range(self.N):
                self.grid[i, j] = 0 if self.grid[i, j] == 1 else 1

    def get_neighbors(self, i, j):
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni, nj = i + di, j + dj

                if self.boundary == "wrap":
                    ni %= self.M
                    nj %= self.N
                    neighbors.append(self.grid[ni, nj])
                else:
                    if 0 <= ni < self.M and 0 <= nj < self.N:
                        neighbors.append(self.grid[ni, nj])
                    else:
                        neighbors.append(0)

        return neighbors

    def step(self):
        new_grid = np.zeros_like(self.grid)

        for i in range(self.M):
            for j in range(self.N):
                state = self.grid[i, j]
                neighbors = self.get_neighbors(i, j)
                new_grid[i, j] = self.rule(state, neighbors)

        self.grid = new_grid

    def run(self, steps, remove_pockets=False):
        for _ in range(steps):
            self.step()
            if remove_pockets and _ == steps - 1:
                self.keep_largest_alive_component()

    # ---------------------------
    # POCKET REMOVAL ALGORITHM
    # ---------------------------
    def remove_dead_pockets(self):
        """
        Fills enclosed regions of dead cells (0s).
        """
        visited = np.zeros_like(self.grid, dtype=bool)
        queue = deque()

        # Start flood fill from boundary dead cells
        for i in range(self.M):
            for j in [0, self.N - 1]:
                if self.grid[i, j] == 1 and not visited[i, j]:
                    queue.append((i, j))
                    visited[i, j] = True

        for j in range(self.N):
            for i in [0, self.M - 1]:
                if self.grid[i, j] == 1 and not visited[i, j]:
                    queue.append((i, j))
                    visited[i, j] = True

        # Flood fill
        while queue:
            i, j = queue.popleft()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.M and 0 <= nj < self.N:
                    if self.grid[ni, nj] == 1 and not visited[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))

        # Fill unvisited dead cells (pockets)
        self.grid[(self.grid == 1) & (~visited)] = 0

    def keep_largest_alive_component(self):
        """
        Keeps only the largest connected component of alive cells (1s).
        All other alive components are removed (set to 0).
        """
        # 1. Label each isolated connected area with a unique ID
        # structure=np.ones((3,3)) would include diagonals; default is 4-connectivity
        labeled_array, num_features = label(self.grid)
        if num_features == 0:
            return None
        # 2. Count the pixels in each labeled area
        # bincount calculates the frequency of each ID (0 is background)
        counts = np.bincount(labeled_array.ravel())

        # 3. Find the ID of the largest area (ignoring index 0, the background)
        print(num_features, counts)
        largest_label = counts[1:].argmax() + 1

        # 4. Create a new map where only the largest_label is kept
        result = (labeled_array == largest_label).astype(int)
        
        self.grid = result


def game_of_life_rule(state, neighbors):
    BIRTH_THRESHOLD = [5, 6, 7, 8]
    SURVIVAL_THRESHOLD = [4, 5, 6, 7, 8]
    alive_neighbors = sum(neighbors)
    if state == 0:
        return 1 if alive_neighbors in BIRTH_THRESHOLD else 0
    else:
        return 1 if alive_neighbors in SURVIVAL_THRESHOLD else 0


def area(grid):
    return sum(sum(row) for row in grid)


def generate_heightmap(tilemap, scale=0.15, height_scale=5, seed=42):
    noise = OpenSimplex(seed)
    h, w = tilemap.shape
    heightmap = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            if tilemap[y, x] == 1:
                n = noise.noise2(x * scale, y * scale)
                n = (n + 1) / 2  # [-1,1] → [0,1]
                heightmap[y, x] = n * height_scale

    return heightmap


def scale_spheres(spheres, scale_factor, center_point=None):
    """
    Scale a list of spheres in 3D.

    Parameters:
    spheres      : list of dicts, each with 'center' (x,y,z) and 'radius'
    scale_factor : float, how much to scale
    center_point : optional tuple (x,y,z), the point to scale around.
                   If None, scales around origin (0,0,0)

    Returns:
    List of scaled spheres
    """
    if center_point is None:
        center_point = np.array([0.0, 0.0, 0.0])
    else:
        center_point = np.array(center_point)

    scaled = []
    for s in spheres:
        center = np.array(s["center"])
        # Scale relative to center_point
        new_center = center_point + np.array([scale_factor, scale_factor, 1]) * (center - center_point)
        new_radius = s["radius"] * scale_factor
        scaled.append({"center": tuple(new_center), "radius": new_radius})

    return scaled


# Function to check if two circles intersect or touch
def circles_connected(c1, c2):
    x1, y1 = c1["center"][0], c1["center"][1]
    r1 = c1["radius"]
    x2, y2 = c2["center"][0], c2["center"][1]
    r2 = c2["radius"]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance <= r1 + r2


# Build adjacency list for the circle graph
def build_graph(circles):
    graph = {i: [] for i in range(len(circles))}
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            if circles_connected(circles[i], circles[j]):
                graph[i].append(j)
                graph[j].append(i)
    return graph


# DFS traversal with backtracking path
def dfs_with_backtracking(graph, start=0):
    visited = set()
    path = []

    def dfs(node):
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
                path.append(node)  # backtrack to this node after exploring neighbor

    dfs(start)
    return path


def greedy_dfs_min_backtrack(graph, start=0):
    visited = set()
    path = []
    total_nodes = len(graph)

    def dfs(node):
        visited.add(node)
        path.append(node)
        # Sort neighbors by fewest unvisited neighbors first
        neighbors = sorted(
            [n for n in graph[node] if n not in visited],
            key=lambda n: sum(1 for nbr in graph[n] if nbr not in visited),
        )
        for neighbor in neighbors:
            if len(visited) == total_nodes:
                return  # stop if all nodes visited
            dfs(neighbor)
            if len(visited) < total_nodes:
                path.append(node)  # backtrack only if there are still nodes left

    dfs(start)
    return path


def normalize(angle):
    return angle % (2 * np.pi)


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


def find_leafs(graph):
    leaves = [node for node, neighbors in graph.items() if len(neighbors) == 1]
    return leaves

def generate_PE_features(circles):
    # For PE we choose a few circles at random and put the minehead there.
    number_of_mineheads = random.randrange(1, 4) if len(circles) > 4 else 1
    chosen_circles = random.choices(range(len(circles)), k=number_of_mineheads)
    return [(circles[circle]["center"][0], circles[circle]["center"][1], circles[circle]["center"][2] + 200) for circle in chosen_circles]

def generate_entrances(traversal_path, circles, graph):

    spheres_w_entrance = [0]
    # First decide the entrances:
    circles_w_one_neighbor = [k for k in graph if len(graph[k]) == 1 and k != 0 and 0 not in graph[k]]
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
                    circles[sphere_idx]["center"][2] + circles[sphere_idx]["radius"] * 0.2,
                    0,
                    0,
                    vector_angle,
                    t,
                )
            )

    return entrances


def build_json(traversal_list, circles, entrances, room_num, key, settings, pe_features) -> dict:

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
                        "Location": {
                            "X": feature[0],
                            "Y": feature[1],
                            "Z": feature[2]
                            }
                }})

    result["Name"] = f"RMA_ICGEN_{key}{room_num}"
    result["Bounds"] = settings["bounds"]
    result["Tags"] = settings["tags"]

    return result


def plot_sphere_wireframe(ax, center, radius, color="b", n=30):
    """
    Plot a single sphere as a wireframe.

    Parameters:
    ax     : matplotlib 3D axis
    center : (x, y, z)
    radius : float
    color  : wireframe color
    n      : resolution
    """
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    ax.plot_wireframe(x, y, z, color=color, linewidth=0.8)

def compute_room_volume(circles) -> float:
    min_x = min(circle[0] - circle[2] for circle in circles)
    max_x = max(circle[0] + circle[2] for circle in circles)
    min_y = min(circle[1] - circle[2] for circle in circles)
    max_y = max(circle[1] + circle[2] for circle in circles)

    return (max_x - min_x) * (max_y - min_y)

def compute_room_scale_factor(circles, max_side: int) -> float:
    target_max_volume = max_side ** 2
    current_volume = compute_room_volume(circles)
    scale_factor = target_max_volume / current_volume
    return int(math.sqrt(scale_factor)) if scale_factor > 1 else 1 

def run_automata(height_scale=20, inverted=False, remove_pockets=True, size=15, iterations=8, max_cube_vol=1500, randomize_size=True):
    M = random.choices([15, size], k=1, weights=[0.66, 0.33])[0] if randomize_size else size
    ca = CellularAutomaton(
        M=M,
        N=M,
        rule=game_of_life_rule,
        p_alive=0.45,  # 25% initially alive
        boundary="wrap",
    )
    ca.run(iterations, remove_pockets=remove_pockets)

    if inverted:
        ca.invert()

    heightmap = generate_heightmap(ca.grid, height_scale=height_scale, seed=random.randrange(10000))

    skeleton = skeletonize(ca.grid)
    # Distance transform
    distance_map = distance_transform_edt(ca.grid)

    # Get skeleton pixels with radius
    ys, xs = np.where(skeleton)
    skeleton_points = [
        (x, y, distance_map[y, x]) for y, x in zip(ys, xs) if distance_map[y, x] > 0
    ]

    # Sort by radius descending
    skeleton_points.sort(key=lambda c: c[2], reverse=True)

    # Filter overlapping circles
    selected_circles = []
    for x, y, r in skeleton_points:
        overlap = False
        for sx, sy, sr in selected_circles:
            # Distance between centers
            d = np.hypot(x - sx, y - sy)
            if d < 0.4 * (r + sr):  # allow some small overlap, adjust 0.8 as needed
                overlap = True
                break
        if not overlap:
            selected_circles.append((x, y, r))

    # We skip empty rooms and rooms with 1 circle (too simple):
    if len(selected_circles) <= 1:
        return (None, None, None, None, None, None, None)

    circles_w_height = []
    for c in selected_circles:
        circles_w_height.append((c[0], c[1], c[2], heightmap[c[1], c[0]]))

    spheres = [{"center": (c[0], c[1], c[3]), "radius": c[2]} for c in circles_w_height]
    center_point = (0, 0, 0) if len(spheres) < 1 else spheres[0]["center"]
    
    scale_factor = compute_room_scale_factor(circles_w_height, max_cube_vol)

    scaled_spheres = scale_spheres(spheres, scale_factor, center_point=center_point)

    graph = build_graph(scaled_spheres)
    # Traverse graph
    traversal_path = greedy_dfs_min_backtrack(graph)
    grid_area = area(ca.grid)

    return grid_area, traversal_path, scaled_spheres, graph, heightmap, ca.grid, selected_circles


settings = {
    "SMALL": {
        "max_cube_vol": 1500,
        "bounds": 2500,
        "inverted": False,
        "PE": False,
        "remove_pockets": True,
        "size": 20,
        "randomize_size": True,
        "tags": ["Rooms.Linear.CustomSmall"],
    },
    "MEDIUM": {
        "max_cube_vol": 4500,
        "bounds": 3500,
        "inverted": False,
        "PE": False,
        "remove_pockets": True,
        "size": 30,
        "randomize_size": True,
        "tags": ["Rooms.Linear.CustomMedium"],
    },
    "BIG": {
        "max_cube_vol": 7000,
        "bounds": 4500,
        "inverted": False,
        "PE": False,
        "remove_pockets": True,
        "size": 45,
        "randomize_size": True,
        "tags": ["Rooms.Linear.CustomBig"],
    },
    "PE": {
        "max_cube_vol": 6000,
        "bounds": 4500,
        "inverted": True,
        "PE": True,
        "remove_pockets": False,
        "size": 50,
        "randomize_size": False,
        "tags": ["Rooms.Linear.CustomPE"],
    },
}

DEBUG = 1
TARGET = 20
rooms_count = {k: 0 for k in settings}

for tag in settings.keys():
    while rooms_count[tag] < TARGET:
        name = f"RMA_ICGEN_{tag}{rooms_count[tag]}.json"
        grid_area, traversal_path, circles, graph, heightmap, ca_grid, selected_circles = run_automata(
            height_scale=random.choices([1000], weights=[0.66], k=1)[0],
            inverted=settings[tag]["inverted"],
            remove_pockets=settings[tag]["remove_pockets"],
            size=settings[tag]["size"],
            max_cube_vol=settings[tag]["max_cube_vol"],
            randomize_size=settings[tag]["randomize_size"],
        )
        print(name)
        if traversal_path is None or len(circles) == 1:
            continue
        if DEBUG:
            plt.cla()
            plt.imshow(heightmap, cmap="terrain")
            plt.savefig(f"generated/debug/{name}.png")
            plt.cla()
            plt.imshow(ca_grid, cmap="gray")
            for x, y, r in selected_circles:
                circle = plt.Circle((x, y), r, color="red", fill=False, linewidth=1)
                plt.gca().add_patch(circle)
            plt.gca().set_aspect("equal")
            plt.savefig(f"generated/debug/circles_{name}.png")


        entrances = generate_entrances(traversal_path, circles, graph)
        if settings[tag]["PE"]:
            pe_features = generate_PE_features(circles)
        else:
            pe_features = []
        room_json = build_json(
            traversal_path, circles, entrances, rooms_count[tag], tag, settings[tag], pe_features
        )
        
        with open(f"generated/{name}", "w") as f:
            json.dump(room_json, f, indent=4)
        rooms_count[tag] += 1
