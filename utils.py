import math
import numpy as np
import random


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
        new_center = center_point + np.array([scale_factor, scale_factor, scale_factor]) * (
            center - center_point
        )
        new_radius = s["radius"] * scale_factor
        scaled.append({"center": tuple(new_center), "radius": new_radius})

    return scaled


# Function to check if two circles intersect or touch:
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


def compute_room_volume(circles) -> float:
    min_x = min(circle["center"][0] - circle["radius"] for circle in circles)
    max_x = max(circle["center"][0] + circle["radius"] for circle in circles)
    min_y = min(circle["center"][1] - circle["radius"] for circle in circles)
    max_y = max(circle["center"][1] + circle["radius"] for circle in circles)
    min_z = min(circle["center"][2] for circle in circles)
    max_z = max(circle["center"][2] + circle["radius"] for circle in circles)

    return (max_x - min_x) * (max_y - min_y) * (max_z - min_z)


def compute_room_scale_factor(circles, scale_coeff: int, optional_size=None) -> float:
    target_max_volume = random.choice([2000, 3000, 4000]) if optional_size is None else optional_size
    print(target_max_volume)
    target_max_volume = target_max_volume ** 3 * scale_coeff
    current_volume = compute_room_volume(circles)
    scale_factor = target_max_volume / current_volume
    return math.cbrt(scale_factor)
