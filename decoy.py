import numpy as np
from scipy.spatial import ConvexHull
import random

from heapq import heappush, heappop


def choose_edge_spheres(spheres):
    points = np.array([s["center"] for s in spheres])

    hull = ConvexHull(points)
    edge_indices = hull.vertices
    start_idx, end_idx = random.sample(list(edge_indices), 2)

    return start_idx, end_idx


def build_grid(spheres, voxel_size=1.0, margin=10):

    centers = np.array([s["center"] for s in spheres])

    mins = centers.min(axis=0) - margin
    maxs = centers.max(axis=0) + margin

    shape = np.ceil((maxs - mins) / voxel_size).astype(int)

    grid = np.zeros(shape, dtype=bool)

    for sphere in spheres:

        c = np.array(sphere["center"])
        r = sphere["radius"]

        lo = np.floor((c - r - mins) / voxel_size).astype(int)
        hi = np.ceil((c + r - mins) / voxel_size).astype(int)

        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(shape) - 1)

        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):

                    p = mins + voxel_size * np.array([x, y, z])

                    if np.linalg.norm(p - c) <= r:
                        grid[x, y, z] = True

    return grid, mins, voxel_size


def world_to_grid(point, origin, voxel_size):
    return tuple(np.round((np.array(point) - origin) / voxel_size).astype(int))


def astar(grid, start, goal):

    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    open_set = []
    heappush(open_set, (0, start))

    came_from = {}
    gscore = {start: 0}

    while open_set:

        _, current = heappop(open_set)

        if current == goal:

            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            return path[::-1]

        for dx, dy, dz in neighbors:

            nxt = (current[0] + dx, current[1] + dy, current[2] + dz)

            if (
                nxt[0] < 0
                or nxt[1] < 0
                or nxt[2] < 0
                or nxt[0] >= grid.shape[0]
                or nxt[1] >= grid.shape[1]
                or nxt[2] >= grid.shape[2]
            ):
                continue

            if grid[nxt]:
                continue

            cost = np.sqrt(dx * dx + dy * dy + dz * dz)

            tentative = gscore[current] + cost

            if tentative < gscore.get(nxt, float("inf")):

                came_from[nxt] = current
                gscore[nxt] = tentative

                f = tentative + heuristic(nxt, goal)

                heappush(open_set, (f, nxt))

    return None


def grid_to_world(cell, origin, voxel_size):
    return origin + voxel_size * np.array(cell)


def sphere_surface_point(sphere, direction):
    c = np.array(sphere["center"])
    d = direction / np.linalg.norm(direction)

    return c + d * (sphere["radius"] + 1.0)


def create_path(spheres):
    start, end = choose_edge_spheres(spheres)
    start_sphere, end_sphere = spheres[start], spheres[end]
    start_world = sphere_surface_point(start_sphere, np.array([-1, 0, 0]))
    end_world = sphere_surface_point(end_sphere, np.array([1, 0, 0]))

    origin = (0, 0, 0)
    voxel_size = 200
    start = world_to_grid(start_world, origin, voxel_size)
    print(start)
    goal = world_to_grid(end_world, origin, voxel_size)
    grid, _, _ = build_grid(spheres, voxel_size=200, margin=2000)
    print(grid)
    path = astar(grid, start, goal)
    path_world = [grid_to_world(p, origin, voxel_size) for p in path]
    print(path_world)
    return path_world
