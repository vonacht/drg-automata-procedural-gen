import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random
import math
from collections import defaultdict


# -----------------------------
# Check if point is inside ANY disk
# -----------------------------
def in_any_disk(p, disks):
    x, y = p
    for cx, cy, r in disks:
        if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
            return True
    return False


# -----------------------------
# Bounding box of all disks
# -----------------------------
def bounding_box(disks):
    xs = [x - r for x, y, r in disks] + [x + r for x, y, r in disks]
    ys = [y - r for x, y, r in disks] + [y + r for x, y, r in disks]
    return min(xs), max(xs), min(ys), max(ys)


# -----------------------------
# Global Poisson disk sampling
# -----------------------------
def poisson_disk_union(disks, r_min, k=30):
    xmin, xmax, ymin, ymax = bounding_box(disks)

    cell_size = r_min / np.sqrt(2)
    grid_w = int((xmax - xmin) / cell_size) + 1
    grid_h = int((ymax - ymin) / cell_size) + 1

    grid = [[None for _ in range(grid_h)] for _ in range(grid_w)]

    def grid_coords(p):
        return int((p[0] - xmin) / cell_size), int((p[1] - ymin) / cell_size)

    samples = []
    active = []

    # Initial point
    while True:
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        if in_any_disk((x, y), disks):
            samples.append((x, y))
            active.append((x, y))
            gx, gy = grid_coords((x, y))
            grid[gx][gy] = (x, y)
            break

    # Sampling loop
    while active:
        idx = random.randint(0, len(active) - 1)
        base = active[idx]
        found = False

        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(r_min, 2 * r_min)

            new_pt = (
                base[0] + r * math.cos(angle),
                base[1] + r * math.sin(angle),
            )

            if not (xmin <= new_pt[0] <= xmax and ymin <= new_pt[1] <= ymax):
                continue

            if not in_any_disk(new_pt, disks):
                continue

            gx, gy = grid_coords(new_pt)

            ok = True
            for i in range(max(0, gx - 2), min(grid_w, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_h, gy + 3)):
                    neighbor = grid[i][j]
                    if neighbor is not None:
                        dx = neighbor[0] - new_pt[0]
                        dy = neighbor[1] - new_pt[1]
                        if dx * dx + dy * dy < r_min**2:
                            ok = False
                            break
                if not ok:
                    break

            if ok:
                samples.append(new_pt)
                active.append(new_pt)
                grid[gx][gy] = new_pt
                found = True

        if not found:
            active.pop(idx)

    return np.array(samples)


def build_graph_with_edges(tri):
    graph = defaultdict(lambda: defaultdict(int))

    for simplex in tri.simplices:
        for i in range(3):
            a = simplex[i]
            b = simplex[(i + 1) % 3]
            graph[a][b] += 1
            graph[b][a] += 1

    return graph


def odd_degree_nodes(graph):
    return [v for v in graph if sum(graph[v].values()) % 2 == 1]


def pair_odds(graph, points):
    odds = odd_degree_nodes(graph)
    odds = odds.copy()

    while len(odds) > 1:
        u = odds.pop()

        # find closest node
        dists = [(v, np.linalg.norm(points[u] - points[v])) for v in odds]
        v = min(dists, key=lambda x: x[1])[0]

        odds.remove(v)

        # duplicate edge
        graph[u][v] += 1
        graph[v][u] += 1


def eulerian_path(graph, start=0):
    g = {u: dict(v) for u, v in graph.items()}  # copy
    stack = [start]
    path = []

    while stack:
        u = stack[-1]
        if g[u]:
            v = next(iter(g[u]))

            # remove edge
            g[u][v] -= 1
            if g[u][v] == 0:
                del g[u][v]

            g[v][u] -= 1
            if g[v][u] == 0:
                del g[v][u]

            stack.append(v)
        else:
            path.append(stack.pop())

    return path[::-1]


# -----------------------------
# Example disks (overlapping)
# -----------------------------
disks = [(0, 0, 1.0), (1.2, 0.2, 1.0), (0.6, 1.0, 0.9)]


def delaunay_within_room(circles, r_min=50):
    # -----------------------------
    # Sample points globally
    # -----------------------------
    circles = [(c["center"][0], c["center"][1], c["radius"]) for c in circles]
    points = poisson_disk_union(circles, r_min=r_min)

    # -----------------------------
    # Delaunay triangulation
    # -----------------------------
    tri = Delaunay(points)
    graph = build_graph_with_edges(tri)

    # fix odd degrees
    pair_odds(graph, points)

    # get traversal
    edge_path = eulerian_path(graph, start=0)

    # coordinates
    path_coords = points[edge_path]
    return path_coords
