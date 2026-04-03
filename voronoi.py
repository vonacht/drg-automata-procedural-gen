from collections import defaultdict
from scipy.spatial import Voronoi
import numpy as np
import random


def poisson_disk_sampling(n, min_dist=0.08):
    points = []
    while len(points) < n:
        p = np.random.rand(2)
        if all(np.linalg.norm(p - q) > min_dist for q in points):
            points.append(p)
    return np.array(points)


def compute_voronoi(points):
    return Voronoi(points)


def extract_voronoi_edges(vor):
    edges = []

    for ridge_vertices in vor.ridge_vertices:
        # Ignore infinite ridges
        if -1 in ridge_vertices:
            continue

        v0 = vor.vertices[ridge_vertices[0]]
        v1 = vor.vertices[ridge_vertices[1]]
        edges.append((v0, v1))

    return edges


def make_eulerian(edges, graph):
    odd_nodes = [n for n in graph if len(graph[n]) % 2 == 1]

    new_edges = []

    # Pair nodes naively (better: nearest-neighbor pairing)
    for i in range(0, len(odd_nodes), 2):
        a = odd_nodes[i]
        b = odd_nodes[i + 1]
        new_edges.append((a, b))

    return new_edges


def build_graph(edges):
    graph = defaultdict(list)

    for v0, v1 in edges:
        v0 = tuple(v0)
        v1 = tuple(v1)
        graph[v0].append(v1)
        graph[v1].append(v0)

    return graph


def find_odd_nodes(graph):
    return [node for node in graph if len(graph[node]) % 2 == 1]


def find_eulerian_path(graph):
    graph = {k: v[:] for k, v in graph.items()}  # copy

    odd_nodes = [n for n in graph if len(graph[n]) % 2 == 1]

    # Choose start node
    if len(odd_nodes) == 0:
        start = next(iter(graph))
    elif len(odd_nodes) == 2:
        start = odd_nodes[0]
    else:
        return None  # Not possible

    path = []
    stack = [start]

    while stack:
        v = stack[-1]
        if graph[v]:
            u = graph[v].pop()
            graph[u].remove(v)
            stack.append(u)
        else:
            path.append(stack.pop())

    return path[::-1]


def generate_voronoi_path(n_points, min_dist):
    # Generate random 2D points
    xmin, xmax = -0, 4
    ymin, ymax = 0, 4

    points = poisson_disk_sampling(12, min_dist=min_dist)
    points[:, 0] = xmin + points[:, 0] * (xmax - xmin)
    points[:, 1] = ymin + points[:, 1] * (ymax - ymin)

    points = np.random.rand(10, 2)

    return get_voronoi_traversal(points)


def voronoi_traversal(points, start_index=0):
    """
    Generate a Voronoi diagram and return a DFS traversal path
    (including backtracking).

    Args:
        points (array-like): List of (x, y) coordinates
        start_index (int): Index of starting vertex in Voronoi vertices

    Returns:
        path (list): List of vertex indices visited (with backtracking)
        vertices (ndarray): Voronoi vertices (coordinates)
    """

    # Create Voronoi diagram
    vor = Voronoi(points)

    # Build adjacency list from Voronoi ridges
    graph = {i: set() for i in range(len(vor.vertices))}

    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue  # skip infinite edges
        a, b = ridge
        graph[a].add(b)
        graph[b].add(a)

    # DFS traversal with backtracking
    visited = set()
    path = []

    def dfs(node):
        visited.add(node)
        path.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
                path.append(node)  # backtrack step

    # Guard against empty Voronoi vertices
    if len(vor.vertices) == 0:
        return [], vor.vertices

    start_index = min(start_index, len(vor.vertices) - 1)

    dfs(start_index)

    return path, vor.vertices


def voronoi_traversal_bounded(points, start_index=0, max_edge_length=None):
    """
    Generate a Voronoi diagram and return a DFS traversal path
    (including backtracking), bounded by max_edge_length.

    Args:
        points (array-like): List of (x, y) coordinates
        start_index (int): Index of starting vertex in Voronoi vertices
        max_edge_length (float): Maximum allowed edge length; edges longer
                                 than this are ignored. If None, automatically
                                 set based on point spread.

    Returns:
        path (list): List of vertex indices visited (with backtracking)
        vertices (ndarray): Voronoi vertices (coordinates)
    """
    points = np.asarray(points)
    vor = Voronoi(points)

    if len(vor.vertices) == 0:
        return [], vor.vertices

    # Auto max_edge_length if not provided
    if max_edge_length is None:
        max_edge_length = (
            np.ptp(points, axis=0).max() * 0.5
        )  # half the bounding box size

    # Build adjacency list with edge filtering
    graph = {i: set() for i in range(len(vor.vertices))}
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue  # skip infinite edges
        a, b = ridge
        if (
            np.linalg.norm(vor.vertices[a] - vor.vertices[b])
            <= max_edge_length
        ):
            graph[a].add(b)
            graph[b].add(a)

    # DFS traversal with backtracking
    visited = set()
    path = []

    def dfs(node):
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
                path.append(node)  # backtrack step

    start_index = min(start_index, len(vor.vertices) - 1)
    dfs(start_index)

    return path, vor.vertices


def voronoi_finite_polygons_2d(vor, radius=None):
    """(Standard clipping function from previous step)"""
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for i, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[i]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[i]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[i, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def get_voronoi_traversal(points):
    vor = Voronoi(points)
    # USE THE PREVIOUS FUNCTION HERE to get finite geometry
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # 1. Build Adjacency List from the clipped regions
    adj = defaultdict(set)
    for region in regions:
        # Each region is a list of vertex indices in order
        for i in range(len(region)):
            v1 = region[i]
            v2 = region[(i + 1) % len(region)]  # Loop back to start
            adj[v1].add(v2)
            adj[v2].add(v1)

    # 2. DFS Traversal
    path_indices = []
    visited_edges = set()
    start_node = list(adj.keys())[0]

    def dfs(u):
        path_indices.append(u)
        # Sort neighbors to ensure deterministic pathing
        for v in sorted(list(adj[u])):
            edge = tuple(sorted((u, v)))
            if edge not in visited_edges:
                visited_edges.add(edge)
                dfs(v)
                path_indices.append(u)  # Backtrack

    dfs(start_node)
    return vertices[path_indices]
