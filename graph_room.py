import random
import math
import heapq
from scipy.spatial import Delaunay


def generate_cave_graph(
    num_rooms: int,
    bounds: tuple[float, float] = (100.0, 100.0),
    max_tunnel_length: float = 30.0,
) -> dict:
    """
    Generates a 2D spatial graph suitable for cave/dungeon layouts.

    :param num_rooms: Total number of cave chambers (N).
    :param bounds: (width, height) area to place chambers.
    :param max_tunnel_length: Maximum distance threshold for connecting adjacent rooms.
    :return: Dictionary containing 'nodes' with coordinates and 'edges' (connections).
    """
    # 1. Assign random 2D coordinates to each room
    nodes = {
        i: (
            round(random.uniform(0, bounds[0]), 2),
            round(random.uniform(0, bounds[1]), 2),
        )
        for i in range(num_rooms)
    }

    adjacency_list = {i: [] for i in range(num_rooms)}

    def euclidean_distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # 2. Connect rooms that are within the max tunnel distance threshold
    for u in range(num_rooms):
        for v in range(u + 1, num_rooms):
            dist = euclidean_distance(nodes[u], nodes[v])
            if dist <= max_tunnel_length:
                adjacency_list[u].append((v, round(dist, 2)))
                adjacency_list[v].append((u, round(dist, 2)))

    # 3. Ensure full connectivity (prevent disconnected isolated caves)
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor, _ in adjacency_list[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(0)

    # If there are disconnected components, connect unvisited rooms to their nearest visited room
    if len(visited) < num_rooms:
        unvisited = set(range(num_rooms)) - visited
        for u in unvisited:
            # Find closest room in the main component
            closest_node = min(
                visited, key=lambda v: euclidean_distance(nodes[u], nodes[v])
            )
            dist = euclidean_distance(nodes[u], nodes[closest_node])

            adjacency_list[u].append((closest_node, round(dist, 2)))
            adjacency_list[closest_node].append((u, round(dist, 2)))
            visited.add(u)

    return {"nodes": nodes, "adjacency_list": adjacency_list}


def generate_cave_graph_w_structure(
    num_rooms: int = 150,
    bounds: tuple[float, float] = (8000.0, 8000.0),
    shape: str = "organic",
    extra_connection_chance: float = 0.15,
) -> dict:
    """Generates a connected 2D spatial graph with configurable cave layouts.

    :param num_rooms: Total number of cave chambers.
    :param bounds: (width, height) bound limits (e.g. 8000x8000).
    :param shape: Layout archetype ('organic', 'caverns', 'linear_tunnel',
        'ring', 'river').
    :param extra_connection_chance: Chance (0.0 - 1.0) to add extra loops to
        MST.
    :return: Dict containing 'nodes' coords and 'adjacency_list'.
    """
    width, height = bounds
    center_x, center_y = width / 2.0, height / 2.0
    nodes = {}

    # --- 1. NODE POSITIONING STRATEGIES ---
    if shape == "caverns":
        # Multi-cluster / Gaussian distribution around cavern hubs
        num_hubs = random.randint(3, 6)
        hubs = [
            (
                random.uniform(width * 0.15, width * 0.85),
                random.uniform(height * 0.15, height * 0.85),
            )
            for _ in range(num_hubs)
        ]
        for i in range(num_rooms):
            cx, cy = random.choice(hubs)
            # Normal distribution scatter around chosen cavern center
            x = max(0, min(width, random.gauss(cx, width * 0.08)))
            y = max(0, min(height, random.gauss(cy, height * 0.08)))
            nodes[i] = (round(x, 2), round(y, 2))

    elif shape == "linear_tunnel":
        # Drunkard's Walk for long winding tunnel systems
        curr_x, curr_y = center_x, center_y
        step_len = min(width, height) / math.sqrt(num_rooms) * 1.2
        angle = random.uniform(0, 2 * math.pi)

        for i in range(num_rooms):
            nodes[i] = (
                round(max(0, min(width, curr_x)), 2),
                round(max(0, min(height, curr_y)), 2),
            )
            angle += random.uniform(-0.8, 0.8)  # Smooth curving
            curr_x += math.cos(angle) * step_len
            curr_y += math.sin(angle) * step_len

    elif shape == "ring":
        # Donut/Atoll structure with radius variance
        base_radius = min(width, height) * 0.35
        for i in range(num_rooms):
            theta = random.uniform(0, 2 * math.pi)
            r = base_radius + random.gauss(0, base_radius * 0.2)
            x = center_x + r * math.cos(theta)
            y = center_y + r * math.sin(theta)
            nodes[i] = (
                round(max(0, min(width, x)), 2),
                round(max(0, min(height, y)), 2),
            )

    elif shape == "river":
        # S-curve / sine river path with branching
        for i in range(num_rooms):
            progress = random.uniform(0.1, 0.9)
            x = progress * width
            # Sine wave center line with distance offset
            sine_y = center_y + math.sin(progress * math.pi * 3) * (
                height * 0.25
            )
            y = sine_y + random.gauss(0, height * 0.08)
            nodes[i] = (
                round(max(0, min(width, x)), 2),
                round(max(0, min(height, y)), 2),
            )

    else:  # 'organic' / default
        # Jittered uniform grid scatter to avoid node clustering
        cols = int(math.sqrt(num_rooms))
        rows = math.ceil(num_rooms / cols)
        cell_w, cell_h = width / cols, height / rows
        i = 0
        for r in range(rows):
            for c in range(cols):
                if i >= num_rooms:
                    break
                x = (c + random.uniform(0.1, 0.9)) * cell_w
                y = (r + random.uniform(0.1, 0.9)) * cell_h
                nodes[i] = (round(x, 2), round(y, 2))
                i += 1

    # Helper function for Euclidean distance
    def dist(u, v):
        return math.hypot(nodes[u][0] - nodes[v][0], nodes[u][1] - nodes[v][1])

    # --- 2. GRAPH CONSTRUCTION (MST + Extra Organic Loops) ---
    adjacency_list = {i: [] for i in range(num_rooms)}

    # Gather all candidate edges sorted by length
    all_edges = []
    for u in range(num_rooms):
        for v in range(u + 1, num_rooms):
            all_edges.append((dist(u, v), u, v))
    all_edges.sort()

    # Kruskal's MST algorithm to guarantee 100% connectivity without cycles
    parent = list(range(num_rooms))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    mst_edges = set()
    for d, u, v in all_edges:
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            mst_edges.add((min(u, v), max(u, v)))
            adjacency_list[u].append((v, round(d, 2)))
            adjacency_list[v].append((u, round(d, 2)))

    # Add back a subset of short non-MST edges to create loops & shortcuts
    for d, u, v in all_edges:
        edge = (min(u, v), max(u, v))
        if edge not in mst_edges:
            # Only connect nearby rooms to keep tunnels realistic
            if d < (min(width, height) / math.sqrt(num_rooms)) * 2.5:
                if random.random() < extra_connection_chance:
                    adjacency_list[u].append((v, round(d, 2)))
                    adjacency_list[v].append((u, round(d, 2)))

    return {"nodes": nodes, "adjacency_list": adjacency_list}


def generate_cave_with_controlled_loops(
    num_rooms: int = 12,
    bounds: tuple[float, float] = (100.0, 100.0),
    loop_percentage: float = 0.15,
) -> dict:
    """
    Generates a spatial cave graph using Delaunay Triangulation + MST with controlled loops.

    :param num_rooms: Total number of cave rooms (N).
    :param bounds: (width, height) boundary for placing rooms.
    :param loop_percentage: Fraction (0.0 to 1.0) of extra candidate tunnels to keep for loops.
    :return: Dict containing 'nodes' coordinates and 'edges' list of (u, v, dist).
    """
    # 1. Spawn random 2D room coordinates
    nodes = {
        i: (
            round(random.uniform(10, bounds[0] - 10), 2),
            round(random.uniform(10, bounds[1] - 10), 2),
        )
        for i in range(num_rooms)
    }

    points = [nodes[i] for i in range(num_rooms)]

    def dist(u, v):
        return math.hypot(nodes[u][0] - nodes[v][0], nodes[u][1] - nodes[v][1])

    # 2. Delaunay Triangulation (Generates non-intersecting candidate connections)
    tri = Delaunay(points)
    candidate_edges = set()

    for simplex in tri.simplices:
        # Each simplex is a triangle of 3 node indices
        for i in range(3):
            u, v = simplex[i], simplex[(i + 1) % 3]
            edge = (min(u, v), max(u, v))
            candidate_edges.add(edge)

    # 3. Minimum Spanning Tree (Prim's Algorithm) to guarantee connectivity with 0 loops
    mst_edges = set()
    visited = {0}
    pq = []

    def push_edges(u):
        for edge in candidate_edges:
            if u in edge:
                v = edge[1] if edge[0] == u else edge[0]
                if v not in visited:
                    heapq.heappush(pq, (dist(u, v), u, v))

    push_edges(0)

    while pq and len(visited) < num_rooms:
        d, u, v = heapq.heappop(pq)
        if v not in visited:
            visited.add(v)
            edge = (min(u, v), max(u, v))
            mst_edges.add(edge)
            push_edges(v)

    # 4. Controlled Loops: Add back a small % of discarded candidate edges
    discarded_edges = list(candidate_edges - mst_edges)
    random.shuffle(discarded_edges)

    num_loops_to_add = int(len(discarded_edges) * loop_percentage)
    loop_edges = set(discarded_edges[:num_loops_to_add])

    final_edges = mst_edges | loop_edges

    # Format adjacency list
    adj_list = {i: [] for i in range(num_rooms)}
    for u, v in final_edges:
        d = round(dist(u, v), 2)
        adj_list[u].append((v, d))
        adj_list[v].append((u, d))

    return {
        "nodes": nodes,
        "edges": list(final_edges),
        "adjacency_list": adj_list,
        "num_loops_added": num_loops_to_add,
    }


# # Example Usage:
# cave = generate_cave_graph(
#     num_rooms=4, bounds=(100, 100), max_tunnel_length=35
# )

# print("Room Coordinates (ID: (X, Y)):")
# for room_id, coords in cave["nodes"].items():
#     print(f"  Room {room_id}: {coords}")

# print("\nTunnels (Room ID -> Connected Room ID, Tunnel Length):")
# for room_id, neighbors in cave["adjacency_list"].items():
#     print(f"  Room {room_id}: {neighbors}")
