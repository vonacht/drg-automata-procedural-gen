import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.stats import qmc
import networkx as nx


def generate_complete_road_traversal(n_points, radius, width=100, height=100):
    # 1. Poisson Disk Sampling for clean node distribution
    d = radius / max(width, height)
    engine = qmc.PoissonDisk(d=2, radius=d)
    sample = engine.random(n=n_points)
    points = qmc.scale(sample, [0, 0], [width, height])

    # 2. Build the Delaunay Graph
    tri = Delaunay(points)
    G = nx.Graph()
    for s in tri.simplices:
        G.add_edges_from([(s[0], s[1]), (s[1], s[2]), (s[2], s[0])])

    # 3. "Eulerize" the graph
    # This finds the shortest way to add duplicate edges so every
    # node has an even degree, making a continuous circuit possible.
    G_euler = nx.eulerize(G)

    # 4. Find the Eulerian Circuit
    # This returns an iterable of (u, v) tuples covering every edge
    circuit = list(nx.eulerian_circuit(G_euler, source=0))

    # Flatten the circuit into a list of points
    path_nodes = [circuit[0][0]]
    for u, v in circuit:
        path_nodes.append(v)

    return points, G, path_nodes


# # --- Execution ---
# radius_val = 18
# points, original_graph, final_path = generate_complete_road_traversal(
#     radius_val
# )

# # --- Visualization ---
# plt.figure(figsize=(10, 8))
# pos = {i: points[i] for i in range(len(points))}

# # Draw original roads (the "Map")
# nx.draw_networkx_edges(
#     original_graph, pos, alpha=0.2, edge_color="black", label="Road Network"
# )
# nx.draw_networkx_nodes(original_graph, pos, node_size=40, node_color="black")

# # Draw the traversal path (the "Route")
# # We'll draw arrows to show the direction of travel
# for i in range(len(final_path) - 1):
#     u, v = final_path[i], final_path[i + 1]
#     # Shift slightly so overlapping lines (backtracks) are visible
#     color = plt.cm.cool(i / len(final_path))
#     plt.annotate(
#         "",
#         xy=pos[v],
#         xytext=pos[u],
#         arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.6),
#     )

# plt.title("Continuous Path Covering 100% of Edges (with Backtracking)")
# plt.show()

# print(f"Unique roads: {original_graph.number_of_edges()}")
# print(f"Total steps taken (including backtracks): {len(final_path) - 1}")
