import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
import networkx as nx
import matplotlib.pyplot as plt


def evenly_spaced_skeleton_points(
    letter, font_path, spacing=8, image_size=300, font_size=240
):
    # Render letter
    img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (image_size - w) // 2
    y = (image_size - h) // 2

    draw.text((x, y), letter, fill=255, font=font)

    # Binary image
    arr = np.array(img) > 0

    # Skeletonize
    skel = skeletonize(arr)

    # Coordinates
    ys, xs = np.where(skel)
    pts = np.column_stack([xs, ys])

    # Shuffle for unbiased selection
    np.random.shuffle(pts)

    selected = []

    tree_points = []

    for p in pts:
        if len(tree_points) == 0:
            selected.append(p)
            tree_points.append(p)
            continue

        tree = cKDTree(tree_points)

        dist, _ = tree.query(p, k=1)

        if dist >= spacing:
            selected.append(p)
            tree_points.append(p)

    selected = np.array(selected)

    return selected


def build_knn_graph(points, k=3):
    tree = cKDTree(points)

    G = nx.Graph()

    for i, p in enumerate(points):
        G.add_node(i, pos=tuple(p))

        dists, idxs = tree.query(p, k=k + 1)

        for j in idxs[1:]:
            dist = np.linalg.norm(points[i] - points[j])
            G.add_edge(i, j, weight=dist)

    return G


def traversal_path_with_backtracking(G, start_node=0):
    """
    Returns a traversal path through the graph
    including backtracking steps.

    Parameters
    ----------
    G : networkx.Graph
    start_node : int

    Returns
    -------
    path : list[int]
        Ordered node visitation sequence
    """

    visited = set()
    path = []

    def dfs(u, parent=None):
        visited.add(u)
        path.append(u)

        for v in G.neighbors(u):
            if v not in visited:
                dfs(v, u)

                # backtrack to current node
                path.append(u)

    dfs(start_node)

    return path


def connect_components(G, points):
    G = G.copy()

    while not nx.is_connected(G):
        components = list(nx.connected_components(G))

        best = None
        best_dist = np.inf

        # Find the closest pair of nodes in different components
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                for u in components[i]:
                    for v in components[j]:
                        d = np.linalg.norm(points[u] - points[v])
                        if d < best_dist:
                            best_dist = d
                            best = (u, v)

        u, v = best
        G.add_edge(u, v, weight=best_dist)

    return G


def plot_points(points, *, connect=False, equal_aspect=True):
    """
    Plot a list of NumPy arrays, each of shape (2,).

    Example:
        points = [np.array([1, 2]), np.array([3, 4]), np.array([5, 1])]
    """
    points = np.asarray(points)  # shape (N, 2)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected a list/array of shape (N, 2).")

    plt.figure(figsize=(6, 6))

    if connect:
        plt.plot(points[:, 0], points[:, 1], "-o")
    else:
        plt.scatter(points[:, 0], points[:, 1])

    if equal_aspect:
        plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def assemble_letter(letter):
    l = evenly_spaced_skeleton_points(
        letter,
        "usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        spacing=20,
    )
    G = build_knn_graph(l)
    G = connect_components(G, l)
    return l, traversal_path_with_backtracking(G)


def assemble_letter_math(letter):
    l = evenly_spaced_skeleton_points(
        letter,
        "usr/share/fonts/truetype/dejavu/NotoSansMath-Regular.ttf",
        spacing=20,
    )
    G = build_knn_graph(l, k=2)
    G = connect_components(G, l)
    return l, traversal_path_with_backtracking(G)


def assemble_letter_devanagari(letter):
    l = evenly_spaced_skeleton_points(
        letter,
        "usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        spacing=20,
    )

    G = build_knn_graph(l)
    G = connect_components(G, l)
    return l, traversal_path_with_backtracking(G)
