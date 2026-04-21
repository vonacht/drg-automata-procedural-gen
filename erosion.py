import numpy as np
from opensimplex import OpenSimplex
import random

from skimage import measure


def count_neighbors(grid, x, y, z):
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if (
                    0 <= nx < grid.shape[0]
                    and 0 <= ny < grid.shape[1]
                    and 0 <= nz < grid.shape[2]
                ):
                    count += grid[nx, ny, nz]
    return count


def smooth(grid, iterations=3):
    for _ in range(iterations):
        new_grid = grid.copy()
        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                for z in range(1, grid.shape[2] - 1):
                    neighbors = count_neighbors(grid, x, y, z)
                    if neighbors > 13:
                        new_grid[x, y, z] = 1
                    elif neighbors < 10:
                        new_grid[x, y, z] = 0
        grid = new_grid
    return grid


def thermal_erosion(grid, iterations=5):
    for _ in range(iterations):
        new_grid = grid.copy()
        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                for z in range(1, grid.shape[2] - 1):
                    if grid[x, y, z] == 1:  # rock
                        empty_neighbors = 0
                        for dx, dy, dz in [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ]:
                            if grid[x + dx, y + dy, z + dz] == 0:
                                empty_neighbors += 1

                        if empty_neighbors >= 4:
                            new_grid[x, y, z] = 0  # erode
        grid = new_grid
    return grid


def collapse(grid):
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(1, grid.shape[2]):
                if grid[x, y, z] == 1 and grid[x, y, z - 1] == 0:
                    grid[x, y, z] = 0
    return grid


# import matplotlib.pyplot as plt

# verts, faces, normals, values = measure.marching_cubes(grid, level=0.5)


# def export_to_obj(filename, verts, faces):
#     with open(filename, "w") as f:
#         # Write vertices
#         for v in verts:
#             f.write(f"v {v[0]} {v[1]} {v[2]}\n")

#         # Write faces (OBJ is 1-indexed)
#         for face in faces:
#             f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# export_to_obj("cave.obj", verts, faces)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")

# # True = filled voxel (rock)
# voxels = grid == 1

# ax.voxels(voxels, edgecolor="k")
# plt.show()


from scipy.ndimage import distance_transform_edt


# while True:
#     idx = np.unravel_index(np.argmax(dist), dist.shape)
#     r = dist[idx]

#     if r < 1:  # stop threshold
#         break

#     spheres.append((idx, r))

#     # remove covered region
#     zz, yy, xx = np.indices(dist.shape)
#     mask = (zz - idx[0]) ** 2 + (yy - idx[1]) ** 2 + (xx - idx[2]) ** 2 <= r**2

#     dist[mask] = 0


import math


def spheres_touch(s1, s2):
    (c1, r1), (c2, r2) = s1, s2
    d = math.dist(c1, c2)
    return d <= (r1 + r2)


def nearest_path(spheres):
    remaining = set(range(len(spheres)))
    path = []

    current = 0
    path.append(current)
    remaining.remove(current)

    while remaining:
        next_node = min(
            remaining,
            key=lambda j: np.linalg.norm(
                np.array(spheres[current][0]) - np.array(spheres[j][0])
            ),
        )
        path.append(next_node)
        remaining.remove(next_node)
        current = next_node

    return path


from scipy.ndimage import maximum_filter


def find_local_maxima(dist, min_radius=2):
    max_filt = maximum_filter(dist, size=3)
    peaks = (dist == max_filt) & (dist >= min_radius)
    return np.argwhere(peaks)


def plot_spheres_colored(spheres):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    radii = [r for _, r in spheres]
    r_min, r_max = min(radii), max(radii)

    for c, r in spheres:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)

        x = c[0] + r * np.outer(np.cos(u), np.sin(v))
        y = c[1] + r * np.outer(np.sin(u), np.sin(v))
        z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))

        color = plt.cm.viridis((r - r_min) / (r_max - r_min + 1e-8))
        ax.plot_surface(x, y, z, color=color, alpha=0.4)

    ax.set_box_aspect([1, 1, 1])
    plt.show()


def random_walk_3d_grid(steps=1000, size=(10, 10, 10)):
    """
    Generate a 3D random walk and return a voxel grid of visited cells.

    Parameters:
        steps (int): Number of steps
        size (tuple): Half-sizes (sx, sy, sz), grid spans [-sx, sx], etc.

    Returns:
        np.ndarray: 3D array with 1 (visited) and 0 (unvisited)
    """
    size = np.array(size, dtype=int)

    # Grid dimensions: from -size to +size inclusive
    dims = 2 * size + 1
    grid = np.zeros(dims, dtype=int)

    # Start at origin (convert to grid index)
    pos = np.array([0, 0, 0])
    idx = pos + size  # shift to positive indices
    grid[tuple(idx)] = 1

    for _ in range(steps - 1):
        step = np.random.choice([-1, 0, 1], size=3)
        new_pos = pos + step

        # Reflect at boundaries
        for j in range(3):
            if new_pos[j] > size[j]:
                new_pos[j] = size[j] - (new_pos[j] - size[j])
            elif new_pos[j] < -size[j]:
                new_pos[j] = -size[j] - (new_pos[j] + size[j])

        pos = new_pos
        idx = pos + size
        grid[tuple(idx)] = 1

    return grid


import numpy as np
import random
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import skeletonize
import plotly.graph_objects as go


# -----------------------------
# Random walk
# -----------------------------
def random_walk_3d(size=(40, 40, 40), steps=6000):
    grid = np.zeros(size, dtype=np.uint8)

    x, y, z = size[0] // 2, size[1] // 2, size[2] // 2
    grid[x, y, z] = 1

    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    method = random.choice(["flat", "gradient", "vertical"])
    match method:
        case "flat":
            weights = [1, 1, 1, 1, 0, 0]
        case "gradient":
            weights = [1, 1, 1, 1, 0.07, 0]
        case "vertical":
            weights = [1, 1, 1, 1, 0.3, 0]
    print(method)

    for _ in range(steps):
        dx, dy, dz = random.choices(directions, weights=weights, k=1)[0]
        nx, ny, nz = x + dx, y + dy, z + dz

        if 0 <= nx < size[0] and 0 <= ny < size[1] and 0 <= nz < size[2]:
            x, y, z = nx, ny, nz
            grid[x, y, z] = 1

    return grid


# -----------------------------
# Skeleton + spheres
# -----------------------------
def skeleton_spheres(volume, min_radius=0.0):
    distance = distance_transform_edt(volume)
    skeleton = skeletonize(volume)

    coords = np.argwhere(skeleton > 0)

    spheres = []
    for x, y, z in coords:
        r = distance[x, y, z]
        if r >= min_radius:
            spheres.append((x, y, z, r))

    return spheres


# -----------------------------
# Interactive Plot
# -----------------------------
def plot_spheres_interactive(spheres, grid_size, max_spheres=25):
    spheres = sorted(spheres, key=lambda s: -s[3])[:max_spheres]

    fig = go.Figure()

    for cx, cy, cz, r in spheres:

        # Sphere mesh
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)

        x = cx + r * np.outer(np.cos(u), np.sin(v))
        y = cy + r * np.outer(np.sin(u), np.sin(v))
        z = cz + r * np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.5, showscale=False))

    fig.update_layout(
        title="Interactive Maximal Spheres (Skeleton-Based)",
        scene=dict(
            xaxis=dict(range=[0, grid_size[0]]),
            yaxis=dict(range=[0, grid_size[1]]),
            zaxis=dict(range=[0, grid_size[2]]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


def plot_walk_and_spheres(volume, spheres, max_spheres=20):

    fig = go.Figure()

    # --- Plot Walk Voxels ---
    x, y, z = np.where(volume == 1)

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.15),
            name="Random Walk Volume",
        )
    )

    # --- Plot Spheres ---
    spheres = sorted(spheres, key=lambda s: -s[3])[:max_spheres]

    for cx, cy, cz, r in spheres:
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)

        xs = cx + r * np.outer(np.cos(u), np.sin(v))
        ys = cy + r * np.outer(np.sin(u), np.sin(v))
        zs = cz + r * np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                opacity=0.5,
                showscale=False,
                name=f"r={r:.2f}",
            )
        )

    fig.update_layout(
        title="3D Random Walk + Maximal Inscribed Spheres",
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


def filter_overlapping_spheres(spheres, overlap_factor=0.9):
    """
    Greedy removal of overlapping spheres.

    overlap_factor:
        1.0  = no overlap allowed
        0.9  = slight overlap allowed
        0.5  = more aggressive packing
    """

    # Sort largest first
    spheres = sorted(spheres, key=lambda s: -s[3])

    accepted = []

    for candidate in spheres:
        cx, cy, cz, r = candidate
        keep = True

        for ax, ay, az, ar in accepted:
            dist = np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2 + (cz - az) ** 2)

            # Check overlap condition
            if dist < overlap_factor * (r + ar):
                keep = False
                break

        if keep:
            accepted.append(candidate)

    return accepted


def erosion_room(size, steps, method="randomwalk"):

    size = (size, size, size // 2)
    scale = 0.08

    if method == "randomwalk":
        tilemap = random_walk_3d(size=size, steps=steps)
        tilemap = binary_dilation(tilemap, iterations=3).astype(np.uint8)
        spheres = skeleton_spheres(tilemap, min_radius=1.5)
        spheres = filter_overlapping_spheres(spheres, overlap_factor=0.5)
        spheres = [((c[0], c[1], c[2]), c[3]) for c in spheres]
    else:
        grid = np.zeros(size)
        noise = OpenSimplex(random.randrange(10000))

        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    n = noise.noise3(x * scale, y * scale, z * scale)
                    grid[x, y, z] = 1 if n > 0.1 else 0  # 1 = rock, 0 = air
        grid = smooth(grid)
        grid = thermal_erosion(grid)
        # grid: 1 = solid, 0 = empty
        grid = np.array(grid)

        # distance inside empty space
        dist = distance_transform_edt(grid == 0)

        spheres = []
        centers = find_local_maxima(dist)
        spheres = [(tuple(c), dist[tuple(c)]) for c in centers]
    path = nearest_path(spheres)

    return spheres, path
