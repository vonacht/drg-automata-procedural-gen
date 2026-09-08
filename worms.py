import numpy as np
import random
from numba import njit
from dataclasses import dataclass
from scipy.ndimage import maximum_filter
from scipy.ndimage import distance_transform_edt


@njit
def carve_sphere(grid, center_x, center_y, center_z, radius):
    width, height, depth = grid.shape

    r = int(np.ceil(radius))
    radius_sq = radius * radius

    for x in range(center_x - r, center_x + r + 1):
        if x < 0 or x >= width:
            continue

        for y in range(center_y - r, center_y + r + 1):
            if y < 0 or y >= height:
                continue

            for z in range(center_z - r, center_z + r + 1):
                if z < 0 or z >= depth:
                    continue

                dx = x - center_x
                dy = y - center_y
                dz = z - center_z

                if dx * dx + dy * dy + dz * dz <= radius_sq:
                    grid[x, y, z] = 0


@dataclass
class Worm:
    position: np.ndarray
    direction: np.ndarray
    radius: float
    life: int


class CaveGenerator3D:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

        # 1 = solid rock
        # 0 = empty cave
        self.grid = np.ones((width, height, depth), dtype=np.uint8)

    def generate(
        self,
        num_worms=20,
        min_radius=1.5,
        max_radius=5.0,
        min_life=25,
        max_life=50,
        branch_chance=0,
        turn_strength=0.25,
    ):
        worms = []

        # Spawn initial worms
        for _ in range(num_worms):

            pos = np.array(
                [
                    random.randint(10, self.width - 10),
                    random.randint(10, self.height - 10),
                    random.randint(10, self.depth - 10),
                ],
                dtype=float,
            )

            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction)

            worms.append(
                Worm(
                    position=pos,
                    direction=direction,
                    radius=random.uniform(min_radius, max_radius),
                    life=random.randint(min_life, max_life),
                )
            )

        while worms:
            worm = worms.pop()

            for _ in range(worm.life):

                # Carve tunnel
                cx, cy, cz = worm.position.astype(np.int32)
                carve_sphere(self.grid, cx, cy, cz, worm.radius)

                # Smooth random steering
                steer = np.random.normal(scale=turn_strength, size=3)

                worm.direction += steer
                worm.direction /= np.linalg.norm(worm.direction)

                # Move forward
                worm.position += worm.direction

                x, y, z = worm.position

                # Stop if out of bounds
                if (
                    x < 2
                    or x >= self.width - 2
                    or y < 2
                    or y >= self.height - 2
                    or z < 2
                    or z >= self.depth - 2
                ):
                    break

                # Slight radius variation
                worm.radius += random.uniform(-0.05, 0.05)
                worm.radius = max(min_radius, min(max_radius, worm.radius))

                # Branching
                if random.random() < branch_chance:

                    new_dir = worm.direction + np.random.normal(
                        scale=0.75, size=3
                    )
                    new_dir /= np.linalg.norm(new_dir)

                    worms.append(
                        Worm(
                            position=worm.position.copy(),
                            direction=new_dir,
                            radius=worm.radius * random.uniform(0.7, 1.0),
                            life=int(worm.life * random.uniform(0.3, 0.8)),
                        )
                    )

        return self.grid


def find_local_maxima(dist, min_radius=2):
    max_filt = maximum_filter(dist, size=3)
    peaks = (dist == max_filt) & (dist >= min_radius)
    return np.argwhere(peaks)


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


import matplotlib.pyplot as plt


def plot_cave(grid, sample_rate=1):
    """
    Visualize carved voxels (0 = cave).

    sample_rate:
        1 = show all voxels
        2 = show every other voxel
        4 = show 1/16 of voxels
    """

    cave = np.argwhere(grid == 0)

    if sample_rate > 1:
        cave = cave[::sample_rate]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(cave[:, 0], cave[:, 1], cave[:, 2], s=1, alpha=0.4)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Procedural Cave")

    plt.show()


def generate_3d_worm_cave(num_worms):

    cave = CaveGenerator3D(width=128, height=64, depth=128)

    voxels = cave.generate(
        num_worms=3, min_radius=15, max_radius=20, branch_chance=0
    )
    dist = distance_transform_edt(voxels == 0)
    centers = find_local_maxima(dist)
    spheres = [(tuple(c), dist[tuple(c)]) for c in centers]
    path = nearest_path(spheres)

    return spheres, path


def generate_3d_worm_cave_parameterized(num_worms):

    cave = CaveGenerator3D(width=80, height=80, depth=80)

    voxels = cave.generate(
        num_worms=num_worms, min_radius=15, max_radius=20, branch_chance=0
    )
    dist = distance_transform_edt(voxels == 0)
    centers = find_local_maxima(dist)
    spheres = [(tuple(c), dist[tuple(c)]) for c in centers]
    path = nearest_path(spheres)

    return spheres, path
