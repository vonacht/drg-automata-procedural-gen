import numpy as np
import random


def generate_cave(
    size=(64, 64, 64),
    cell_size=12.0,
    threshold=0.35,
    seed=None,
):
    """
    Generate a random 3D cave using Worley noise.

    Parameters
    ----------
    size : tuple[int, int, int]
        Number of voxels in (x, y, z).

    cell_size : float
        Approximate distance between Worley feature points.
        Larger = larger cave structures.

    threshold : float
        Controls how much of the volume becomes cave.
        Lower values generally produce larger/more connected caves.

    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    density : np.ndarray
        3D float array in [0, 1].
        Values below `threshold` are cave/empty space.
        Values above it are solid rock.
    """

    nx, ny, nz = size

    # Coordinates of every voxel.
    x, y, z = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij",
    )

    # Convert voxel coordinates into Worley-cell coordinates.
    px = x / cell_size
    py = y / cell_size
    pz = z / cell_size

    # Cell containing each voxel.
    cx = np.floor(px).astype(int)
    cy = np.floor(py).astype(int)
    cz = np.floor(pz).astype(int)

    # Distance to the closest feature point.
    f1 = np.full(size, np.inf, dtype=np.float32)

    # Check neighboring cells.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):

                cell_x = cx + dx
                cell_y = cy + dy
                cell_z = cz + dz

                # Deterministically generate one feature point per cell.
                #
                # The integer cell coordinates are hashed so we don't
                # have to store all feature points.
                h = (
                    cell_x * 374761393
                    + cell_y * 668265263
                    + cell_z * 2147483647
                    + (seed or 0) * 1274126177
                )

                h = h.astype(np.int64)

                # Hash -> pseudo-random values in [0, 1).
                h1 = np.sin(h * 12.9898) * 43758.5453
                h2 = np.sin(h * 78.233) * 43758.5453
                h3 = np.sin(h * 39.425) * 43758.5453

                fx = h1 - np.floor(h1)
                fy = h2 - np.floor(h2)
                fz = h3 - np.floor(h3)

                # Feature point position.
                feature_x = cell_x + fx
                feature_y = cell_y + fy
                feature_z = cell_z + fz

                # Distance from voxel to feature point.
                d = np.sqrt(
                    (px - feature_x) ** 2
                    + (py - feature_y) ** 2
                    + (pz - feature_z) ** 2
                )

                f1 = np.minimum(f1, d)

    # Normalize approximately to [0, 1].
    density = np.clip(f1 / 0.866, 0.0, 1.0)

    return density


def generate_worley_cave():

    from noise3d import distance_transform_edt, find_local_maxima, nearest_path

    rng = random.randint(0, 10000)
    cave = generate_cave(
        size=(10, 10, 2),
        # size=(2, 8, 10),
        cell_size=2,
        threshold=0.35,
        seed=rng,
    )

    # Convert density to a binary voxel cave.
    # True = solid, False = empty.
    # solid = cave > 0.35
    dist = distance_transform_edt(cave > 0.35)
    centers = find_local_maxima(dist)
    spheres = [(tuple(c), dist[tuple(c)]) for c in centers]
    # print(spheres[0])
    # spheres = filter_overlapping_spheres(
    #     [(c[0][0], c[0][1], c[0][1], c[1]) for c in spheres],
    #     overlap_factor=0.05,
    # )
    # spheres = [((c[0], c[1], c[2]), c[3]) for c in spheres]
    path = nearest_path(spheres)

    return spheres, path
