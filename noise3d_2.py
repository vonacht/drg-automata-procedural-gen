import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from scipy.ndimage import maximum_filter
from scipy.ndimage import distance_transform_edt
import random

noise = OpenSimplex(seed=random.randrange(1, 10000))


class DomainWarpedFBM:
    def __init__(
        self,
        seed,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
    ):
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

        # Separate noise generators
        self.main = OpenSimplex(seed)
        self.warp_x = OpenSimplex(seed + 1000)
        self.warp_y = OpenSimplex(seed + 2000)
        self.warp_z = OpenSimplex(seed + 3000)

    def fbm(self, x, y, z):
        value = 0.0

        amplitude = 1.0
        frequency = 1.0

        amplitude_sum = 0.0

        for _ in range(self.octaves):
            value += (
                self.main.noise3(
                    x * frequency,
                    y * frequency,
                    z * frequency,
                )
                * amplitude
            )

            amplitude_sum += amplitude

            amplitude *= self.persistence
            frequency *= self.lacunarity

        return value / amplitude_sum

    def sample(
        self,
        x,
        y,
        z,
        scale=0.02,
        warp_scale=0.01,
        warp_strength=15.0,
    ):
        # Generate warp offsets
        dx = (
            self.warp_x.noise3(
                x * warp_scale,
                y * warp_scale,
                z * warp_scale,
            )
            * warp_strength
        )

        dy = (
            self.warp_y.noise3(
                x * warp_scale,
                y * warp_scale,
                z * warp_scale,
            )
            * warp_strength
        )

        dz = (
            self.warp_z.noise3(
                x * warp_scale,
                y * warp_scale,
                z * warp_scale,
            )
            * warp_strength
        )

        return self.fbm(
            (x + dx) * scale,
            (y + dy) * scale,
            (z + dz) * scale,
        )


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


def create_noise_room(width=20, height=20, depth=8):

    WIDTH = width
    HEIGHT = height
    DEPTH = depth

    while True:
        voxels = np.zeros((WIDTH, HEIGHT, DEPTH), dtype=bool)
        noise = DomainWarpedFBM(seed=random.randrange(1, 10000))

        for x in range(WIDTH):
            for y in range(HEIGHT):
                for z in range(DEPTH):
                    n = noise.sample(
                        x,
                        y,
                        z,
                        scale=0.1,
                        warp_scale=0.01,
                        warp_strength=20.0,
                    )
                    voxels[x, y, z] = n > 0.1

        # True = air
        # False = rock
        dist = distance_transform_edt(voxels == True)
        centers = find_local_maxima(dist)
        spheres = [(tuple(c), dist[tuple(c)]) for c in centers]
        if spheres:
            break
    # print(spheres[0])
    # spheres = filter_overlapping_spheres(
    #     [(c[0][0], c[0][1], c[0][1], c[1]) for c in spheres],
    #     overlap_factor=0.05,
    # )
    # spheres = [((c[0], c[1], c[2]), c[3]) for c in spheres]
    path = nearest_path(spheres)

    return spheres, path


def create_anisotropic_noise_room(width=20, height=20, depth=8):

    WIDTH = width
    HEIGHT = height
    DEPTH = depth

    x_coeff = 1
    y_coeff = 1
    z_coeff = 1

    while True:
        voxels = np.zeros((WIDTH, HEIGHT, DEPTH), dtype=bool)
        noise = DomainWarpedFBM(seed=random.randrange(1, 10000))

        for x in range(WIDTH):
            for y in range(HEIGHT):
                for z in range(DEPTH):
                    n = noise.sample(
                        x_coeff * x,
                        y_coeff * y,
                        z_coeff * z,
                        scale=0.1,
                        warp_scale=0.01,
                        warp_strength=20.0,
                    )
                    voxels[x, y, z] = n > 0.5

        # True = air
        # False = rock
        dist = distance_transform_edt(voxels == True)
        centers = find_local_maxima(dist)
        spheres = [(tuple(c), dist[tuple(c)]) for c in centers]
        if spheres:
            break
    # print(spheres[0])
    # spheres = filter_overlapping_spheres(
    #     [(c[0][0], c[0][1], c[0][1], c[1]) for c in spheres],
    #     overlap_factor=0.05,
    # )
    # spheres = [((c[0], c[1], c[2]), c[3]) for c in spheres]
    path = nearest_path(spheres)

    return spheres, path
