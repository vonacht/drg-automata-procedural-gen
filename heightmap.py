from opensimplex import OpenSimplex
import numpy as np


def generate_heightmap(tilemap, scale=0.15, height_scale=5, seed=42):
    """
    Returns a randomized height map based on the input tilemap, using simplex noise.
    """
    noise = OpenSimplex(seed)
    h, w = tilemap.shape
    heightmap = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            if tilemap[y, x] == 1:
                n = noise.noise2(x * scale, y * scale)
                n = (n + 1) / 2  # [-1,1] → [0,1]
                heightmap[y, x] = n * height_scale

    return heightmap
