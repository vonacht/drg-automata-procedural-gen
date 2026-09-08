from opensimplex import OpenSimplex
import numpy as np
import random


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


def generate_cliff_terrain(tilemap, scale, height_scale, octaves=4, seed=42):
    noise = OpenSimplex(seed)
    height, width = tilemap.shape
    terrain = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            if tilemap[y, x] == 1:
                # 1. Basic Simplex Noise (-1.0 to 1.0)
                nx = x / scale
                ny = y / scale
                raw_noise = noise.noise2(nx, ny)

                # 2. Create Ridges: Use the absolute value and invert it
                # This turns the "valleys" into sharp "ridges"
                ridged = 1.0 - abs(raw_noise)

                # 3. Add Contrast: Sharpen the cliffs using a power function
                # Values close to 1 stay high, values below 1 drop off quickly
                cliff_profile = ridged**3

                # 4. Optional: Terracing (The "Step" effect)
                # This rounds the height to the nearest 'step' to create plateaus
                steps = 10
                terraced = np.floor(cliff_profile * steps) / steps

                terrain[x][y] = terraced * height_scale

    return terrain


def apply_heightmap_to_spheres(disks):

    noise = OpenSimplex(seed=random.randint(0, 10000))

    HEIGHT_SCALE = 700
    NOISE_SCALE = 0.0002

    for disk in disks:
        x, y, z = disk["center"][0], disk["center"][1], disk["center"][2]
        z += noise.noise2(x * NOISE_SCALE, y * NOISE_SCALE) * HEIGHT_SCALE
        disk["center"] = (x, y, z)
    return disks
