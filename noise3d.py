import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import distance_transform_edt
import opensimplex
import random


def fbm_noise3(x, y, z, octaves=2, persistence=0.5, lacunarity=2.0):
    """
    Generates Fractal Brownian Motion (fBm) noise by layering
    multiple octaves of OpenSimplex noise.
    """
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0  # Used to normalize the result back to a -1.0 to 1.0 range

    for _ in range(octaves):
        # opensimplex.noise3 returns a value between -1 and 1
        total += (
            opensimplex.noise3(x * frequency, y * frequency, z * frequency)
            * amplitude
        )
        max_value += amplitude

        amplitude *= (
            persistence  # Each layer contributes less to the total shape
        )
        frequency *= lacunarity  # Each layer packs in more details

    return total / max_value


def generate_cave_chunk(width, height, depth, scale=50.0, threshold=0.1):
    """
    Generates a 3D boolean array representing a cave system.
    True = Solid Rock
    False = Empty Air
    """
    # Seed the noise generator. Change this number to generate a completely different cave!
    opensimplex.seed(random.randrange(1, 10000))

    # Initialize a 3D numpy array
    cave_data = np.zeros((width, height, depth), dtype=bool)

    for x in range(width):
        for y in range(height):
            for z in range(depth):
                # 1. Generate the custom fBm OpenSimplex noise
                noise_val = fbm_noise3(
                    x / scale, y / scale, z / scale, octaves=3
                )

                # 2. Apply a Vertical Bias (Floor & Ceiling)
                # We want the top and bottom to be solid rock, and the middle hollow.
                center_y = height / 2.0
                distance_from_center = abs(y - center_y) / center_y

                # Add the distance gradient to the noise.
                density = noise_val + distance_from_center

                # 3. Apply the threshold
                if density > threshold:
                    cave_data[x, y, z] = True  # Solid
                else:
                    cave_data[x, y, z] = False  # Air

    return cave_data


def print_cave_slice(cave_data, z_index):
    """
    Prints a 2D cross-section of the 3D cave data to the console.
    """
    width, height, _ = cave_data.shape

    print(f"--- Vertical Slice at Z = {z_index} ---")
    # Iterate backwards through Y so Y=0 is printed at the bottom
    for y in range(height - 1, -1, -1):
        row_chars = []
        for x in range(width):
            # Use '██' for solid rock and '  ' for empty air
            if cave_data[x, y, z_index]:
                row_chars.append("██")
            else:
                row_chars.append("  ")
        print("".join(row_chars))
    print("-" * (width * 2))


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


def create_noise_room(cond, cw=40, ch=40, cd=20):

    from more_itertools import flatten

    CHUNK_WIDTH = cw
    CHUNK_HEIGHT = ch
    CHUNK_DEPTH = cd

    grid = generate_cave_chunk(
        width=CHUNK_WIDTH,
        height=CHUNK_HEIGHT,
        depth=CHUNK_DEPTH,
        scale=30.0,  # Lower scale = tighter, more frequent caves
        threshold=0.4,  # Higher threshold = more empty space, less rock
    )
    dist = distance_transform_edt(grid == cond)
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
