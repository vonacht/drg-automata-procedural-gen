import numpy as np
from scipy.ndimage import gaussian_filter
from noise3d import distance_transform_edt, nearest_path, find_local_maxima


def generate_blur_room(seed, n, ratio, open_fraction, s):

    bitmask = generate_blur(seed, n, ratio, open_fraction, s)
    # print(bitmask)

    dist = distance_transform_edt(bitmask == True)
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


def generate_blur(seed, n, ratio, open_fraction, s):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n))  # or (n, n) for 2D

    field = gaussian_filter(noise, s) - gaussian_filter(noise, s * ratio)
    field /= field.std()  # normalize so thresholds are comparable
    t = np.quantile(
        field, 1 - open_fraction
    )  # pick the threshold by percentile
    open_mask = field > t
    return open_mask
