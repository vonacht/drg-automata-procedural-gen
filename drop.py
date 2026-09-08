import numpy as np


def balanced_line(disks, angle):
    """
    Returns a line that splits the disk centers into two equal groups.

    Parameters
    ----------
    disks : dict
        {"id": {"center": (x,y), "radius": r}, ...}
    angle : float
        Line angle in radians.

    Returns
    -------
    point : ndarray
        A point on the line.
    direction : ndarray
        Direction vector of the line.
    """

    centers = np.array([d["center"] for d in disks])

    # Direction of the line
    direction = np.array([np.cos(angle), np.sin(angle)])

    # Normal vector
    normal = np.array([-direction[1], direction[0]])

    # Project onto normal
    proj = centers @ normal

    # Median position
    proj_sorted = np.sort(proj)
    n = len(proj)

    if n % 2 == 0:
        offset = 0.5 * (proj_sorted[n // 2 - 1] + proj_sorted[n // 2])
    else:
        offset = proj_sorted[n // 2]

    point = offset * normal

    return point, direction


def split_disks(disks, point, direction):
    normal = np.array([-direction[1], direction[0]])

    left = []
    right = []

    for d in disks:
        c = np.array(d["center"])
        if np.dot(c - point, normal) >= 0:
            left.append(tuple(c))
        else:
            right.append(tuple(c))

    return left, right


def return_half_disks(disks):
    disks = [
        {"center": (d["center"][0], d["center"][1]), "radius": d["radius"]}
        for d in disks
    ]
    point, direction = balanced_line(disks, np.pi / 6)
    left, right = split_disks(disks, point, direction)
    return left, right
