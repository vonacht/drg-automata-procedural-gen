import random
import math
from pathlib import Path
import json
import copy
import logging
import string
import numpy as np
from tree_gen3 import generate, ROOMS, OPERATIONS, COMBINERS
from blur import generate_blur_room
from lsystem import generate_lsystem_room

# from pit_rooms import generate_pit_room
from graph_room import generate_cave_graph, generate_cave_graph_w_structure
from tubes import kaleidoscope_room
from lens_and_invert import (
    lens,
    invert_retry,
    complex_map_retry,
    attractor_flow_retry,
    calibrate,
    roll_interesting,
)
from tubes2 import lattice_room, polytope_room
from morph import morph_rooms, crossover, braid, seam
from character import generate_letter_room
from periodic_wrap import periodic_wrap
from bigpillars import generate_big_pillars
from entrances_and_exits import place_doors, MissingEntranceOrExit_new
from typing import List, Dict, Optional
from more_itertools import flatten
from echo_and_bypass import bypass, echo, negative_space, cave_in
from half_rotation import rotate_half
from worms import generate_3d_worm_cave_parameterized, generate_3d_worm_cave
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from heightmap import apply_heightmap_to_spheres
from main import (
    create_room,
    create_random_walk,
    run_automata,
    create_DLA_tilemap,
    shift_room,
    shift_room_2d,
    scale_spheres,
    points_2d_to_shape,
    find_intersection_or_closest,
    points_to_sphere,
    quadratic_bezier_3d,
    perpendicular_2d_normalized,
    rotate_spheres,
    rotation_matrix_axis_angle,
    RoomTooSmall,
    midpoint_displacement,
    return_half_disks,
)
from noise3d_2 import create_noise_room
from erosion import erosion_room
from alphabet import (
    assemble_letter,
    assemble_letter_math,
)
from json_builder import (
    generate_entrances,
    build_json_multiple_ffill,
    generate_exits_furthest,
    generate_PE_features,
)

from pillars_2 import generate_pillars


def union_volume(room, rng=random, samples=20000, links=True, chunk=32):
    """Monte Carlo volume of the room: the hemispheres plus, if links=True, the tubes
    the engine draws between consecutive points of each polyline."""
    groups = room[0]
    sph = [s for g in groups for s in g]
    C = np.array([s["center"] for s in sph], dtype=float)
    R = np.array([s["radius"] for s in sph], dtype=float)
    lo = np.array([(C[:, 0] - R).min(), (C[:, 1] - R).min(), C[:, 2].min()])
    hi = np.array(
        [(C[:, 0] + R).max(), (C[:, 1] + R).max(), (C[:, 2] + R).max()]
    )
    gen = np.random.default_rng(rng.randrange(2**32))
    P = lo + gen.random((samples, 3)) * (hi - lo)
    inside = np.zeros(samples, dtype=bool)

    for s in range(0, len(C), 64):  # the spheres themselves
        d = P[:, None, :] - C[None, s : s + 64]
        inside |= (
            ((d**2).sum(-1) <= R[None, s : s + 64] ** 2) & (d[..., 2] >= 0)
        ).any(axis=1)

    if links:  # the tubes between consecutive points
        pairs = [(a, b) for g in groups for a, b in zip(g, g[1:])]
        if pairs:
            A = np.array([a["center"] for a, _ in pairs], dtype=float)
            B = np.array([b["center"] for _, b in pairs], dtype=float)
            RA = np.array([a["radius"] for a, _ in pairs], dtype=float)
            RB = np.array([b["radius"] for _, b in pairs], dtype=float)
            AB = B - A
            L2 = np.maximum((AB**2).sum(1), 1e-9)
            for s in range(0, len(A), chunk):
                a, ab, l2 = (
                    A[s : s + chunk],
                    AB[s : s + chunk],
                    L2[s : s + chunk],
                )
                ap = P[:, None, :] - a[None]
                t = np.clip(
                    (ap * ab[None]).sum(-1) / l2[None], 0.0, 1.0
                )  # nearest point on the segment
                q = a[None] + t[..., None] * ab[None]
                rad = (
                    RA[s : s + chunk][None]
                    + t * (RB - RA)[s : s + chunk][None]
                )  # radius there
                ok = (np.linalg.norm(P[:, None, :] - q, axis=-1) <= rad) & (
                    P[:, None, 2] >= q[..., 2]
                )
                inside |= ok.any(axis=1)
    return float(np.prod(hi - lo) * inside.mean())


def scale_room(room, factor):
    """Uniformly scales positions and radii about the room's center."""
    groups, path = room
    C = np.array([s["center"] for g in groups for s in g], dtype=float)
    mid = (C.min(0) + C.max(0)) / 2
    out = [
        [
            {
                **s,
                "center": tuple(
                    float(x)
                    for x in mid + (np.array(s["center"]) - mid) * factor
                ),
                "radius": s["radius"] * factor,
            }
            for s in g
        ]
        for g in groups
    ]
    return out, [list(p) for p in path]


def normalize_volume(room, target, rng=random, clamp=(0.6, 1.8)):
    """Scales the room uniformly so its volume approaches `target`.
    clamp:   limits on the length scale (not the volume). A tight clamp means a room that
             is far off stays somewhat off, so it's better to fix its source.
    """
    V = union_volume(room, rng)
    if V <= 0:
        return room
    f = min(max((target / V) ** (1 / 3), clamp[0]), clamp[1])
    # print("scale_factor", f)
    return scale_room(room, f)


def parse(s, i=0):
    token = s[i]

    if token in ROOMS:
        match token:
            case "A":
                return ("Cellular Automaton", token), i + 1
            case "B":
                return ("Random Walk", token), i + 1
            case "C":
                return ("DLA", token), i + 1
            case "D":
                return ("Erosion", token), i + 1
            case "E":
                return ("Noise", token), i + 1
            case "P":
                return ("Character", token), i + 1
            case "R":
                return ("Tunnel", token), i + 1
            case "V":
                return ("Simple Erosion", token), i + 1
            case "U":
                return ("Simple Noise", token), i + 1
            case "W":
                return ("Worms", token), i + 1
            case "X":
                return ("Big Worms", token), i + 1
            case "Z":
                return ("Fancy Tunnel", token), i + 1
            case "a":
                return ("Worley Room", token), i + 1
            case "b":
                return ("Pit Room", token), i + 1
            case "c":
                return ("Big Worley Room", token), i + 1
            case "d":
                return ("Big Pit Room", token), i + 1
            case "e":
                return ("Small Graph Room", token), i + 1
            case "f":
                return ("Big Graph Room", token), i + 1
            case "j":
                return ("Combined Room", token), i + 1
            case "k":
                return ("L-System Room", token), i + 1
            case "m":
                return ("Atomic Room", token), i + 1
            case "n":
                return ("Kaleidoscope Room", token), i + 1

    if token in OPERATIONS:
        child, j = parse(s, i + 1)
        match token:
            case "F":
                return ("Flat Sweep", token, child), j
            case "L":
                return ("Taper", token, child), j
            case "M":
                return ("Mirror", token, child), j
            case "K":
                return ("Self Overlap", token, child), j
            case "H":
                return ("Twist", token, child), j
            case "I":
                return ("Rotation", token, child), j
            case "J":
                return ("Perlin Heightmap", token, child), j
            case "Q":
                return ("Vertical Stack", token, child), j
            case "S":
                return ("Vertical Drop", token, child), j
            case "G":
                return ("Nonlinear Scale", token, child), j
            case "h":
                return ("Barrel Distortion", token, child), j
            case "i":
                return ("Scatter Distortion", token, child), j
            case "l":
                return ("Pit Transform", token, child), j
            case "q":
                return ("Half Room Rotation", token, child), j
            case "r":
                return ("Lens", token, child), j
            case "s":
                return ("Inversion", token, child), j
            case "t":
                return ("Complex Map", token, child), j
            case "u":
                return ("Attractor Flow", token, child), j
            case "x":
                return ("Periodic Wrap", token, child), j
            case "y":
                return ("Echo", token, child), j
            case "w":
                return ("Cave-In", token, child), j
            case "N":
                return ("Negative Space", token, child), j
            case "z":
                return ("Do Nothing", token, child), j

    if token in COMBINERS:
        left, j = parse(s, i + 1)
        right, k = parse(s, j)
        match token:
            case "O":
                return ("Overlap Combine", token, left, right), k
            case "T":
                return ("Tunnel Combine", token, left, right), k
            case "Y":
                return ("Vertical Combine", token, left, right), k
            case "v":
                return ("Morph Combine", token, left, right), k
            case "o":
                return ("Seam Combine", token, left, right), k
            case "p":
                return ("Braid Combine", token, left, right), k
            case "g":
                return ("Crossover Combine", token, left, right), k

    raise ValueError(f"Unknown token: {token}")


def build(node):
    kind = node[1]

    if kind in ROOMS:
        room = generate_room(node[1])
        # print(room)
        # print("union volume", node[1], union_volume(room) / 1e6)
        scaled_room = normalize_volume(room, 12000 * 1e6)
        return scaled_room

    if kind in OPERATIONS:
        transform = node[1]
        room = build(node[2])
        return apply_transform(transform, room)

    if kind in COMBINERS:
        operation = node[1]
        left = build(node[2])
        right = build(node[3])
        return combine(operation, left, right)


# ----------------------------------------------------------
# ---- ROOM TRANSFORMS ------------------------------------
# ----------------------------------------------------------


def room_rotation(room):
    a2, a1 = room
    rotated_room = []
    direction_vector = np.array(a2[0][-1]["center"]) - np.array(
        a2[0][0]["center"]
    )
    axis = (0, 0, 0)
    while axis == (0, 0, 0):
        axis = (
            random.choice([0, 1]),
            random.choice([0, 1]),
            random.choices([0, 1], weights=[0.8, 0.2], k=1)[0],
        )
    theta = np.radians(random.uniform(30, 60))
    R = rotation_matrix_axis_angle(axis, theta)
    center_of_rotation = direction_vector / 2
    for cluster in a2:  # Rotation around a random axis in 3D:
        a_copy = rotate_spheres(cluster, center_of_rotation, R)
        rotated_room += [a_copy]
    return rotated_room, a1


def perlin_heightmap_transform(room):
    a2, a1 = room
    heightmapped_room = []
    for cluster in a2:
        heightmapped_room.append(apply_heightmap_to_spheres(cluster))
    return heightmapped_room, a1


def self_overlap(room):
    a2, a1 = room
    direction_vector = np.array(a2[0][-1]["center"]) - np.array(
        a2[0][0]["center"]
    )
    try:
        perpendicular_vector = perpendicular_2d_normalized(direction_vector)
    except ValueError:
        perpendicular_vector = (1, 0)
    depth_m = random.choices(["depth", "shift"], k=1, weights=[1, 0])[0]
    new_room = a2.copy()
    traversal_paths = a1.copy()
    if depth_m == "depth":
        for cluster in a2:
            a_copy = [
                {
                    "center": (
                        c["center"][0]
                        + perpendicular_vector[0]
                        + a2[0][0]["radius"] * random.uniform(0.7, 1.2),
                        c["center"][1]
                        - perpendicular_vector[1]
                        + a2[0][0]["radius"] * random.uniform(0.7, 1.2),
                        c["center"][2],
                    ),
                    "radius": c["radius"],
                    "floor_depth": -random.randrange(500, 1500),
                }
                for c in cluster
            ]
            new_room += [a_copy]
        traversal_paths += a1
    else:
        axis = (0, 0, 0)
        while axis == (0, 0, 0):
            axis = (
                random.choice([0, 1]),
                random.choice([0, 1]),
                random.choice([0, 1]),
            )
        theta = np.radians(random.uniform(50, 120))
        R = rotation_matrix_axis_angle(axis, theta)
        center_of_rotation = direction_vector / 2
        for cluster in a2:
            a_copy = rotate_spheres(cluster, center_of_rotation, R)
            new_room += [a_copy]
        traversal_paths += a1

    return new_room, traversal_paths


def room_half_drop(room):
    a2, a1 = room
    direction = random.choice([1, -1])
    dropped_room = []
    for cluster in a2:
        points_left, points_right = return_half_disks(cluster)
        points_2 = points_2d_to_shape(
            points_left,
            800,
            lambda x, y, z: (
                random.randrange(-1500, -1000)
                if direction == -1
                else random.randrange(1000, 1500)
            ),
        )
        points_right = points_2d_to_shape(points_right, 600, lambda x, y, z: 0)
        points_2.extend(points_right)
        dropped_room += [points_2]
    return dropped_room, a1


def curve_sweep_transform(room):

    def compute_curve(N):
        A = (0, 0)
        B = random.choice([(10, 5), (-10, -5)])

        curve = midpoint_displacement(A, B, depth=6, roughness=0.35)
        height_function = random.choice([-1, 1])
        curve = points_2d_to_shape(
            curve,
            2,
            lambda i, x, y: i * height_function * random.uniform(0.06, 0.08),
        )
        curve = scale_spheres(curve, 350)
        step = len(curve) // N
        sample = curve[::step]
        return sample

    curve_points = compute_curve(1)
    a2, a1 = room
    swept_room = a2.copy()
    paths = a1.copy()
    for point in curve_points[1:]:
        shifted = shift_room_multiple_2d(a2, point["center"])
        shaft = intersection_shaft(
            list(flatten(shifted)), list(flatten(swept_room))
        )
        if shaft:
            swept_room += [shaft]
            paths += [[0, 1]]
        swept_room += shifted
        paths += a1
    return swept_room, paths


def intersection_shaft(room_1, room_2):
    d = find_intersection_or_closest(room_1, room_2)
    if not d["intersects"]:
        shaft = [
            {
                "center": d["pair"][0]["center"],
                "radius": d["pair"][0]["radius"] * random.uniform(0.8, 1.5),
            },
            {
                "center": d["pair"][1]["center"],
                "radius": d["pair"][1]["radius"] * random.uniform(0.8, 1.5),
            },
        ]
        return shaft
    else:
        return None


def flat_sweep(room):

    random_shift_point = (
        random.randint(500, 2000),
        random.randint(500, 2000),
        random.randint(-500, 500),
    )
    a2, a1 = room
    swept_room = a2.copy()
    paths = a1.copy()
    shifted = shift_room_multiple_2d(a2, random_shift_point)
    shaft = intersection_shaft(
        list(flatten(shifted)), list(flatten(swept_room))
    )
    if shaft:
        swept_room += [shaft]
        paths += [[0, 1]]
    swept_room += shifted
    paths += a1
    return swept_room, paths


def vertical_stack(room):
    a2, a1 = room
    stacks = 2
    shifted_room = a2.copy()
    paths = a1.copy()
    for height in [400, 800, 1200][: stacks - 1]:
        shifted = shift_room_multiple(a2, (0, 0, height))
        shifted_room += shifted
        paths += a1
    return shifted_room, paths


def taper_transform(room):
    a2, a1 = room

    new_room = a2.copy()
    paths = a1.copy()
    center_point = get_room_center_multiple(a2)
    center_point = (center_point[0], center_point[1], center_point[2] - 500)

    for ii in range(1):
        scaled_room = scale_room_multiple(a2, 0.8)
        shifted_room = []
        for cluster in scaled_room:
            shifted_room += [
                [
                    {
                        "radius": r["radius"],
                        "center": (
                            r["center"][0],
                            r["center"][1],
                            r["center"][2] - 400,
                        ),
                    }
                    for r in cluster
                ]
            ]

        new_room += shifted_room
        paths += a1

    return new_room, paths


def mirror_room_transform(room):
    def mirror(spheres, plane="xy"):
        """
        Reflect sphere centers across a coordinate plane.

        plane:
            'xy' -> z -> -z
            'xz' -> y -> -y
            'yz' -> x -> -x
        """
        mirrored = []
        for cluster in spheres:
            m = []
            for sphere in cluster:
                x, y, z = sphere["center"]

                if plane == "xy":
                    center = (x, y, -z)
                elif plane == "xz":
                    center = (x, -y, z)
                elif plane == "yz":
                    center = (-x, y, z)
                else:
                    raise ValueError("plane must be 'xy', 'xz', or 'yz'")

                m.append({"center": center, "radius": sphere["radius"]})
            mirrored.append(m)

        return mirrored

    a2, a1 = room
    new_room = a2 + mirror(a2, random.choice(["xy", "xz", "yz"]))
    path = a1 + a1

    return new_room, path


def pit_transform(room):

    room, paths = room

    pit_room, pit_path = generate_pit_room(random.randint(2000, 3500))

    entrances = generate_entrances(
        list(flatten(room)),
        exit_interval=(200, 280),
        entrance_interval=(30, 140),
        return_type="Entrances",
    )

    if not entrances:
        entrances = generate_entrances(
            list(flatten(room)),
            exit_interval=(200, 280),
            entrance_interval=(0, 370),
            theta=5,
            return_type="Entrances",
        )

    exits = generate_exits_furthest(list(flatten(room)), entrances)
    chosen_entrance = random.choice(exits)
    # print(chosen_entrance)
    x, y, z = chosen_entrance[0], chosen_entrance[1], chosen_entrance[2]

    pit_room = shift_room(pit_room, (x, y, z))
    # print(room + pit_room)
    # print(paths + pit_path)

    return room + [pit_room], paths + [pit_path]


def _frame(room):
    """Flatten spheres and compute the principal-axis frame. Returns None if degenerate."""
    groups, _ = room
    spheres = [s for g in groups for s in g]
    if len(spheres) < 2:
        return None
    centers = np.array([s["center"] for s in spheres], dtype=float)
    centroid = centers.mean(axis=0)
    centered = centers - centroid
    vals, vecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    axis = vecs[:, np.argmax(vals)]
    axis = axis / np.linalg.norm(axis)
    proj = centered @ axis
    length = proj.max() - proj.min()
    if length == 0:
        return None
    return spheres, centroid, centered, axis, proj, length


def _rebuild(room, new_centers):
    groups, path = room
    result, i = [], 0
    for g in groups:
        new_group = []
        for s in g:
            new_group.append({**s, "center": tuple(new_centers[i])})
            i += 1
        result.append(new_group)
    return result, [list(p) for p in path]


def twist_transform(room, angle=None, profile=1.0, rng=random):
    """
    Twists spheres around the principal axis. Radii are untouched.
    angle:   total twist in radians (None = random in [-pi, pi])
    profile: 1.0 = linear twist along the axis (your original behavior)
             > 1 concentrates the twist near the ends, < 1 near the middle
    """
    if angle is None:
        angle = rng.uniform(-math.pi, math.pi)
    frame = _frame(room)
    if frame is None:
        return _rebuild(room, [s["center"] for g in room[0] for s in g])
    spheres, centroid, centered, axis, proj, length = frame

    # position along the axis in [-1, 1], with 0 at the middle of the shape
    x = 2 * (proj - proj.min()) / length - 1
    theta = (0.5 * angle * np.sign(x) * np.abs(x) ** max(profile, 1e-3))[
        :, None
    ]

    radial = (
        centered - proj[:, None] * axis
    )  # perpendicular part of each offset
    rotated = radial * np.cos(theta) + np.cross(axis, radial) * np.sin(theta)

    new_centers = centroid + proj[:, None] * axis + rotated
    return _rebuild(room, new_centers)


def random_scale_power():
    if np.random.random() < 0.5:
        # Compress center / expand ends
        power = np.random.normal(1.7, 0.25)
    else:
        # Expand center / compress ends
        power = np.random.normal(0.65, 0.10)

    return np.clip(power, 0.4, 2.2)


def _random_power(rng):
    # log-uniform, so "expand the center" (<1) and "expand the ends" (>1)
    # are equally likely, and never too close to 1 to notice
    mag = rng.uniform(math.log(1.3), math.log(2.5))
    return math.exp(mag * rng.choice((-1, 1)))


def nonlinear_scaling(
    room,
    rng=random,
    power=None,
    pivot=None,
    taper=None,
    bend=None,
    p_taper=0.5,
    p_bend=0.5,
):
    """
    Rescales sphere centers along the principal axis, with optional taper
    and bend. Radii are untouched, so volume is unchanged. Both ends of the
    room stay fixed along the axis.

    power: < 1 expands the center / compresses the ends, > 1 the opposite
           (None = random)
    pivot: position in (0, 1) the curve is symmetric about
           (None = random near the middle, 0.5 = symmetric)
    taper: perpendicular scale goes from 1 - taper at one end to 1 + taper
           at the other (None = random; 0 = off)
    bend:  arch height as a fraction of the room's length; positive arches
           up, negative sags (None = random; 0 = off)
    p_taper, p_bend: chance that a random taper / bend is applied at all
    """
    if power is None:
        power = _random_power(rng)
    if pivot is None:
        pivot = rng.uniform(0.35, 0.65)
    if taper is None:
        taper = (
            rng.uniform(0.2, 0.7) * rng.choice((-1, 1))
            if rng.random() < p_taper
            else 0.0
        )
    if bend is None:
        bend = (
            rng.uniform(0.05, 0.3) * rng.choice((-1, 1))
            if rng.random() < p_bend
            else 0.0
        )

    power = max(power, 1e-3)
    pivot = float(np.clip(pivot, 0.05, 0.95))
    taper = float(
        np.clip(taper, -0.95, 0.95)
    )  # keep the perpendicular scale positive

    frame = _frame(room)
    if frame is None:
        return _rebuild(room, [s["center"] for g in room[0] for s in g])
    spheres, centroid, centered, axis, proj, length = frame

    lo = proj.min()
    u = (proj - lo) / length

    # power curve around the pivot; endpoints map to themselves
    x = np.where(u < pivot, (u - pivot) / pivot, (u - pivot) / (1 - pivot))
    new_x = np.sign(x) * np.abs(x) ** power
    new_u = pivot + new_x * np.where(x < 0, pivot, 1 - pivot)
    new_proj = lo + new_u * length

    # perpendicular part, with taper (funnel shape along the axis)
    radial = centered - proj[:, None] * axis
    radial = radial * (1 + taper * (2 * new_u - 1))[:, None]

    new_centers = centroid + new_proj[:, None] * axis + radial

    # bend: arch / sag along the "up" direction perpendicular to the axis
    if bend:
        up = np.array([0.0, 0.0, 1.0])
        up = up - (up @ axis) * axis
        if (
            np.linalg.norm(up) < 1e-6
        ):  # axis is vertical: bend sideways instead
            up = np.array([1.0, 0.0, 0.0]) - axis[0] * axis
        up /= np.linalg.norm(up)
        new_centers = new_centers + np.outer(
            bend * length * np.sin(np.pi * new_u), up
        )

    return _rebuild(room, new_centers)


def hemi_vol(r):
    return 2 / 3 * math.pi * r**3


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# d = distance from the core in units of the reach (0 at the core, 1 at the edge)
FALLOFFS = {
    "linear": lambda d: max(0.0, 1 - d),
    "spike": lambda d: max(0.0, 1 - d) ** 3,
    "plateau": lambda d: _smooth((1 - d) / 0.5),
    "gauss": lambda d: math.exp(-((2.2 * d) ** 2)),
    "ring": lambda d: max(math.sin(math.pi * d), 0.15 * max(0.0, 1 - 2 * d)),
}

# region scales (along, across, vertical); a small scale makes the region thinner there
SHAPES = {
    "round": (1.0, 1.0, 1.0),
    "slab": (1.4, 1.4, 0.5),
    "chimney": (0.7, 0.7, 2.0),
    "lobe": (2.0, 0.7, 0.8),
}

PRESETS = {
    "classic": dict(
        falloff="linear",
        shape="round",
        side="none",
        mode="scale",
        lumps=0.0,
        n=1,
        reach=1.5,
    ),
    "chamber": dict(
        falloff="plateau",
        shape="slab",
        side="none",
        mode="scale",
        lumps=0.15,
        n=1,
        reach=1.6,
    ),
    "blister": dict(
        falloff="spike",
        shape="round",
        side="outward",
        mode="scale",
        lumps=0.0,
        n=1,
        reach=1.5,
    ),
    "chimney": dict(
        falloff="gauss",
        shape="chimney",
        side="none",
        mode="scale",
        lumps=0.1,
        n=1,
        reach=1.6,
    ),
    "gallery": dict(
        falloff="ring",
        shape="slab",
        side="none",
        mode="scale",
        lumps=0.1,
        n=1,
        reach=2.0,
    ),
    "lobe": dict(
        falloff="linear",
        shape="lobe",
        side="outward",
        mode="scale",
        lumps=0.1,
        n=1,
        reach=1.8,
    ),
    "cluster": dict(
        falloff="gauss",
        shape="round",
        side="none",
        mode="add",
        lumps=0.5,
        n=3,
        reach=1.0,
    ),
}


def _solve_k(volume, target):
    """Bisection on the (monotonic) volume as a function of the growth factor k."""
    lo, hi = -1.0, 1.0
    while volume(hi) < target and hi < 1e9:
        hi *= 2
    while volume(lo) > target and lo > -1e9:
        lo *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if volume(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bulge(room, rng=random, preset=None, floor_depth=-1000, **overrides):
    """
    Grows the room's total sphere volume to exactly mult times the original.
    preset:    a key of PRESETS (None = random preset)
    overrides: any preset field, e.g. reach=2.5, falloff="ring", n=2
    """

    mult = 1.5
    room, path = room
    room = copy.deepcopy(room)
    flat = [s for poly in room for s in poly]
    cfg = dict(PRESETS[preset or rng.choice(list(PRESETS))])
    cfg.update(overrides)

    target = sum(hemi_vol(s["radius"]) for s in flat) * mult
    r_ref = sum(s["radius"] for s in flat) / len(flat)
    mx = sum(s["center"][0] for s in flat) / len(flat)
    my = sum(s["center"][1] for s in flat) / len(flat)

    # n distinct cores, biased toward big spheres (by index: no dict comparison)
    pool, cores = list(range(len(flat))), []
    for _ in range(min(cfg["n"], len(flat))):
        i = rng.choices(pool, [flat[j]["radius"] for j in pool])[0]
        cores.append(i)
        pool.remove(i)

    falloff = FALLOFFS[cfg["falloff"]]
    sa, sc, sz = SHAPES[cfg["shape"]]
    w = [0.0] * len(flat)

    for ci in cores:
        cx, cy, cz = flat[ci]["center"]
        reach = cfg["reach"] * flat[ci]["radius"]

        # direction the bulge faces, if one-sided
        side, u = cfg["side"], None
        if side == "outward":
            ox, oy = cx - mx, cy - my
            n = math.hypot(ox, oy)
            if n < 1e-9:
                a = rng.uniform(0, 2 * math.pi)
                ox, oy = math.cos(a), math.sin(a)
            else:
                ox, oy = ox / n, oy / n
            u = (ox, oy, 0.0)
        elif side == "up":
            u = (0.0, 0.0, 1.0)
        elif side == "down":
            u = (0.0, 0.0, -1.0)

        # the long axis of the region follows the facing direction when there is one
        phi = (
            math.atan2(u[1], u[0])
            if side == "outward"
            else rng.uniform(0, 2 * math.pi)
        )
        cp, sp = math.cos(phi), math.sin(phi)

        for j, s in enumerate(flat):
            x, y, z = s["center"]
            dx, dy, dz = x - cx, y - cy, z - cz
            along, across = dx * cp + dy * sp, -dx * sp + dy * cp
            d = (
                math.sqrt(
                    (along / sa) ** 2 + (across / sc) ** 2 + (dz / sz) ** 2
                )
                / reach
            )
            if d >= 1:
                continue
            wj = falloff(d)
            if u is not None:  # fade out spheres behind the core
                wj *= _smooth(
                    ((dx * u[0] + dy * u[1] + dz * u[2]) / reach + 0.5) / 0.5
                )
            w[j] = max(w[j], wj)

    if cfg["lumps"]:
        w = [wj * max(0.0, 1 + cfg["lumps"] * rng.uniform(-1, 1)) for wj in w]

    def grown(r, wj, k):
        g = r + k * wj * r_ref if cfg["mode"] == "add" else r * (1 + k * wj)
        return max(0.05 * r, g)

    def volume(k):
        return sum(
            hemi_vol(grown(s["radius"], wj, k)) for s, wj in zip(flat, w)
        )

    k = _solve_k(volume, target)
    for s, wj in zip(flat, w):
        s["radius"] = grown(s["radius"], wj, k)
    for ci in cores:
        flat[ci]["floor_depth"] = floor_depth
    return room, path


def scatter(
    room, reach=2.0, strength=6, z_scale=1, coherence=0.7, pick="weighted", n=1
):
    """
    Scatters the spheres around each of n core spheres. Radii (and volume) unchanged.
    reach:     influence radius, as a multiple of each core's radius
    strength:  max displacement, as a fraction of each sphere's own radius
    z_scale:   vertical displacement relative to horizontal (0 = flat scatter)
    coherence: 0 = independent jitter per sphere, 1 = one shared offset per core
    pick:      "weighted" (random, biased toward big spheres) or "largest"
               (the n largest spheres)
    n:         number of cores (clamped to the number of spheres in the room)
    """

    room, path = room
    flat = [s for poly in room for s in poly]
    n = min(n, len(flat))

    # choose n distinct cores up front (by index, so no dict comparison)
    if pick == "largest":
        order = sorted(
            range(len(flat)), key=lambda i: flat[i]["radius"], reverse=True
        )
        cores = [flat[i] for i in order[:n]]
    else:
        pool = list(range(len(flat)))
        cores = []
        for _ in range(n):
            i = random.choices(
                pool, weights=[flat[j]["radius"] for j in pool]
            )[0]
            cores.append(flat[i])
            pool.remove(i)  # ints compare safely

    def offset():
        a = random.uniform(0, 2 * math.pi)
        return (math.cos(a), math.sin(a), random.uniform(-1, 1) * z_scale)

    for core in cores:
        cx, cy, cz = core["center"]
        r_reach = reach * core["radius"]
        shared = offset()
        for s in flat:
            if s is core:  # each core stays anchored in its own scatter
                continue
            d = math.dist(s["center"], (cx, cy, cz))
            w = max(0.0, 1 - d / r_reach)
            if w == 0:
                continue
            own = offset()
            v = [
                coherence * a + (1 - coherence) * b
                for a, b in zip(shared, own)
            ]
            amount = strength * s["radius"] * w
            c = s["center"]
            s["center"] = tuple(c[i] + v[i] * amount for i in range(3))

    return room, path


def do_nothing_transform(room):
    return room


ROOM_TRANSFORMS = {
    "I": room_rotation,
    "J": perlin_heightmap_transform,
    "H": twist_transform,
    "F": flat_sweep,
    "G": nonlinear_scaling,
    "K": self_overlap,
    "L": taper_transform,
    "M": mirror_room_transform,
    "Q": vertical_stack,
    "S": room_half_drop,
    "h": bulge,
    "i": scatter,
    "l": pit_transform,
    "q": rotate_half,
    "r": lens,
    "s": invert_retry,
    "t": complex_map_retry,
    "u": attractor_flow_retry,
    "x": periodic_wrap,
    "y": echo,
    "w": cave_in,
    "N": negative_space,
    "z": do_nothing_transform,
}


def apply_transform(transform, room):
    r = ROOM_TRANSFORMS[transform](room)
    return r


# ----------------------------------------------------------
# -------- ROOM COMBINERS  ---------------------------------
# ----------------------------------------------------------


def combine_overlap(left, right):
    a2, a1 = left
    b2, b1 = right
    b2 = shift_room_multiple(
        b2,
        (
            random.randint(500, 900),
            random.randint(500, 900),
            random.randint(-400, 500),
        ),
    )
    new_room = a2
    for cluster in b2:
        new_room.append(cluster)
    traversal_paths = a1
    for path in b1:
        traversal_paths.append(path)
    return (new_room, traversal_paths)


def combine_tunnel(left, right):
    a2, a1 = left
    b2, b1 = right
    b2 = shift_room_multiple(
        b2,
        (
            random.randint(3000, 4000),
            random.randint(3000, 4000),
            random.randint(-500, 500),
        ),
    )
    new_room = a2.copy()
    traversal_paths = a1.copy()
    shaft = intersection_shaft(list(flatten(b2)), list(flatten(a2)))
    if shaft:
        tunnel_method = random.choice(["straight", "curved"])
        if tunnel_method == "straight":
            new_room += [shaft]
            traversal_paths += [[0, 1]]
        else:
            tunnel = quadratic_bezier_3d(
                shaft[0]["center"],
                shaft[1]["center"],
                np.array(shaft[0]["center"]) - np.array(shaft[1]["center"]),
                n_points=25,
            )
            tunnel = points_to_sphere(tunnel, random.randrange(500, 800))
            new_room += [tunnel]
            traversal_paths += [list(range(25))]

    new_room += b2
    traversal_paths += b1
    return (new_room, traversal_paths)


def vertical_combine(left, right):
    a2, a1 = left
    b2, b1 = right

    # Get the biggest sphere from a2:
    biggest = sorted(
        list(flatten(a2)), key=lambda x: x["radius"], reverse=True
    )[0]

    method = random.choice(["vertical", "angled"])

    if method == "vertical":

        shaft_depth = -random.randint(2500, 3000)
        shifting_point = (
            biggest["center"][0],
            biggest["center"][1],
            biggest["center"][2] + shaft_depth,
        )
        b2 = shift_room_multiple(b2, shifting_point)

        shaft = [
            {
                "center": biggest["center"],
                "radius": biggest["radius"] * random.uniform(0.4, 0.7),
            },
            {
                "center": shifting_point,
                "radius": biggest["radius"] * random.uniform(0.4, 0.7),
            },
        ]

    else:
        shaft_depth = random.choice([1, -1]) * random.randint(2500, 3000)
        circle_radius = random.randint(2000, 2500)
        angle = np.deg2rad(random.randint(20, 40))
        shifting_point = (
            biggest["center"][0] + math.cos(angle) * circle_radius,
            biggest["center"][1] + math.sin(angle) * circle_radius,
            biggest["center"][2] + shaft_depth,
        )
        b2 = shift_room_multiple(b2, shifting_point)

        shaft = [
            {
                "center": biggest["center"],
                "radius": biggest["radius"] * random.uniform(0.4, 0.7),
            },
            {
                "center": shifting_point,
                "radius": biggest["radius"] * random.uniform(0.4, 0.7),
            },
        ]

    new_room = a2.copy()
    new_room += [shaft]
    new_room += b2
    traversal_paths = a1.copy()
    traversal_paths += [[0, 1]]
    traversal_paths += b1

    return new_room, traversal_paths


ROOM_COMBINERS = {
    "O": combine_overlap,
    "T": combine_tunnel,
    "Y": vertical_combine,
    "v": morph_rooms,
    "o": seam,
    "p": braid,
    "g": crossover,
}


def combine(operation, left, right):
    return ROOM_COMBINERS[operation](left, right)


# ----------------------------------------------------------
# -------- ROOM GENERATORS ---------------------------------
# ----------------------------------------------------------
def transform_cave(cave):
    # 1. Anisotropic scaling
    scaling = random.choice(["x", "y", "none"])
    sx = sy = 1
    if scaling == "x":
        sx = np.random.uniform(2, 2.5)
        sy = np.random.uniform(1, 1.1)
    if scaling == "y":
        sx = np.random.uniform(1, 1.1)
        sy = np.random.uniform(2, 2.5)

    cave = zoom(cave, (sy, sx), order=0)

    # 2. Random warp
    # strength = np.random.uniform(15, 40)
    strength = 30
    smoothness = np.random.uniform(15, 40)

    cave = random_warp(cave, strength=strength, smoothness=smoothness)

    return cave


def random_warp(tilemap, strength=5, smoothness=20):
    h, w = tilemap.shape

    # Random displacement fields
    dx = np.random.randn(h, w)
    dy = np.random.randn(h, w)

    # Make displacement spatially coherent
    dx = gaussian_filter(dx, smoothness)
    dy = gaussian_filter(dy, smoothness)

    # Normalize
    dx /= np.std(dx)
    dy /= np.std(dy)

    dx *= strength
    dy *= strength

    y, x = np.indices((h, w))

    warped = map_coordinates(
        tilemap, [y + dy, x + dx], order=0, mode="nearest"
    )

    return warped.astype(np.uint8)


def random_point_on_circle(cx, cy, radius):
    angle = random.uniform(0, 2 * math.pi)

    x = cx + radius * math.cos(angle)
    y = cy + radius * math.sin(angle)

    return x, y


def generate_atomic_room():

    main_shaft_radius = random.randint(600, 850)
    room = [{"center": (0, 0, 0), "radius": main_shaft_radius}]
    for ii in range(random.randint(2, 3)):
        x, y = random_point_on_circle(0, 0, main_shaft_radius)
        room.append(
            {
                "center": (int(x), int(y), random.uniform(-500, 500)),
                "radius": random.randint(450, 550),
            }
        )
    return room, list(range(len(room)))


def generate_small_graph_room():
    return generate_graph_room(3)


def generate_big_graph_room():
    return generate_graph_room(random.randint(4, 5))


def generate_graph_room(number_of_rooms):

    N = number_of_rooms
    # N = 100
    # bounds = random.choice([(8000, 8000), (1000, 10000)])
    # :param shape: Layout archetype ('organic', 'caverns', 'linear_tunnel',
    #     'ring', 'river').
    shape = random.choice(
        ["ring", "linear_tunnel", "river", "organic", "caverns"]
    )
    graph = generate_cave_graph_w_structure(N, shape=shape)
    new_room = []
    traversal_paths = []
    for room_id, coords in graph["nodes"].items():
        # print(f"  Room {room_id}: {coords}")
        atomic_cave, path = generate_atomic_room()
        atomic_cave = shift_room(atomic_cave, (coords[0], coords[1], 0))
        new_room += [atomic_cave]
        traversal_paths += [path]

    new_room = shift_room_multiple(new_room, (0, 0, 0))

    visited = {}
    for ii, (_, neighbors) in enumerate(graph["adjacency_list"].items()):
        for neighbor, _ in neighbors:
            if neighbor not in visited.get(ii, {}):
                shaft = intersection_shaft(new_room[ii], new_room[neighbor])
                if shaft:
                    for e in shaft:
                        e["radius"] = min(400, e["radius"] * 0.7)
                    new_room += [shaft]
                    traversal_paths += [[0, 1]]
                if ii in visited:
                    visited[ii].append(neighbor)
                else:
                    visited[ii] = [neighbor]
    return new_room, traversal_paths


def generate_combined_room():
    # Generates either an automaton, random walk or DLA room:
    return random.choice(
        [
            generate_automaton_room(),
            generate_random_walk_room(),
            generate_DLA_room(),
        ]
    )


def generate_automaton_room():
    while True:
        try:
            tilemap = run_automata(
                inverted=False,
                randomize_size=False,
                remove_pockets=True,
                size=random.randint(20, 35),
            )
            # mutated_tilemap = transform_cave(tilemap)
            path, circles, *_ = create_room(tilemap)
            break
        except RoomTooSmall:
            continue
    circles = scale_spheres(circles, random.uniform(1.1, 1.6))
    circles = shift_room(circles, (0, 0, 0))
    return circles, path


def generate_random_walk_room():
    while True:
        try:
            tilemap = create_random_walk(50, 50, 100)
            path, circles, *_ = create_room(tilemap)
            break
        except RoomTooSmall:
            continue
    circles = scale_spheres(circles, random.uniform(1.1, 1.6))
    circles = shift_room(circles, (0, 0, 0))
    return circles, path


def generate_worms_room():
    tilemap, path = generate_3d_worm_cave_parameterized(3)
    circles = [
        {
            "center": (float(s[0][0]), float(s[0][1]), float(s[0][2])),
            "radius": s[1],
        }
        for s in tilemap
    ]
    circles = shift_room(circles, (0, 0, 0))
    circles = scale_spheres(circles, 50)
    return circles, path


def generate_worms_room_big():
    tilemap, path = generate_3d_worm_cave(3)
    circles = [
        {
            "center": (float(s[0][0]), float(s[0][1]), float(s[0][2])),
            "radius": s[1],
        }
        for s in tilemap
    ]
    circles = shift_room(circles, (0, 0, 0))
    circles = scale_spheres(circles, 50)
    return circles, path


def generate_DLA_room():
    # tilemap = create_DLA_tilemap(20, 20, 0.05, 0.4)
    tilemap = create_DLA_tilemap(30, 30, 0.02, 0.8)
    path, circles, *_ = create_room(tilemap)
    circles = scale_spheres(circles, random.uniform(1.1, 1.6))
    return circles, path


def generate_erosion_room():
    spheres, path = erosion_room(25, 500, method="erosion")
    circles = [
        {
            "center": (float(s[0][0]), float(s[0][1]), float(s[0][2])),
            "radius": s[1],
        }
        for s in spheres
    ]
    circles = shift_room(circles, (0, 0, 0))
    circles = scale_spheres(circles, 140)
    return circles, path


def simplified_erosion_room():
    spheres, path = erosion_room(20, 500, method="erosion")
    circles = [
        {
            "center": (float(s[0][0]), float(s[0][1]), float(s[0][2])),
            "radius": s[1],
        }
        for s in spheres
    ]
    circles = shift_room(circles, (0, 0, 0))
    circles = scale_spheres(circles, 140)
    return circles, path


def generate_noise_room():
    spheres, path = create_noise_room()
    circles = [
        {
            "center": (float(s[0][0]), float(s[0][1]), float(s[0][2])),
            "radius": s[1],
        }
        for s in spheres
    ]
    circles = shift_room(circles, (0, 0, 0))
    circles = scale_spheres(circles, 350)
    return circles, path


def simplified_noise_room():
    spheres, path = create_noise_room(15, 15, 8)
    circles = [
        {
            "center": (float(s[0][0]), float(s[0][1]), float(s[0][2])),
            "radius": s[1],
        }
        for s in spheres
    ]
    circles = shift_room(circles, (0, 0, 0))
    circles = scale_spheres(circles, 350)
    return circles, path


def generate_tunnel_room():

    A = (0, 0)
    B = random.choice([(10, 5), (-10, -5)])

    curve = midpoint_displacement(A, B, depth=6, roughness=0.35)
    height_function = random.choice([-1, 1])
    curve = points_2d_to_shape(
        curve,
        2,
        lambda i, x, y: i * height_function * random.uniform(0.06, 0.08),
    )
    curve = scale_spheres(curve, 350)
    curve = shift_room(curve, (0, 0, 0))
    path = list(range(len(curve)))

    return curve, path


def generate_fancy_tunnel_room():

    from tunnels_2 import generate_tunnel

    tunnel = generate_tunnel(
        (0, 0, 0), (1, 0, -0.5), length=500, step=20, max_segments=1
    )
    # print(tunnel)

    radius = random.randint(80, 100)
    curve = [{"center": p, "radius": radius} for p in tunnel]
    curve = scale_spheres(curve, random.randint(5, 8))
    curve = shift_room(curve, (0, 0, 0))
    path = list(range(len(curve)))

    return curve, path


def generate_character_room():

    letter_type = random.choice(["alphabet", "math"])

    def random_character():
        match letter_type:
            case "alphabet":
                chars = (
                    string.ascii_uppercase
                    + string.ascii_lowercase
                    + string.digits
                )
            case "math":
                chars = (
                    list(range(0x2A00, 0x02AFF))
                    + list(range(0x2200, 0x22FF))
                    + list(range(0x1EE05, 0x1EE1F))
                )

        return random.choice(chars)

    def make_letter_room(letter):
        match letter_type:
            case "alphabet":
                points, path = assemble_letter(letter)
            case "math":
                letter = chr(letter)
                points, path = assemble_letter_math(letter)
        points = [(int(p1), int(p2)) for p1, p2 in points]
        points = points_2d_to_shape(points, 23, lambda x, y, z: 0)
        points = shift_room(points, (0, 0, 0))
        points = scale_spheres(points, 35)
        return points, path

    while True:
        try:
            circles, path = make_letter_room(random_character())
            break
        except IndexError:
            continue
    return circles, path


def generate_big_worley_room():
    return generate_worley_room([(10, 10, 8), (8, 8, 8)])


def generate_small_worley_room():
    return generate_worley_room([(10, 10, 4), (8, 8, 4)])


def generate_worley_room(shapes):
    from worley import generate_worley_cave

    spheres, path = generate_worley_cave(shapes)
    circles = [
        {
            "center": (float(s[0][0]), float(s[0][1]), float(s[0][2])),
            "radius": s[1],
        }
        for s in spheres
    ]
    circles = shift_room(circles, (0, 0, 0))
    circles = scale_spheres(circles, 300)
    return circles, path


def generate_small_pit_room():
    return generate_pit_room(2000)


def generate_big_pit_room():
    return generate_pit_room(4500)


def generate_pit_room(max_height):

    from main import sample_helix, circle_points

    pit_method = random.choice(["spiral", "ribbed", "funnel", "straight"])
    # print(pit_method)
    direction = 1
    radius = random.randrange(700, 1200)
    height = random.uniform(1500, max_height)
    if pit_method == "beaded":
        small_r = min(0.5 * radius, 500)
        big_r = radius
        n = random.randint(5, 10)
        main_shaft = [
            {
                "center": (0, ii * n, (small_r + big_r) * ii),
                "radius": small_r if ii % 2 == 0 else big_r,
            }
            for ii in range(n)
        ]
        traversal_paths = [list(range(n))]
        new_room = [main_shaft]
    if pit_method == "straight":
        n = random.randint(5, 10)
        main_shaft = [
            {
                "center": (0, ii * n, ii * height / n),
                "radius": radius,
            }
            for ii in range(n)
        ]
        traversal_paths = [list(range(n))]
        new_room = [main_shaft]
    elif pit_method == "funnel":
        n = random.randint(5, 10)
        main_shaft = [
            {
                "center": (0, ii * n, ii * height / n),
                "radius": 300 + 150 * ii,
            }
            for ii in range(n)
        ]
        traversal_paths = [list(range(n))]
        new_room = [main_shaft]
    if pit_method == "spiral":
        main_shaft = [
            {"center": (0.0, 0.0, 0.0), "radius": radius},
            {"center": (0.0, 0.0, 0.0), "radius": radius},
            {"center": (0.0, 0.0, direction * height), "radius": radius},
            {"center": (0.0, 0.0, direction * height), "radius": radius * 1.4},
        ]
        traversal_paths = [[0, 1, 2, 3]]
        new_room = [main_shaft]
        turns = 1
        main_helix = sample_helix(radius * 1.3, direction * height, turns, 40)
        main_helix = [{"center": h, "radius": 500} for h in main_helix]
        new_room += [main_helix]
        traversal_paths += [[ii + 4 for ii in list(range(len(main_helix)))]]
    if pit_method == "ribbed":
        main_shaft = [
            {"center": (0.0, 0.0, 0.0), "radius": radius},
            {"center": (0.0, 0.0, 0.0), "radius": radius},
            {"center": (0.0, 0.0, direction * height), "radius": radius},
            {"center": (0.0, 0.0, direction * height), "radius": radius * 1.4},
        ]
        traversal_paths = [[0, 1, 2, 3]]
        new_room = [main_shaft]
        for ii in [random.random() for _ in range(random.randint(1, 3))]:
            circle = circle_points(6, radius * 1.3, (0, 0))
            circle = [
                {
                    "center": (h[0], h[1], height * ii),
                    "radius": random.randint(300, 600),
                }
                for h in circle
            ]
            new_room += [circle]
            traversal_paths += [
                [ii + len(list(flatten(traversal_paths))) for ii in range(6)]
            ]

    new_room = shift_room(
        list(flatten(new_room)),
        (0, 0, height / 2),
    )

    return (new_room, list(flatten(traversal_paths)))


def create_blur_room():

    spheres, path = generate_blur_room(
        random.randint(1, 100000), 16, 2, 0.2, 3
    )
    circles = [
        {
            "center": (float(s[0][0]), float(s[0][1]), 0),
            "radius": s[1],
        }
        for s in spheres
    ]
    circles = shift_room(circles, (0, 0, 0))
    circles = scale_spheres(circles, 300)
    # print(circles)
    return circles, path


ROOM_OPERATIONS = {
    "A": generate_automaton_room,
    "B": generate_random_walk_room,
    "C": generate_DLA_room,
    "D": generate_erosion_room,
    "E": generate_noise_room,
    "P": generate_letter_room,
    "R": generate_tunnel_room,
    "U": simplified_erosion_room,
    "V": simplified_noise_room,
    "W": generate_worms_room,
    "X": generate_worms_room_big,
    "Z": generate_fancy_tunnel_room,
    "a": generate_small_worley_room,
    "b": generate_small_pit_room,
    "c": generate_big_worley_room,
    "d": generate_big_pit_room,
    "e": generate_small_graph_room,
    "f": generate_big_graph_room,
    "g": create_blur_room,
    "j": generate_combined_room,
    "k": generate_lsystem_room,
    "m": generate_atomic_room,
    "n": kaleidoscope_room,
    "o": lattice_room,
    "p": polytope_room,
}


def generate_room(room_type):
    a2, a1 = ROOM_OPERATIONS[room_type]()
    if type(a2[0]) is list:
        return a2, a1
    else:
        return [a2], [a1]


def shift_room_multiple(spheres, point):
    # shifted_room = []
    # for cluster in spheres:
    # shifted_room.append(shift_room(cluster, point))
    shift_vector = np.array(point) - np.array(spheres[0][0]["center"])
    shifted_room = []
    for cluster in spheres:
        shifted_room.append(
            [
                {
                    "center": (np.array(c["center"] + shift_vector)),
                    "radius": c["radius"],
                }
                for c in cluster
            ]
        )
    return shifted_room


def shift_room_multiple_2d(spheres, point):
    shifted_room = []
    for cluster in spheres:
        shifted_room.append(shift_room_2d(cluster, point))
    return shifted_room


def scale_room_multiple(spheres, scale_factor):
    scaled_room = []
    for cluster in spheres:
        scaled_room.append(scale_spheres(cluster, scale_factor))
    return scaled_room


def scale_by_volume(sphere_groups, cube_size):
    """
    Uniformly scales a nested list of spheres so they fit inside a cube.

    Parameters
    ----------
    sphere_groups : list[list[dict]]
        Each sphere is:
            {
                "center": (x, y, z),
                "radius": r
            }

    cube_size : float or (sx, sy, sz)
        Side length of the cube or dimensions of a rectangular box.

    Returns
    -------
    scaled_groups : list[list[dict]]
        Same nested structure as the input.
    scale : float
        The scale factor applied.
    """

    if not sphere_groups:
        return [], 1.0

    # Flatten to compute the bounding box
    all_spheres = [s for group in sphere_groups for s in group]

    if not all_spheres:
        return [[] for _ in sphere_groups], 1.0

    # Allow either a scalar cube size or (sx, sy, sz)
    if isinstance(cube_size, (int, float)):
        sx = sy = sz = float(cube_size)
    else:
        sx, sy, sz = cube_size

    min_x = min(s["center"][0] - s["radius"] for s in all_spheres)
    max_x = max(s["center"][0] + s["radius"] for s in all_spheres)

    min_y = min(s["center"][1] - s["radius"] for s in all_spheres)
    max_y = max(s["center"][1] + s["radius"] for s in all_spheres)

    min_z = min(s["center"][2] - s["radius"] for s in all_spheres)
    max_z = max(s["center"][2] + s["radius"] for s in all_spheres)

    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z

    scale = min(
        sx / width if width > 0 else float("inf"),
        sy / height if height > 0 else float("inf"),
        sz / depth if depth > 0 else float("inf"),
    )

    scaled_groups = []
    for group in sphere_groups:
        scaled_group = []
        for s in group:
            x, y, z = s["center"]
            scaled_group.append(
                {
                    "center": (x * scale, y * scale, z * scale),
                    "radius": s["radius"] * scale,
                }
            )
        scaled_groups.append(scaled_group)

    return scaled_groups, scale


def get_room_center_multiple(room):
    """
    Compute the volume-weighted geometric center of a collection of spheres.

    Each sphere is a dict with:
        {"center": [x, y, z], "radius": r}
    """
    total_volume = 0.0
    weighted_center = np.zeros(3)

    spheres = list(flatten(room))

    for sphere in spheres:
        center = np.asarray(sphere["center"], dtype=float)
        radius = float(sphere["radius"])

        volume = (4 / 3) * np.pi * radius**3

        weighted_center += volume * center
        total_volume += volume

    if total_volume == 0:
        raise ValueError("Total sphere volume must be greater than zero")

    return weighted_center / total_volume


def sample_cube_size(target_mean=10, sigma=0.3):
    mu = math.log(target_mean) - sigma**2 / 2
    return random.lognormvariate(mu, sigma)


def has_entrance_and_exit(tuples_list):
    types = {t[-1] for t in tuples_list}
    check = "Entrance" in types and "Exit" in types
    centers = list(flatten([(t[0], t[1], t[2]) for t in tuples_list]))
    if not check or np.any(np.isnan(centers)):
        raise MissingEntranceOrExit


class MissingEntranceOrExit(Exception):
    pass


def run():
    # generation = REFINERY for refinery and escort rooms in custom rooms
    # generation = WILD for complex rooms (chaotic preset)
    # generation = SIMPLE for simpler rooms (neutral preset)
    with open("ref/volume_1.json", "r") as f:
        ref_1 = json.load(f)
    with open("ref/volume_1.5.json", "r") as f:
        ref_1_5 = json.load(f)
    with open("ref/volume_2.json", "r") as f:
        ref_2 = json.load(f)
    with open("ref/volume_2.5.json", "r") as f:
        ref_2_5 = json.load(f)
    with open("ref/volume_3.json", "r") as f:
        ref_3 = json.load(f)
    with open("ref/volume_3.5.json", "r") as f:
        ref_3_5 = json.load(f)
    with open("ref/volume_4.json", "r") as f:
        ref_4 = json.load(f)
    calibration_data = {
        1: ref_1,
        1.5: ref_1_5,
        2: ref_2,
        2.5: ref_2_5,
        3: ref_3,
        3.5: ref_3_5,
        4: ref_4,
    }
    generation = "SIMPLE"
    if generation == "REFINERY" or generation == "REFINERY_CUSTOMROOMS":
        tags = ["small", "medium", "big"]
        num_rooms = {"small": 40, "medium": 40, "big": 40}
    else:
        tags = ["small", "medium", "big", "end"]
        num_rooms = {"small": 300, "medium": 300, "big": 300, "end": 50}
        # num_rooms = {"small": 3, "medium": 3, "big": 3, "end": 5}
    for tag in tags:
        for i in range(num_rooms[tag]):
            while True:
                try:
                    match tag:
                        case "small":
                            if generation == "REFINERY_CUSTOMROOMS":
                                room_tag = ["Flat.Linear.CustomSmall"]
                            else:
                                room_tag = ["Rooms.Linear.CustomSmall"]
                        case "medium":
                            if generation == "REFINERY_CUSTOMROOMS":
                                room_tag = ["Flat.Linear.CustomMedium"]
                            else:
                                room_tag = ["Rooms.Linear.CustomMedium"]
                        case "end":
                            room_tag = ["Rooms.Linear.End"]
                        case "big":
                            if generation == "REFINERY_CUSTOMROOMS":
                                room_tag = ["Flat.Linear.CustomBig"]
                            else:
                                room_tag = ["Rooms.Linear.CustomBig"]
                    room_num = i
                    if generation == "WILD":
                        if tag == "small":
                            target_volume = 1.5
                            new_room, traversal_path = roll_interesting(
                                lambda r: make_room(target_volume),
                                random,
                                calibration_data[target_volume],
                            )

                        elif tag == "medium" or tag == "end":
                            target_volume = 2
                            new_room, traversal_path = roll_interesting(
                                lambda r: make_room(target_volume),
                                random,
                                calibration_data[target_volume],
                            )
                        else:
                            target_volume = random.choices(
                                [2, 3, 4], weights=[0.15, 0.7, 0.15], k=1
                            )[0]
                            new_room, traversal_path = roll_interesting(
                                lambda r: make_room(target_volume),
                                random,
                                calibration_data[target_volume],
                            )
                    elif generation == "SIMPLE":
                        if tag == "small":
                            target_volume = random.choice([1, 1.5])
                            new_room, traversal_path = roll_interesting(
                                lambda r: make_room(target_volume),
                                random,
                                calibration_data[target_volume],
                            )
                            # room = generate(
                            #     target_volume=random.choice([1, 1.5])
                            #     # target_volume=3
                            # )
                        elif tag == "medium" or tag == "end":
                            target_volume = random.choice([1.5, 2])
                            new_room, traversal_path = roll_interesting(
                                lambda r: make_room(target_volume),
                                random,
                                calibration_data[target_volume],
                            )
                            # room = generate(
                            #     target_volume=random.choice([1.5, 2]),
                            # )
                        else:
                            target_volume = random.choices(
                                [1.5, 2, 2.5, 3],
                                weights=[0.15, 0.35, 0.35, 0.35],
                                k=1,
                            )[0]
                            new_room, traversal_path = roll_interesting(
                                lambda r: make_room(target_volume),
                                random,
                                calibration_data[target_volume],
                            )
                            # room = generate(
                            #     target_volume=random.choices(
                            #         [1.5, 2, 2.5, 3],
                            #         weights=[0.15, 0.35, 0.35, 0.35],
                            #         k=1,
                            #     )[0],
                            # )
                    else:
                        if tag == "small":
                            room = generate(target_volume=1)
                        elif tag == "medium" or tag == "end":
                            room = generate(
                                target_volume=random.choice([1, 1.5]),
                            )
                        else:
                            room = generate(
                                target_volume=random.choices(
                                    [2, 3], weights=[0.85, 0.15], k=1
                                )[0],
                            )

                    # tree, _ = parse(room)
                    # print(room, tree)
                    # new_room, traversal_path = build(tree)
                    if len(new_room) == 0:
                        raise MissingEntranceOrExit
                    if len(new_room) == 1 and len(new_room[0]) == 1:
                        raise MissingEntranceOrExit
                    # print(new_room, traversal_path)
                    # entrances = generate_entrances(
                    #     list(flatten(new_room)),
                    #     exit_interval=(200, 280),
                    #     entrance_interval=(30, 140),
                    #     return_type="Entrances",
                    # )
                    # exits = generate_exits_furthest(
                    #     list(flatten(new_room)), entrances
                    # )
                    entrances = place_doors((new_room, traversal_path))
                    # print(entrances)
                    # has_entrance_and_exit(entrances + exits)
                    break
                except (MissingEntranceOrExit, MissingEntranceOrExit_new):
                    continue
            pillars = generate_pillars(new_room) + generate_big_pillars(
                new_room
            )
            # print(pillars)
            pe = []
            if generation == "SIMPLE":
                name = f"RMA_ICGEN_{tag.upper()}_GENERAL_SIMPLE{room_num}.json"
                room_name = f"{tag.upper()}_GENERAL_SIMPLE"
            elif generation == "REFINERY":
                name = f"RMA_ICGEN_{tag.upper()}_GENERAL_REFINERY_ESCORT{room_num}.json"
                room_name = f"{tag.upper()}_GENERAL_REFINERY_ESCORT"
            elif generation == "REFINERY_CUSTOMROOMS":
                name = f"RMA_ICGEN_{tag.upper()}_FLAT{room_num}.json"
                room_name = f"{tag.upper()}_FLAT"
            else:
                name = f"RMA_ICGEN_{tag.upper()}_GENERAL{room_num}.json"
                room_name = f"{tag.upper()}_GENERAL"

            room_json = build_json_multiple_ffill(
                traversal_path,
                new_room,
                entrances,
                pillars,
                room_num,
                room_name,
                {"bounds": 4000, "tags": room_tag},
                pe,
            )

            # try:
            #     with open(f"room_log_{generation}.txt", "w") as logfile:
            #         logfile.write(f"{room_name}{room_num} {room}\n")
            # except Exception as e:
            #     logging.error(f"Couldn't write {name} on logfile: {e}")

            try:
                path = Path("generated")
                path.mkdir(parents=True, exist_ok=True)
                with open(f"generated/{name}", "w") as f:
                    json.dump(room_json, f, indent=4)
                logging.info(f"Written {name} to generated/{name}")
            except Exception as e:
                logging.error(f"Couldn't write {name}: {e}")


def make_room(target_volume):
    room = generate(target_volume)
    tree, _ = parse(room)
    print(room, tree)
    new_room, path = build(tree)
    return new_room, path


def calibration():
    for target_volume in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
        ref = calibrate(
            lambda r: make_room(1), n=200
        )  # once per volume tier; save it to a file
        data = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in ref.items()
        }
        with open(f"ref/volume_{target_volume}.json", "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    run()
