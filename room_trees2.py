import random
import math
from pathlib import Path
import json
import logging
import string
import numpy as np
from more_itertools import flatten
from worms import generate_3d_worm_cave_parameterized, generate_3d_worm_cave
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
from noise3d_2 import create_noise_room, create_anisotropic_noise_room
from erosion import erosion_room
from alphabet import (
    assemble_letter,
    assemble_letter_devanagari,
    assemble_letter_math,
)
from json_builder import (
    generate_entrances,
    generate_exit_w_vector,
    generate_PE_features,
    build_json,
    build_json_multiple_ffill,
    generate_pillars,
    generate_pillars_balls,
)

# Normal dictionary:
# ROOMS = "ABCDEUVPRWXZa"
ROOMS = "a"
TRANSFORMS = ""
MODIFIERS = "IJQSK"
COMBINERS = "OTY"

# Some rooms require transforms
REQUIRED_MODIFIERS = {
    "A": "JIQSK",
    "B": "JIQSK",
    "C": "JIQSK",
    "P": "JIQSK",
}
# # Refinery / Escort dictionary:
# ROOMS = "ABCUVPRW"
# TRANSFORMS = "MF"
# MODIFIERS = "JK"
# COMBINERS = "O"


# # Some rooms require transforms
# REQUIRED_MODIFIERS = {
#     "A": "JK",
#     "B": "JK",
#     "C": "JK",
#     "P": "JK",
# }


def make_room(rooms_list):
    room = random.choice(rooms_list)

    # Prefix notation means transforms come BEFORE the room.
    transforms = REQUIRED_MODIFIERS.get(room, "")
    if transforms:
        transforms = random.choice(transforms)

    # "GHC" means Scale(Mirror(RoomC))
    return transforms + room


def generate(depth=2, n_rooms=1, rooms_list="ACBDEPRUVW"):
    # Must produce exactly one room.
    if n_rooms == 1:
        room = make_room(rooms_list)

        # Optional transforms.
        while depth > 0 and random.random() < 0.7:
            room = random.choice(TRANSFORMS + MODIFIERS) + room
            depth -= 1

        return room

    # Split the rooms between the left and right branches.
    left_rooms = random.randint(1, n_rooms - 1)
    right_rooms = n_rooms - left_rooms

    return (
        # random.choices(COMBINERS, weights=[0.6, 0.2, 0.2], k=1)[0]
        random.choices(COMBINERS, weights=[0.8, 0.1, 0.1], k=1)[0]
        # random.choice(COMBINERS)
        + generate(depth - 1, left_rooms, rooms_list)
        + generate(depth - 1, right_rooms, rooms_list)
    )


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

    if token in TRANSFORMS:
        child, j = parse(s, i + 1)
        match token:
            case "F":
                return ("Curve Sweep", token, child), j
            case "G":
                return ("Spiral Sweep", token, child), j
            case "L":
                return ("Taper", token, child), j
            case "M":
                return ("Mirror", token, child), j
            case "N":
                return ("Flat Sweep", token, child), j

    if token in MODIFIERS:
        child, j = parse(s, i + 1)
        match token:
            case "K":
                return ("Self Overlap", token, child), j
            case "H":
                return ("Flat", token, child), j
            case "I":
                return ("Rotation", token, child), j
            case "J":
                return ("Perlin Heightmap", token, child), j
            case "Q":
                return ("Vertical Stack", token, child), j
            case "S":
                return ("Vertical Drop", token, child), j

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

    raise ValueError(f"Unknown token: {token}")


def build(node):
    kind = node[1]

    if kind in ROOMS:
        room = generate_room(node[1])
        return room

    if kind in TRANSFORMS or kind in MODIFIERS:
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
    perpendicular_vector = perpendicular_2d_normalized(direction_vector)
    depth_m = random.choices(["depth", "shift"], k=1, weights=[0.8, 0.2])[0]
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
    if d["intersects"]:
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

    def compute_curve(N):
        A = (0, 0)
        B = random.choice([(10, 5), (-10, -5)])

        curve = midpoint_displacement(A, B, depth=6, roughness=0.35)
        height_function = 0
        curve = points_2d_to_shape(
            curve,
            2,
            lambda i, x, y: i * height_function * random.uniform(0.06, 0.08),
        )
        curve = scale_spheres(curve, 350)
        step = len(curve) // N
        sample = curve[::step]
        return sample

    curve_points = compute_curve(3)
    a2, a1 = room
    swept_room = a2.copy()
    paths = a1.copy()
    for point in curve_points:
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


def do_nothing_transform(room):
    return room


ROOM_TRANSFORMS = {
    "I": room_rotation,
    "J": perlin_heightmap_transform,
    "H": do_nothing_transform,
    "F": curve_sweep_transform,
    "G": do_nothing_transform,
    "K": self_overlap,
    "L": taper_transform,
    "M": mirror_room_transform,
    "N": flat_sweep,
    "Q": vertical_stack,
    "S": room_half_drop,
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
            random.randint(2000, 3000),
            random.randint(2000, 3000),
            random.randint(-400, 400),
        ),
    )
    new_room = a2.copy()
    traversal_paths = a1.copy()
    shaft = intersection_shaft(list(flatten(b2)), list(flatten(a2)))
    if shaft:
        tunnel_method = random.choices(["straight", "curved"])
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

        shaft_depth = -random.randint(2000, 2500)
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
        shaft_depth = random.choice([1, -1]) * random.randint(2000, 2500)
        circle_radius = random.randint(2000, 2500)
        angle = np.deg2rad(random.randint(0, 360))
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
}


def combine(operation, left, right):
    return ROOM_COMBINERS[operation](left, right)


# ----------------------------------------------------------
# -------- ROOM GENERATORS ---------------------------------
# ----------------------------------------------------------


def generate_automaton_room():
    while True:
        try:
            tilemap = run_automata(
                inverted=False,
                randomize_size=False,
                remove_pockets=True,
                size=random.randint(15, 30),
            )
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
    curve = scale_spheres(curve, random.randint(5, 12))
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


def generate_worley_room():
    from worley import generate_worley_cave

    spheres, path = generate_worley_cave()
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


ROOM_OPERATIONS = {
    "A": generate_automaton_room,
    "B": generate_random_walk_room,
    "C": generate_DLA_room,
    "D": generate_erosion_room,
    "E": generate_noise_room,
    "P": generate_character_room,
    "R": generate_tunnel_room,
    "U": simplified_erosion_room,
    "V": simplified_noise_room,
    "W": generate_worms_room,
    "X": generate_worms_room_big,
    "Z": generate_fancy_tunnel_room,
    "a": generate_worley_room,
}


def generate_room(room_type):
    a2, a1 = ROOM_OPERATIONS[room_type]()
    return [a2], [a1]


def shift_room_multiple(spheres, point):
    shifted_room = []
    for cluster in spheres:
        shifted_room.append(shift_room(cluster, point))
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
    if not check:
        raise MissingEntranceOrExit


class MissingEntranceOrExit(Exception):
    pass


if __name__ == "__main__":

    # generation = REFINERY for refinery and escort rooms in custom rooms
    # generation = WILD for complex rooms (chaotic preset)
    # generation = SIMPLE for simpler rooms (neutral preset)
    generation = "SIMPLE"
    if generation == "REFINERY" or generation == "REFINERY_CUSTOMROOMS":
        tags = ["small", "medium", "big"]
        num_rooms = {"small": 40, "medium": 40, "big": 40}
    else:
        tags = ["small", "medium", "big", "end"]
        num_rooms = {"small": 100, "medium": 100, "big": 100, "end": 25}
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
                    room_chance = random.random()
                    if generation == "WILD":
                        if tag == "small":
                            room = generate(
                                depth=1, n_rooms=1, rooms_list="ABCPRWZa"
                            )
                        elif tag == "medium" or tag == "end":
                            if room_chance < 0.7:
                                room = generate(
                                    depth=2, n_rooms=2, rooms_list="ABCPRUVWZa"
                                )
                            else:
                                room = generate(
                                    depth=2, n_rooms=1, rooms_list="UVW"
                                )
                        else:
                            if room_chance < 0.1:
                                room = generate(
                                    depth=2, n_rooms=1, rooms_list="DEX"
                                )
                            elif room_chance < 0.2:
                                room = generate(
                                    depth=2, n_rooms=2, rooms_list="UVWa"
                                )
                            elif room_chance < 0.6:
                                room = generate(
                                    depth=2, n_rooms=3, rooms_list="ABCPRUVWZa"
                                )
                            else:
                                room = generate(
                                    depth=2,
                                    n_rooms=2,
                                    rooms_list="ABCPRUVWXZa",
                                )
                    elif generation == "SIMPLE":
                        if tag == "small":
                            room = generate(
                                # depth=1,
                                # n_rooms=1,
                                # rooms_list="ABCPRa",
                                depth=1,
                                n_rooms=1,
                                rooms_list="a",
                            )
                        elif tag == "medium" or tag == "end":
                            if room_chance < 0.5:
                                room = generate(
                                    depth=2, n_rooms=2, rooms_list="ABCPRZa"
                                )
                            elif room_chance < 0.8:
                                room = generate(
                                    depth=2, n_rooms=1, rooms_list="ABCPRUVWZa"
                                )
                            else:
                                room = generate(
                                    depth=2, n_rooms=1, rooms_list="UVW"
                                )
                        else:
                            if room_chance < 0.1:
                                room = generate(
                                    depth=2, n_rooms=1, rooms_list="DEX"
                                )
                            elif room_chance < 0.2:
                                room = generate(
                                    depth=2, n_rooms=3, rooms_list="ABCPRZa"
                                )
                            else:
                                room = generate(
                                    depth=2,
                                    n_rooms=2,
                                    rooms_list="ABCPRUVWXZa",
                                )
                    else:
                        if tag == "small":
                            room = generate(
                                depth=1, n_rooms=1, rooms_list="ABCPRWZa"
                            )
                        elif tag == "medium" or tag == "end":
                            if room_chance < 0.5:
                                room = generate(
                                    depth=2, n_rooms=2, rooms_list="ABCPRWZa"
                                )
                            elif room_chance < 0.8:
                                room = generate(
                                    depth=2, n_rooms=1, rooms_list="ABCPRUVWZa"
                                )
                            else:
                                room = generate(
                                    depth=2, n_rooms=1, rooms_list="UVW"
                                )
                        else:
                            if room_chance < 0.3:
                                room = generate(
                                    depth=2, n_rooms=3, rooms_list="ABCPRZa"
                                )
                            else:
                                room = generate(
                                    depth=2, n_rooms=2, rooms_list="ABCPRUVWZa"
                                )

                    tree, _ = parse(room)
                    print(room, tree)
                    # Optional scaling step:
                    new_room, traversal_path = build(tree)
                    # entrances = generate_entrances(
                    #     list(flatten(new_room)),
                    #     exit_interval=(200, 280),
                    #     entrance_interval=(30, 140),
                    # )
                    entrance_order = random.random()
                    if entrance_order < 0.8:
                        entrances = generate_entrances(
                            new_room[0],
                            return_type="Entrances",
                            entrance_interval=(0, 110),
                        )
                        exits = generate_entrances(
                            new_room[-1],
                            return_type="Exits",
                            exit_interval=(200, 290),
                        )
                    else:
                        entrances = generate_entrances(
                            new_room[-1],
                            return_type="Entrances",
                            entrance_interval=(0, 110),
                        )
                        exits = generate_entrances(
                            new_room[0],
                            return_type="Exits",
                            exit_interval=(200, 310),
                        )
                    has_entrance_and_exit(entrances + exits)
                    break
                except MissingEntranceOrExit:
                    continue
            pillars = generate_pillars(list(flatten(new_room)))
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
                entrances + exits,
                pillars,
                room_num,
                room_name,
                {"bounds": 4000, "tags": room_tag},
                pe,
            )

            try:
                path = Path("generated")
                path.mkdir(parents=True, exist_ok=True)
                with open(f"generated/{name}", "w") as f:
                    json.dump(room_json, f, indent=4)
                logging.info(f"Written {name} to generated/{name}")
            except Exception as e:
                logging.error(f"Couldn't write {name}: {e}")
