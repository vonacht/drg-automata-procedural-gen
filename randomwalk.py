import math
import logging
import random
import numpy as np
from dataclasses import dataclass
from more_itertools import flatten

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

@dataclass
class Point:
    center: tuple
    radius: int
    height: int
    angle: int

def compute_room_volume(circles) -> float:
    min_x = min(circle.center[0] - circle.radius for circle in flatten(circles))
    max_x = max(circle.center[0] + circle.radius for circle in flatten(circles))
    min_y = min(circle.center[1] - circle.radius for circle in flatten(circles))
    max_y = max(circle.center[1] + circle.radius for circle in flatten(circles))
    min_z = min(circle.center[2] - circle.height for circle in flatten(circles))
    max_z = max(circle.center[2] + circle.height for circle in flatten(circles))

    return (max_x - min_x) * (max_y - min_y) * (max_z - min_z)


def compute_room_scale_factor(circles, max_side: int) -> float:
    target_max_volume = max_side**3
    current_volume = compute_room_volume(circles)
    scale_factor = target_max_volume / current_volume
    return math.cbrt(scale_factor)

def scale_room(points, settings, center_point=None):
    scale_factor = compute_room_scale_factor(points, settings["max_side"])
    if center_point is None:
        center_point = np.array([0.0, 0.0, 0.0])
    else:
        center_point = np.array(center_point)
    rooms = []
    for ffill in points:
        current_ffill = []
        for point in ffill:
            center = np.array(point.center)
            # Scale relative to center_point
            new_center = center_point + np.array([scale_factor, scale_factor, 1]) * (
                center - center_point
            )
            new_radius = point.radius * scale_factor
            new_height = point.height * scale_factor
            current_ffill.append(Point(new_center, new_radius, new_height, point.angle))
        rooms.append(current_ffill)
    return rooms

def generate_room(settings: dict) -> list:
    
    first_room_size = random.randrange(400, 850)
    current_room = Point((0, 0, 0), first_room_size, first_room_size, 0)
    rooms = [[current_room]]
    current_line = 0
    num_rooms = settings.get("num_rooms", random.randrange(1, 5))

    match settings.get("verticality", random.choices(["up", "down", "flat"])[0]):
        case "up":
            down_verticality = 0 
            up_verticality = settings["up_room_verticality"]
        case "down":
            down_verticality = settings["down_room_verticality"]
            up_verticality = 0
        case "flat":
            down_verticality = settings["up_room_verticality"] * 0.4 
            up_verticality = settings["up_room_verticality"] * 0.4
        case _:
            print("")

    for room in range(num_rooms):
        for _ in range(random.randrange(1, 5)):
            angle = (
                random.randint(10, 50)
                if random.randint(1, 100) >= (100 - settings["angle_percent"])
                else 0
            )
            next_feature = Point(
                (
                    current_room.center[0] + 
                    random.choice([1, -1])
                    * int(
                        np.random.normal(
                            current_room.radius, 50, 1
                        )[0]
                    ),
                    current_room.center[1] + 
                    random.choice([1, -1])
                    * int(
                        np.random.normal(
                            current_room.radius, 50, 1
                        )[0]
                    ),
                    current_room.center[2] + 
                    int(
                        np.random.uniform(
                            -current_room.height,
                            current_room.height,
                            1,
                        )[0]
                    ),
                ),
                int(
                    np.random.normal(
                        random.randrange(400, 800),
                        50,
                        1,
                    )[0]
                ),
                int(
                    np.random.normal(
                        random.randrange(400, 800),
                        50,
                        1,
                    )[0]
                ),
                angle,
            )
            rooms[current_line].append(next_feature)
        # Advance line:
        angle = (
            random.randint(10, 50)
            if random.randint(1, 100) >= (100 - settings["angle_percent"])
            else 0
        )
        new_direction = random.randint(0, 180)
        sign = 1 if new_direction <= 90 else -1
        new_direction = new_direction if new_direction <= 90 else (180 - new_direction)
        new_direction = math.radians(new_direction)


        next_room = Point(
            (
                int(
                    current_room.center[0] + 
                    math.sin(new_direction) * np.random.uniform(
                        current_room.radius,
                        current_room.radius + random.randrange(1000),
                        1
                    )[0]
                ),
                int(
                    current_room.center[1] + 
                    sign * 
                    math.cos(new_direction) * 
                    np.random.uniform(
                        current_room.radius,
                        current_room.radius + random.randrange(1000),
                        1,
                    )[0]
                ),
                int(
                    np.random.uniform(
                        -down_verticality + current_room.center[2],
                        up_verticality + current_room.center[2],
                        1,
                    )[0]
                ),
            ),
            int(
                np.random.normal(
                    random.randrange(400, 800), 50, 1
                )[0]
            ),
            int(
                np.random.normal(
                    random.randrange(400, 800), 50, 1
                )[0]
            ),
            angle,
        )
        rooms[current_line].append(next_room)
        if room != num_rooms - 1:
            current_line += 1
            current_room = next_room
            rooms.append([next_room])
        else:
            current_room = next_room

    return rooms


def generate_entrances(points: list) -> list:
    # entrance_rooms = (len(points) + 1) // 2
    # exit_rooms = len(points) - entrance_rooms
    def generate_e(idx, t):
        idx_max, max_y = max(
            [
                (ii, point.center[1] + point.radius)
                for ii, point in enumerate(points[idx])
            ],
            key=lambda x: x[1],
        )
        idx_min, min_y = min(
            [
                (ii, point.center[1] - point.radius)
                for ii, point in enumerate(points[idx])
            ],
            key=lambda x: x[1],
        )
        #angle = -90 if t == "Entrance" else 90
        angle = 90
        if t == "Exit" and len(points) == 1:
            first_entrance = []
        else:
            first_entrance = [
                points[idx][idx_max].center[0],
                max_y,
                points[idx][idx_max].center[2] + 150,
                0,
                0,
                angle,
                t,
            ]
        #angle = 90 if t == "Entrance" else -90
        angle = -90
        if t == "Entrance" and len(points) == 1:
            second_entrance = []
        else:
            second_entrance = [
                points[idx][idx_min].center[0],
                min_y,
                points[idx][idx_min].center[2] + 150,
                0,
                0,
                angle,
                t,
            ]
        return [entrance for entrance in [first_entrance, second_entrance] if entrance]

    return [generate_e(i, p) for i, p in zip([0, -1], ["Entrance", "Exit"])]


def build_json(
        points: list, entrances: list, room_num: int, key: str, settings: dict
) -> dict:

    result = {"FloodFillLines": {}}
    for ii, line in enumerate(points):
        lines = {"Points": []}
        for point in line:
            lines["Points"].append(
                {
                    "Location": {
                        "X": point.center[0],
                        "Y": point.center[1],
                        "Z": point.center[2],
                    },
                    "HRange": point.radius,
                    "VRange": point.height,
                    "CeilingHeight": point.height,
                    "FloorAngle": point.angle,
                }
            )
        result["FloodFillLines"][f"FFill_{ii}"] = lines

    result["Entrances"] = {}
    for ii, entrance in enumerate(flatten(entrances)):
        result["Entrances"].update(
            {
                f"Entrance_{ii}": {
                    "Location": {
                        "X": entrance[0],
                        "Y": entrance[1],
                        "Z": entrance[2],
                    },
                    "Type": entrance[-1],
                    "Direction": {
                        "Roll": entrance[3],
                        "Pitch": entrance[4],
                        "Yaw": entrance[5],
                    },
                }
            }
        )

    result["Name"] = f"RMA_ICGEN_{key}{room_num}"
    result["Bounds"] = settings["bounds"]
    result["Tags"] = settings["tags"]

    return result
