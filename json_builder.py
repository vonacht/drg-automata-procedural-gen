import random
import numpy as np
import math

def most_free_angle(main_circle, surrounding_circles):
    mx, my, mR = main_circle
    blocked_intervals = []

    for sx, sy, sr in surrounding_circles:
        dx = sx - mx
        dy = sy - my
        d = math.sqrt(dx**2 + dy**2)

        if d == 0:
            if sr >= mR: return None, 0, "Main circle is entirely covered."
            continue

        # Law of Cosines
        cos_alpha = (mR**2 + d**2 - sr**2) / (2 * mR * d)

        if cos_alpha <= -1:
            return None, 0, "Main circle is entirely covered."
        if cos_alpha >= 1:
            continue

        alpha = math.acos(cos_alpha)
        phi = math.atan2(dy, dx)

        # Normalize to [0, 2pi]
        start = (phi - alpha) % (2 * math.pi)
        end = (phi + alpha) % (2 * math.pi)
        
        if start > end:
            blocked_intervals.append((start, 2 * math.pi))
            blocked_intervals.append((0, end))
        else:
            blocked_intervals.append((start, end))

    if not blocked_intervals:
        return 0.0, 2 * math.pi, "Entire perimeter is free. Midpoint set to 0."

    # Merge intervals
    blocked_intervals.sort()
    merged = []
    curr_s, curr_e = blocked_intervals[0]
    for next_s, next_e in blocked_intervals[1:]:
        if next_s <= curr_e:
            curr_e = max(curr_e, next_e)
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = next_s, next_e
    merged.append((curr_s, curr_e))

    # If the first and last intervals touch the 2pi boundary, merge them
    if len(merged) > 1 and merged[0][0] == 0 and merged[-1][1] == 2 * math.pi:
        new_start = merged[-1][0]
        new_end = merged[0][1]
        merged.pop()
        merged[0] = (new_start - 2 * math.pi, new_end)

    # Find the largest gap
    max_gap = -1
    best_mid = 0
    
    for i in range(len(merged)):
        next_idx = (i + 1) % len(merged)
        gap_start = merged[i][1]
        gap_end = merged[next_idx][0]
        
        if next_idx == 0:
            gap_end += 2 * math.pi
            
        gap_size = gap_end - gap_start
        if gap_size > max_gap:
            max_gap = gap_size
            # Calculate midpoint
            best_mid = (gap_start + gap_end) / 2
            
    # Normalize midpoint back to [0, 2pi]
    best_mid = best_mid % (2 * math.pi)

    return best_mid, max_gap, f"Largest gap: {math.degrees(max_gap):.1f}°. Mid-angle: {math.degrees(best_mid):.1f}°."

def generate_PE_features(circles):
    # For PE we choose a few circles at random and put the minehead there.
    number_of_mineheads = random.randrange(1, 4) if len(circles) > 4 else 1
    chosen_circles = random.choices(range(len(circles)), k=number_of_mineheads)
    return [
        (
            circles[circle]["center"][0],
            circles[circle]["center"][1],
            circles[circle]["center"][2] + 200,
        )
        for circle in chosen_circles
    ]


def generate_entrances(circles, graph):

    spheres_w_entrance = [0]
    # First decide the entrances:
    circles_w_one_neighbor = [
        k for k in graph if len(graph[k]) == 1 and k != 0 and 0 not in graph[k]
    ]
    if not circles_w_one_neighbor:
        spheres_w_exit = [random.randrange(1, len(circles))]
    else:
        spheres_w_exit = circles_w_one_neighbor + [x for x in random.choices(range(len(circles)), k=random.randrange(1, 3)) if x not in circles_w_one_neighbor and x != 0 and 0 not in graph[x]]

    entrances = []
    for ii, t in [(spheres_w_entrance, "Entrance"), (spheres_w_exit, "Exit")]:
        for sphere_idx in ii:
            sphere_coords = (
                circles[sphere_idx]["center"][0],
                circles[sphere_idx]["center"][1],
                circles[sphere_idx]["radius"],
            )
            neighbors = graph[sphere_idx]
            neighbor_coords = [
                (
                    circles[circle]["center"][0],
                    circles[circle]["center"][1],
                    circles[circle]["radius"],
                )
                for circle in neighbors
            ]
            angle, _, _ = most_free_angle(sphere_coords, neighbor_coords)
            angle = 0 if angle is None else angle
            vector_angle = np.degrees(angle)
            entrances.append(
                (
                    circles[sphere_idx]["center"][0]
                    + circles[sphere_idx]["radius"] * math.cos(angle),
                    circles[sphere_idx]["center"][1]
                    + circles[sphere_idx]["radius"] * math.sin(angle),
                    circles[sphere_idx]["center"][2]
                    + circles[sphere_idx]["radius"] * 0.2,
                    0,
                    0,
                    vector_angle,
                    t,
                )
            )

    return entrances


def generate_pillars(circles):
    """
    Here we generate some random pillars, not too many, just for added decoration.
    """
    pillars = []
    circles_w_pillars = random.randrange(len(circles) // 2)
    for c in random.choices(circles, k=circles_w_pillars):
        if c["radius"] < 400:
            continue
        x = c["center"][0] 
        y = c["center"][1]
        pillars.append(
            {
                "Location_1": (
                    x,
                    y,
                    c["center"][2] - c["radius"] * 0.3,
                ),
                "Location_2": (
                    x,
                    y,
                    c["center"][2] + c["radius"] * 1.3,
                ),
                "Range": random.randrange(150, 200),
                "NoiseRange": random.randrange(100, 150)
            }
        )
    return pillars


def build_json(
    traversal_list, circles, entrances, pillars, room_num, key, settings, pe_features
) -> dict:

    result = {"FloodFillLines": {}}
    lines = {"Points": []}
    for ii in traversal_list:
        circle = circles[ii]
        lines["Points"].append(
            {
                "Location": {
                    "X": circle["center"][0],
                    "Y": circle["center"][1],
                    "Z": circle["center"][2],
                },
                "HRange": circle["radius"],
                "VRange": circle["radius"],
                "CeilingHeight": circle["radius"]
            }
        )
    result["FloodFillLines"]["FFill_0"] = lines

    result["Entrances"] = {}
    for ii, entrance in enumerate(entrances):
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

    if pillars:
        result["FloodFillPillars"] = {}
        for ii, p in enumerate(pillars):
            result["FloodFillPillars"].update(
                {
                    f"FloodFillPillar_{ii}": {
                        "Points": [
                            {
                                "Location": {
                                    "X": p["Location_1"][0],
                                    "Y": p["Location_1"][1],
                                    "Z": p["Location_1"][2]
                                },
                                "Range": {
                                    "Min": p["Range"],
                                    "Max": p["Range"] 
                                },
                                "NoiseRange": {
                                    "Min": p["NoiseRange"],
                                    "Max": p["NoiseRange"]
                                }   
                            },
                            {
                                "Location": {
                                        "X": p["Location_2"][0],
                                        "Y": p["Location_2"][1],
                                        "Z": p["Location_2"][2]
                                },
                                "Range": {
                                    "Min": p["Range"],
                                    "Max": p["Range"] 
                                },
                                "NoiseRange": {
                                    "Min": p["NoiseRange"],
                                    "Max": p["NoiseRange"]
                                }   
                            }
                        ]
                    }    
                }
            )

    if pe_features:
        result["PE_MiningHead"] = {}
        for ii, feature in enumerate(pe_features):
            result["PE_MiningHead"].update(
                {
                    f"MineHead_{ii}": {
                        "Location": {"X": feature[0], "Y": feature[1], "Z": feature[2]}
                    }
                }
            )

    result["Name"] = f"RMA_ICGEN_{key}{room_num}"
    result["Bounds"] = settings["bounds"]
    result["Tags"] = settings["tags"]

    return result
