import math

def get_largest_free_sector_with_midpoint(main_circle, surrounding_circles):
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

# --- Example ---
main = (2, 2, 1)
others = [
    (1, 1, 1),   # Blocks around 0 degrees
    (3, 1, 1),   # Blocks around 90 degrees
]

mid, size, msg = get_largest_free_sector_with_midpoint(main, others)
print(msg)

if mid is not None:
    # Example: Converting mid-angle to a coordinate on the perimeter
    target_x = main[0] + main[2] * math.cos(mid)
    target_y = main[1] + main[2] * math.sin(mid)
    print(f"Point through the gap: ({target_x:.2f}, {target_y:.2f})")
