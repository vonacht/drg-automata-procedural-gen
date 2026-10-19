import random
from fractions import Fraction as F

MAX_OPS = 3  # max chained operations on a single room
MAX_FREE_OPS = 2

ROOMS = {
    "j": 1,
    "C": 1,
    "D": 3,
    "E": 3,
    "U": 1,
    "V": 1,
    "P": 1,
    "R": 1,
    "W": 1,
    "X": 3,
    "Z": 1,
    "a": 1,
    # "b": 1,
    "c": 3,
    # "d": 5,
    "e": 1,
    "f": 3,
    "n": 1,
    # "m": 1,
}
# ROOMS = {
#     "j": 1,
#     "C": 1,
#     # "D": 2,
#     # "E": 2,
#     "U": 1,
#     # "V": 1,
#     "P": 1,
#     "R": 1,
#     "W": 1,
#     # "X": 2,
#     "Z": 1,
#     "a": 1,
#     # "b": 1,
#     # "c": 2,
#     # "d": 5,
#     "e": 1,
#     # "f": 2,
#     # "m": 1,
# }
# ROOMS = {
#     "p": 1,
# }

# OPERATIONS = {
#     "M": 1,
#     "Q": 1,
#     "K": 0.5,
#     "I": 0,
#     "J": 0,
#     "L": 0.5,
#     "S": 1,
#     "H": 0,
#     "G": 0,
#     "F": 2,
#     "h": 0,
#     "i": 0,
#     "l": 1,
#     "q": 0,
#     "s": 0,
#     "t": 0,
#     "u": 0,
# }

OPERATIONS = {
    "s": 0.5,
}

# OPERATIONS = {"g": 1, "h": 1, "i": 2, "j": 3}

COMBINERS = {"O": 0.8, "T": 0.1, "Y": 0.1}

# Chance that an operation is applied to a room:
OPERATION_CHANCE = 0.95
# Chance that when the volume is high enough, the tree will be split:
COMBINER_CHANCE = 0.5
# Chance that an operation is applied to a combiner:


def fr(x):
    return F(str(x))  # exact: 1.5 -> 3/2


def build_leaf_table(rooms, ops, max_ops, max_free_ops, cap):
    """volume -> list of (room, ops_tuple), outermost op first."""
    table = {}
    for room, rv in rooms.items():
        if fr(rv) > cap:  # can never fit
            continue
        frontier = [((), fr(rv), 0)]
        for depth in range(max_ops + 1):
            nxt = []
            for seq, vol, free in frontier:
                table.setdefault(vol, []).append((room, seq))
                if depth == max_ops:
                    continue
                for op, cost in ops.items():
                    c = fr(cost)
                    nv, nf = vol + c, free + (c == 0)
                    if nv <= cap and nf <= max_free_ops:
                        nxt.append(((op,) + seq, nv, nf))
            frontier = nxt
    return table


def reachable_sums(leaf_vols, cap):
    """All volumes obtainable by summing leaf volumes (<= cap)."""
    reach, stack = set(), [F(0)]
    while stack:
        x = stack.pop()
        for lv in leaf_vols:
            y = x + lv
            if y <= cap and y not in reach:
                reach.add(y)
                stack.append(y)
    return reach


def make_generator(
    rooms,
    operations,
    combiners,
    target_volume,
    operation_chance=0.7,
    combiner_chance=0.5,
    seed=None,
):
    rng = random.Random(seed)
    cap = fr(target_volume)
    leaves = build_leaf_table(rooms, operations, MAX_OPS, MAX_FREE_OPS, cap)
    index = {}  # volume -> room -> chain length -> [ops tuples]
    for vol, opts in leaves.items():
        for room, ops in opts:
            index.setdefault(vol, {}).setdefault(room, {}).setdefault(
                len(ops), []
            ).append(ops)
    feasible = reachable_sums(list(leaves), cap)
    comb_names = list(combiners)
    comb_weights = list(combiners.values())
    split_cache = {}

    def splits_for(v):
        # a <= v - a: one representative per mirrored pair
        if v not in split_cache:
            split_cache[v] = [
                a for a in feasible if 0 < a <= v - a and (v - a) in feasible
            ]
        return split_cache[v]

    def gen(v):
        leaf_opts = index.get(v)  # was leaves.get(v)
        splits = splits_for(v)
        if not leaf_opts and not splits:
            raise ValueError(f"volume {v} is not achievable")

        if splits and (not leaf_opts or rng.random() < combiner_chance):
            a = rng.choice(splits)
            c = rng.choices(comb_names, comb_weights)[0]
            left, right = gen(a), gen(v - a)
            # equal volumes: order children canonically so (x, y) == (y, x)
            if a == v - a and to_prefix(left) > to_prefix(right):
                left, right = right, left
            return (c, left, right)

        # leaf: pick a room, then a chain length, then a sequence
        room = rng.choice(list(leaf_opts))
        by_len = leaf_opts[room]
        lengths = list(by_len)
        n = rng.choices(lengths, [operation_chance**k for k in lengths])[0]
        ops = rng.choice(by_len[n])
        node = room
        for op in reversed(ops):
            node = (op, node)
        return node

    return lambda: gen(cap)


def to_prefix(node):
    if isinstance(node, str):
        return node
    return node[0] + "".join(to_prefix(c) for c in node[1:])


def generate(target_volume):
    # usage
    gen = make_generator(
        ROOMS,
        OPERATIONS,
        COMBINERS,
        target_volume,
        operation_chance=OPERATION_CHANCE,
        combiner_chance=COMBINER_CHANCE,
    )
    tree = gen()
    # print(to_prefix(tree))
    return to_prefix(tree)
