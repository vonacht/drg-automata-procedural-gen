import random
from fractions import Fraction as F

MAX_OPS = 3  # max chained operations on a single room
MAX_FREE_OPS = 2

ROOMS = {
    "j": 1,
    "C": 1,
    "D": 2,
    "E": 2,
    "U": 1,
    "V": 1,
    "P": 1,
    "R": 1,
    "W": 1,
    "X": 2,
    "Z": 1,
    "a": 1,
    "b": 1,
    "c": 2,
    "d": 5,
    "e": 1,
    "f": 2,
    "m": 0.5,
}
# ROOMS = {
#     "k": 1,
# }

OPERATIONS = {
    "M": 1.5,
    "Q": 1.5,
    "K": 1.5,
    "I": 1,
    "J": 1,
    "L": 1.5,
    "S": 1.5,
    "H": 1,
    "G": 1,
    "F": 1.5,
    "h": 1,
    "i": 1,
    "l": 1.5,
}

# }
# OPERATIONS = {
#     "l": 1,
# }

# OPERATIONS = {"g": 1, "h": 1, "i": 2, "j": 3}

COMBINERS = {"O": 0.8, "T": 0.1, "Y": 0.1}

# Chance that an operation is applied to a room:
OPERATION_CHANCE = 0.7
# Chance that when the volume is high enough, the tree will be split:
COMBINER_CHANCE = 0.5
# Chance that an operation is applied to a combiner:


def fr(x):
    return F(str(x))  # exact: 1.5 -> 3/2


def build_leaf_table(rooms, ops, max_ops, max_free_ops, cap):
    """volume -> list of (room, ops_tuple), outermost op first."""
    table = {}
    for room, rv in rooms.items():
        frontier = [((), fr(rv), 0)]  # (ops, volume, free_ops_used)
        for depth in range(max_ops + 1):
            nxt = []
            for seq, vol, free in frontier:
                table.setdefault(vol, []).append((room, seq))
                if depth == max_ops:
                    continue
                for op, mult in ops.items():
                    m = fr(mult)
                    nv, nf = vol * m, free + (m == 1)
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
        leaf_opts = leaves.get(v)
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

        w = [operation_chance ** len(ops) for _, ops in leaf_opts]
        room, ops = rng.choices(leaf_opts, w)[0]
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
