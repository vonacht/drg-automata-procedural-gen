from typing import Dict, List, Optional
import random
import math
from itertools import combinations

# ROOMS = {
#     "A": 1,
#     "B": 1,
#     "C": 1,
#     "D": 2,
#     "E": 2,
#     "U": 1,
#     "V": 1,
#     "P": 1,
#     "R": 1,
#     "W": 1,
#     "X": 2,
#     "Z": 1,
#     "a": 1,
#     "b": 1,
#     "c": 2,
#     "d": 2,
#     "e": 1,
#     "f": 2,
# }

ROOMS = {
    "g": 1,
}

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
}

# OPERATIONS = {"g": 1, "h": 1, "i": 2, "j": 3}

COMBINERS = {"O": 0.8, "T": 0.1, "Y": 0.1}

# Chance that an operation is applied to a room:
OPERATION_CHANCE = 0.7
# Chance that when the volume is high enough, the tree will be split:
COMBINER_CHANCE = 0.5
# Chance that an operation is applied to a combiner:


def get_random_single_product_combination(
    products: Dict[str, float],
    operations: Dict[str, float],
    budget: float,
    tolerance: float = 1e-4,
) -> Optional[Dict[str, any]]:
    """
    Finds a RANDOM combination consisting of EXACTLY ONE product
    and ANY NUMBER (0 or more) of operations that equal the budget.

    :param products: Dict mapping product names to costs
    :param operations: Dict mapping operation names to multipliers
    :param budget: Target budget to match
    :param tolerance: Floating point comparison tolerance
    :return: Dict with chosen product, operations list, and exact cost, or None
    """
    op_names = list(operations.keys())
    valid_combinations = []

    # 1. Evaluate every product
    for prod_name, prod_cost in products.items():
        # 2. Evaluate all possible operation combinations (from 0 operations up to ALL operations)
        for r in range(len(op_names) + 1):
            for op_combo in combinations(op_names, r):
                # Calculate total multiplier
                total_mult = 1.0
                for op in op_combo:
                    total_mult *= operations[op]

                total_cost = prod_cost * total_mult

                # Check if total cost hits the exact budget
                if math.isclose(total_cost, budget, abs_tol=tolerance):
                    valid_combinations.append(
                        {
                            "product": prod_name,
                            "product_cost": prod_cost,
                            "operations": list(op_combo),
                            "op_multipliers": [
                                operations[op] for op in op_combo
                            ],
                            "total_cost": total_cost,
                        }
                    )

    if not valid_combinations:
        return None

    # Pick a random combination from all matching solutions
    return random.choice(valid_combinations)


def choose_suitable_node(node_map: Dict, volume):
    room = random.choice([x for x, v in node_map.items() if v <= volume])
    return room, node_map[room]


def random_split(n):
    a = random.randint(1, n - 1)
    return a, n - a


def generate(target_volume):

    if target_volume == 1:
        room, _ = choose_suitable_node(ROOMS, 1)
        if random.random() < OPERATION_CHANCE:
            operation, _ = choose_suitable_node(OPERATIONS, 1)
            return f"{operation}{room}"
        return f"{room}"
    else:
        if random.random() <= COMBINER_CHANCE:
            # TODO: add possible combiner operation
            combiner_weights = [v for _, v in COMBINERS.items()]
            combiner = random.choices(list(COMBINERS), combiner_weights, k=1)[
                0
            ]
            budget_left, budget_right = random_split(target_volume)
            return f"{combiner}{generate(budget_left)}{generate(budget_right)}"
        else:
            choice = get_random_single_product_combination(
                ROOMS, OPERATIONS, target_volume
            )
            print(choice)
            return f"{''.join(choice['operations'])}{choice['product']}"


# for ii in range(10):
#     print(generate(3))
