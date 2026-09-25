from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Node:
    """
    A node in the operation tree.

    kind:
        "room"     -> a base room
        "operation"-> unary operation
        "combine"  -> binary combiner

    name:
        Name of the room/operation/combiner.

    children:
        Child nodes.
    """

    kind: str
    name: str
    children: Tuple["Node", ...] = ()

    def dna(self) -> str:
        """Serialize the tree into a compact DNA string."""

        if self.kind == "room":
            return self.name

        if self.kind == "operation":
            return f"{self.name}({self.children[0].dna()})"

        if self.kind == "combine":
            return (
                f"{self.name}("
                f"{self.children[0].dna()},"
                f"{self.children[1].dna()}"
                f")"
            )

        raise ValueError(f"Unknown node type: {self.kind}")


@dataclass
class Candidate:
    node: Node
    volume: float
    rooms: int
    operations: int
    combiners: int

    @property
    def complexity(self) -> int:
        return self.operations + self.combiners

    @property
    def dna(self) -> str:
        return self.node.dna()


def generate_dna(
    target_volume: float,
    ROOMS: Sequence[str],
    OPERATIONS: Sequence[str],
    COMBINERS: Sequence[str],
    ROOM_VOLUMES: Dict[str, float],
    OPERATION_VOLUMES: Dict[str, float],
    REQUIRED_OPERATIONS: Dict[str, Sequence[str]],
    *,
    max_rooms: int = 8,
    max_operations: int = 8,
    candidates_per_size: int = 100,
    random_seed: Optional[int] = None,
    # How aggressively to prefer simpler trees when volumes are similarly good.
    complexity_weight: float = 0.001,
    # Optional custom volume calculation for combiners.
    # Default: volume_a + volume_b
    combine_volume: Optional[Callable[[str, float, float], float]] = None,
) -> str:
    """
    Generate an operation tree whose volume is as close as possible
    to target_volume.

    The search considers trees with different numbers of rooms and
    operations, so it can naturally choose between:

        room + room + room

    and:

        operation(operation(room))

    or combinations of both.

    Returns:
        DNA string representing the selected operation tree.
    """

    if target_volume <= 0:
        raise ValueError("target_volume must be greater than zero")

    if not ROOMS:
        raise ValueError("ROOMS cannot be empty")

    if not COMBINERS and max_rooms > 1:
        raise ValueError(
            "COMBINERS cannot be empty when trees may contain multiple rooms"
        )

    for room in ROOMS:
        if room not in ROOM_VOLUMES:
            raise ValueError(f"Missing volume for room: {room}")

    for operation in OPERATIONS:
        if operation not in OPERATION_VOLUMES:
            raise ValueError(
                f"Missing volume multiplier for operation: {operation}"
            )

    for room, required in REQUIRED_OPERATIONS.items():
        if room not in ROOMS:
            raise ValueError(
                f"REQUIRED_OPERATIONS contains unknown room: {room}"
            )

        for operation in required:
            if operation not in OPERATIONS:
                raise ValueError(
                    f"Required operation {operation!r} is not in OPERATIONS"
                )

    rng = random.Random(random_seed)

    if combine_volume is None:

        def combine_volume(name: str, a: float, b: float) -> float:
            return a + b

    # ------------------------------------------------------------------
    # Apply the mandatory operations to a room.
    # ------------------------------------------------------------------

    def make_room(room: str) -> Candidate:
        node = Node("room", room)
        volume = ROOM_VOLUMES[room]
        operations = 0

        # Required operations are applied whenever the room is picked.
        for operation in REQUIRED_OPERATIONS.get(room, ()):
            node = Node(
                "operation",
                operation,
                (node,),
            )

            volume *= OPERATION_VOLUMES[operation]
            operations += 1

        return Candidate(
            node=node,
            volume=volume,
            rooms=1,
            operations=operations,
            combiners=0,
        )

    # ------------------------------------------------------------------
    # Score a candidate.
    #
    # The volume error dominates. Complexity is only a tie breaker.
    # ------------------------------------------------------------------

    def score(candidate: Candidate) -> float:
        volume_error = abs(candidate.volume - target_volume)

        # Normalize the error so complexity_weight behaves reasonably
        # for both small and large target volumes.
        normalized_error = volume_error / max(abs(target_volume), 1.0)

        return normalized_error + complexity_weight * candidate.complexity

    # ------------------------------------------------------------------
    # Keep a diverse collection of candidates.
    #
    # We don't want to retain only the best candidate at each room count,
    # because that can prematurely eliminate useful intermediate trees.
    # ------------------------------------------------------------------

    pools: Dict[int, List[Candidate]] = {}

    # Initial one-room candidates.
    pools[1] = [make_room(room) for room in ROOMS]

    # Randomly expand the operation choices around a candidate.
    #
    # This is useful because exhaustive enumeration becomes enormous very
    # quickly when there are many operations.
    def mutate(candidate: Candidate) -> Candidate:
        node = candidate.node

        # Find a random unary-operation insertion point.
        # For simplicity, this implementation primarily mutates by adding
        # an operation around the complete current tree.
        if OPERATIONS and rng.random() < 0.7:
            operation = rng.choice(OPERATIONS)

            new_node = Node(
                "operation",
                operation,
                (node,),
            )

            return Candidate(
                node=new_node,
                volume=candidate.volume * OPERATION_VOLUMES[operation],
                rooms=candidate.rooms,
                operations=candidate.operations + 1,
                combiners=candidate.combiners,
            )

        return candidate

    # ------------------------------------------------------------------
    # Build trees of increasing room count.
    # ------------------------------------------------------------------

    for room_count in range(2, max_rooms + 1):
        previous = pools.get(room_count - 1, [])

        if not previous:
            break

        new_candidates: List[Candidate] = []

        # Combine existing trees with a fresh room.
        #
        # This deliberately allows asymmetric trees:
        #
        #     combine(existing_tree, new_room)
        #
        # which is enough to produce arbitrary binary trees over time.
        for candidate in previous:
            for room in ROOMS:
                room_candidate = make_room(room)

                combiner = rng.choice(COMBINERS)

                volume = combine_volume(
                    combiner,
                    candidate.volume,
                    room_candidate.volume,
                )

                node = Node(
                    "combine",
                    combiner,
                    (
                        candidate.node,
                        room_candidate.node,
                    ),
                )

                new_candidates.append(
                    Candidate(
                        node=node,
                        volume=volume,
                        rooms=room_count,
                        operations=(
                            candidate.operations + room_candidate.operations
                        ),
                        combiners=candidate.combiners + 1,
                    )
                )

        # Keep the closest candidates, but also preserve some diversity.
        new_candidates.sort(key=score)

        # Take some best candidates.
        selected = new_candidates[:candidates_per_size]

        # Add some random candidates so that the search does not become
        # completely greedy.
        remaining = new_candidates[candidates_per_size:]

        if remaining:
            sample_count = min(
                candidates_per_size // 3,
                len(remaining),
            )
            selected.extend(rng.sample(remaining, sample_count))

        pools[room_count] = selected

    # ------------------------------------------------------------------
    # Add operation variants.
    #
    # Starting from every structural tree, try repeatedly adding unary
    # operations until max_operations is reached.
    # ------------------------------------------------------------------

    all_candidates: List[Candidate] = []

    for candidates in pools.values():
        for candidate in candidates:
            current = candidate

            # Include the unmodified candidate.
            all_candidates.append(current)

            # Generate a chain of operation variants.
            while current.operations < max_operations:
                current = mutate(current)

                all_candidates.append(current)

                # Stop occasionally to preserve a variety of operation
                # counts rather than making every tree maximally deep.
                if rng.random() < 0.25:
                    break

    # ------------------------------------------------------------------
    # Pick the candidate closest to the target.
    # ------------------------------------------------------------------

    if not all_candidates:
        raise RuntimeError("Unable to generate any candidates")

    best = min(
        all_candidates,
        key=score,
    )

    return best.dna
