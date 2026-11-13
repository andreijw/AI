"""Independent oracles for the planning tests, written without rl_navigation.planning: the move
rule of an 8-connected grid that never cuts a blocked corner, exhaustive search over simple paths,
Bellman-Ford over the whole grid, and the distance from a point to a square."""

import math

import numpy as np
from numpy.typing import ArrayLike

SQRT2 = math.sqrt(2.0)
MOVES = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def allowed(blocked: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether one move from ``a`` to its neighbour ``b`` is allowed: ``b`` is in the grid and
    free, and a diagonal move has both orthogonal neighbours free (``a`` itself may be blocked)."""
    rows, cols = blocked.shape
    (i, j), (k, m) = a, b
    if not (0 <= k < rows and 0 <= m < cols) or blocked[k, m]:
        return False
    if i != k and j != m:
        return not (blocked[k, j] or blocked[i, m])
    return True


def step_cost(a: tuple[int, int], b: tuple[int, int], resolution: float) -> float:
    return resolution * (SQRT2 if a[0] != b[0] and a[1] != b[1] else 1.0)


def brute_force_cost(
    blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int], resolution: float
) -> float:
    """The cheapest cost over every simple path from ``start`` to ``goal`` (inf if none), by
    exhaustive depth-first enumeration: only for grids of a dozen cells."""
    best = math.inf

    def walk(cell: tuple[int, int], cost: float, seen: set[tuple[int, int]]) -> None:
        nonlocal best
        if cell == goal:
            best = min(best, cost)
            return
        for di, dj in MOVES:
            nxt = (cell[0] + di, cell[1] + dj)
            if nxt not in seen and allowed(blocked, cell, nxt):
                seen.add(nxt)
                walk(nxt, cost + step_cost(cell, nxt, resolution), seen)
                seen.remove(nxt)

    if start == goal:
        return 0.0  # the robot already stands on its goal, even a blocked one
    if blocked[goal]:
        return math.inf
    walk(start, 0.0, {start})
    return best


def bellman_ford(blocked: np.ndarray, source: tuple[int, int], resolution: float) -> np.ndarray:
    """``(rows, cols)`` cheapest costs from ``source`` to every cell (inf where unreachable), by
    relaxing every allowed move of the whole grid until nothing changes. ``source`` may be
    blocked: it is the robot's own cell."""
    rows, cols = blocked.shape
    cost = np.full(blocked.shape, np.inf)
    cost[source] = 0.0
    padded = np.pad(blocked, 1, constant_values=True)

    def shifted(di: int, dj: int) -> np.ndarray:
        """``shifted(di, dj)[i, j]`` is ``blocked[i + di, j + dj]``, True beyond the edge."""
        return padded[1 + di : 1 + di + rows, 1 + dj : 1 + dj + cols]

    while True:
        before = cost.copy()
        for di, dj in MOVES:
            # A move from (i - di, j - dj) into (i, j).
            ok = ~blocked & ~shifted(-di, -dj)
            if di and dj:
                ok &= ~shifted(0, -dj) & ~shifted(-di, 0)
            arrive = np.full(blocked.shape, np.inf)
            src = np.pad(cost, 1, constant_values=np.inf)[
                1 - di : 1 - di + rows, 1 - dj : 1 - dj + cols
            ]
            arrive[ok] = src[ok] + resolution * (SQRT2 if di and dj else 1.0)
            # The source cell itself is passable even if blocked, as the robot stands on it.
            if blocked[source]:
                k, m = source[0] + di, source[1] + dj
                if 0 <= k < rows and 0 <= m < cols and allowed(blocked, source, (k, m)):
                    arrive[k, m] = min(arrive[k, m], resolution * (SQRT2 if di and dj else 1.0))
            np.minimum(cost, arrive, out=cost)
        if np.array_equal(before, cost):
            return cost


def distance_to_square(point: ArrayLike, lower: ArrayLike, side: float) -> float:
    """Euclidean distance from ``point`` to the closed axis-aligned square ``lower + [0, side]^2``."""
    p, lo = np.asarray(point, dtype=float), np.asarray(lower, dtype=float)
    gap = np.maximum(np.maximum(lo - p, p - (lo + side)), 0.0)
    return float(np.hypot(*gap))
