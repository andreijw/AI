"""Time the planning core on the navigation layouts (TESTING.md ``PLAN-1``): A*, the reachability
flood fill that answers "no path", shortcut smoothing and the built map's dilation.

On the default 5 x 5 m layouts drawn from seeds ``--first-seed`` to ``--first-seed + --count - 1``
(default 10000 to 10199, the planning item's tuning seeds; ``generate_layout`` with
``default_rng(seed)``, the layouts ``NavigationGymEnv.reset(seed=seed)`` draws): ``astar`` on the
true map inflated by the robot's radius, half a cell and a 0.08 m margin, or without the margin
where ``reachable`` finds no path with it, timing only searches the check passes; ``shortcut`` of
each path; and, on a 100 x 100 grid with a free cell walled in, ``reachable`` and a failed
``astar`` that expands every outer cell; and ``from_occupied`` of a 2 % occupied grid. Prints the
milliseconds as the row's one-line command did. Run alone on the machine with
``OPENBLAS_NUM_THREADS=1``:

    python scripts/planning_timing/time_planner.py [--first-seed 10000] [--count 200]
"""

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np

from rl_navigation.planning import (
    PlanningGrid,
    astar,
    from_arena,
    from_occupied,
    reachable,
    shortcut,
)
from rl_navigation.sim import LayoutParams, generate_layout

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # run_provenance.py, at the repository root
import run_provenance  # noqa: E402


def timed(f: Callable[[], object]) -> float:
    """Seconds one call of ``f`` takes."""
    start = time.perf_counter()
    f()
    return time.perf_counter() - start


def ms(times: Sequence[float], q: float) -> float:
    """The ``q`` quantile of ``times`` in milliseconds, to two decimals."""
    return round(1e3 * float(np.quantile(times, q)), 2)


def parse_seeds(argv: Sequence[str] | None = None) -> range:
    """The layout seeds the command line names (default 10000 to 10199)."""
    parser = argparse.ArgumentParser(description="Time A*, reachability and smoothing")
    parser.add_argument("--first-seed", type=int, default=10000, help="First layout seed")
    parser.add_argument("--count", type=int, default=200, help="Layouts (default: 200)")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be a positive integer")
    return range(args.first_seed, args.first_seed + args.count)


def main(argv: Sequence[str] | None = None) -> None:
    seeds = parse_seeds(argv)
    print(
        f"commit {run_provenance.commit(ROOT)}; working tree {run_provenance.working_tree(ROOT)}; "
        f"seeds {seeds.start} to {seeds.stop - 1}",
        flush=True,
    )
    p = LayoutParams()
    r = p.grid_resolution
    i0 = p.robot_radius + r / 2
    layouts = [generate_layout(np.random.default_rng(s), p) for s in seeds]
    grids = [
        next(
            g
            for g in (from_arena(layout.arena, r, i) for i in (i0 + 0.08, i0))
            if reachable(g, g.cell_of(layout.start[:2]), g.cell_of(layout.goal))
        )
        for layout in layouts
    ]
    cases = [
        (g, g.cell_of(layout.start[:2]), g.cell_of(layout.goal))
        for g, layout in zip(grids, layouts, strict=True)
    ]

    def search(case: tuple[PlanningGrid, tuple[int, int], tuple[int, int]]) -> Callable[[], object]:
        return lambda: astar(*case)

    def smooth(grid: PlanningGrid, points: np.ndarray) -> Callable[[], object]:
        return lambda: shortcut(grid, points)

    searches = [timed(search(c)) for c in cases]
    paths = []
    for c, layout in zip(cases, layouts, strict=True):
        path = astar(*c)
        assert path is not None
        paths.append(np.vstack([layout.start[:2], c[0].centres(path.cells[1:-1]), layout.goal]))
    smoothing = [timed(smooth(c[0], q)) for c, q in zip(cases, paths, strict=True)]
    walled = np.zeros((100, 100), bool)
    walled[30:70, 30:70] = True
    walled[35:65, 35:65] = False
    grid = PlanningGrid(walled, r)
    fills = [timed(lambda: reachable(grid, (2, 2), (50, 50))) for _ in range(50)]
    failed = [timed(lambda: astar(grid, (2, 2), (50, 50))) for _ in range(5)]
    occupied = np.random.default_rng(0).random((100, 100)) < 0.02
    dilations = [timed(lambda: from_occupied(occupied, r, i0 + 0.08)) for _ in range(50)]
    print(
        "astar", ms(searches, 0.5), ms(searches, 0.99), ms(searches, 1),
        "shortcut", ms(smoothing, 0.5), ms(smoothing, 1),
        "no-path reachable", ms(fills, 0.5), ms(fills, 1),
        "no-path astar", ms(failed, 0.5),
        "from_occupied", ms(dilations, 0.5),
    )  # fmt: skip


if __name__ == "__main__":
    main()
