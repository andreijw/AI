"""Random obstacle layouts with a start pose and a goal the robot can reach.

A layout is drawn by rejection: obstacles, start and goal are drawn together and redrawn until the
start and goal are clear of every obstacle, far enough apart, and joined by a path wide enough for
the robot. :class:`LayoutParams` rejects up front settings that plainly cannot be drawn (too
dense, or start and goal unable to be far enough apart), so most impossible settings fail when
they are made rather than after the attempts run out. It is a necessary condition, not a
sufficient one: settings that pass may still exhaust ``max_attempts``.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.sim.geometry import Arena, FloatArray, clearance, occupancy_grid


class Layout(NamedTuple):
    """An arena with a start pose ``(x, y, theta)`` and a goal ``(x, y)``."""

    arena: Arena
    start: FloatArray
    goal: FloatArray


@dataclass(frozen=True)
class LayoutParams:
    """How layouts are drawn.

    Args:
        width: Arena width in metres.
        height: Arena height in metres.
        num_obstacles: Obstacles per layout.
        circle_radius: ``(low, high)`` range of circle radii in metres.
        box_size: ``(low, high)`` range of each box side in metres.
        circle_fraction: Probability that an obstacle is a circle rather than a box.
        robot_radius: The robot's radius in metres.
        margin: Clearance beyond the robot's radius that the start and goal keep from obstacles.
        min_start_goal_distance: Smallest straight-line start-to-goal distance in metres.
        max_fill: Largest share of the arena that ``num_obstacles`` of the largest size may cover.
        grid_resolution: Cell size in metres of the grid the path check searches.
        max_attempts: Draws before :func:`generate_layout` gives up.

    Raises:
        ValueError: A setting out of range, or one plainly impossible to draw: obstacles of the
            largest size covering more than ``max_fill`` of the arena, or a start-to-goal distance
            longer than the diagonal of the area the start and goal are drawn from. Passing these
            checks does not guarantee a draw succeeds.
    """

    width: float = 5.0
    height: float = 5.0
    num_obstacles: int = 6
    circle_radius: tuple[float, float] = (0.2, 0.5)
    box_size: tuple[float, float] = (0.3, 1.0)
    circle_fraction: float = 0.5
    robot_radius: float = 0.1
    margin: float = 0.1
    min_start_goal_distance: float = 2.0
    max_fill: float = 0.5
    grid_resolution: float = 0.05
    max_attempts: int = 100

    def __post_init__(self) -> None:
        for name in ("circle_radius", "box_size"):
            low, high = getattr(self, name)
            if not 0.0 < low <= high:
                raise ValueError(f"LayoutParams {name} needs 0 < low <= high, got {(low, high)}.")
        if self.num_obstacles < 0:
            raise ValueError(
                f"LayoutParams num_obstacles must be at least 0, got {self.num_obstacles}."
            )
        if not 0.0 <= self.circle_fraction <= 1.0:
            raise ValueError(
                f"LayoutParams circle_fraction must be in [0, 1], got {self.circle_fraction}."
            )
        if not self.grid_resolution > 0.0:
            raise ValueError(
                f"LayoutParams grid_resolution must be positive, got {self.grid_resolution}."
            )
        if not self.robot_radius > 0.0:
            raise ValueError(
                f"LayoutParams robot_radius must be positive, got {self.robot_radius}."
            )
        if not self.margin >= 0.0:
            raise ValueError(f"LayoutParams margin must be at least 0, got {self.margin}.")
        if self.max_attempts < 1:
            raise ValueError(
                f"LayoutParams max_attempts must be at least 1, got {self.max_attempts}."
            )
        largest = max(math.pi * self.circle_radius[1] ** 2, self.box_size[1] ** 2)
        fill = self.num_obstacles * largest / (self.width * self.height)
        if fill > self.max_fill:
            raise ValueError(
                f"LayoutParams num_obstacles={self.num_obstacles} of up to {largest:.3g} m^2 each could "
                f"cover {fill:.0%} of the {self.width} x {self.height} m arena, more than max_fill="
                f"{self.max_fill:.0%}: use fewer or smaller obstacles, or a larger arena."
            )
        gap = self.robot_radius + self.margin
        reach = math.hypot(self.width - 2 * gap, self.height - 2 * gap)
        if self.width <= 2 * gap or self.height <= 2 * gap or reach < self.min_start_goal_distance:
            raise ValueError(
                f"LayoutParams min_start_goal_distance={self.min_start_goal_distance} m cannot be met: "
                f"start and goal are drawn at least {gap} m from the walls of the {self.width} x "
                f"{self.height} m arena, at most {max(reach, 0.0):.3g} m apart."
            )


def _draw_obstacles(rng: np.random.Generator, params: LayoutParams) -> Arena:
    """Obstacles with centres anywhere in the arena; they may overlap each other and the walls."""
    is_circle = rng.random(params.num_obstacles) < params.circle_fraction
    centres = rng.uniform((0.0, 0.0), (params.width, params.height), size=(params.num_obstacles, 2))
    radii = rng.uniform(*params.circle_radius, size=params.num_obstacles)
    halves = rng.uniform(*params.box_size, size=(params.num_obstacles, 2)) / 2.0
    boxes = np.concatenate([centres - halves, centres + halves], axis=1)[~is_circle]
    return Arena(
        params.width,
        params.height,
        circle_centers=centres[is_circle],
        circle_radii=radii[is_circle],
        boxes=boxes,
    )


def path_exists(
    arena: Arena, start: ArrayLike, goal: ArrayLike, robot_radius: float, resolution: float = 0.05
) -> bool:
    """Whether a disc of ``robot_radius`` can travel from ``start`` to ``goal``.

    Searches the occupancy grid inflated by ``robot_radius`` plus half a cell (the robot's
    configuration space, made conservative: a cell centre is free only if the disc fits anywhere
    within half a cell of it) with a 4-connected flood fill from the start's cell, grown by array
    shifts. A gap with less than half a cell to spare can be refused.

    Args:
        arena: The arena.
        start: ``(2,)`` start position.
        goal: ``(2,)`` goal position.
        robot_radius: The robot's radius in metres.
        resolution: Grid cell size in metres.

    Returns:
        True if the goal's cell is reachable from the start's cell through free cells.
    """
    free = ~occupancy_grid(arena, resolution, inflate=robot_radius + resolution / 2.0)
    rows, cols = free.shape

    def cell(point: ArrayLike) -> tuple[int, int]:
        x, y = np.asarray(point, dtype=np.float64)[:2]
        return min(int(y / resolution), rows - 1), min(int(x / resolution), cols - 1)

    (si, sj), (gi, gj) = cell(start), cell(goal)
    if not (free[si, sj] and free[gi, gj]):
        return False
    reached = np.zeros_like(free)
    reached[si, sj] = True
    while not reached[gi, gj]:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= free
        if np.array_equal(grown, reached):
            return False
        reached = grown
    return True


def generate_layout(rng: np.random.Generator, params: LayoutParams | None = None) -> Layout:
    """Draw a layout whose start and goal are clear, far enough apart and joined by a path.

    Each attempt draws the obstacles, then the start position and goal uniformly from the arena
    shrunk by ``robot_radius + margin``, then the start heading uniformly from ``[-pi, pi)``.

    Args:
        rng: Generator for every draw.
        params: Layout settings; None uses the defaults.

    Returns:
        The layout.

    Raises:
        RuntimeError: No attempt of ``params.max_attempts`` met the conditions.
    """
    params = LayoutParams() if params is None else params
    gap = params.robot_radius + params.margin
    low, high = (gap, gap), (params.width - gap, params.height - gap)
    for _ in range(params.max_attempts):
        arena = _draw_obstacles(rng, params)
        start, goal = rng.uniform(low, high, size=(2, 2))
        heading = rng.uniform(-math.pi, math.pi)
        if (
            np.linalg.norm(goal - start) >= params.min_start_goal_distance
            and (clearance(arena, np.array([start, goal])) >= gap).all()
            and path_exists(arena, start, goal, params.robot_radius, params.grid_resolution)
        ):
            return Layout(arena, np.array([start[0], start[1], heading]), goal)
    raise RuntimeError(
        f"generate_layout found no layout in {params.max_attempts} attempts with {params}: the start "
        "and goal were never both clear, far enough apart and connected. Use fewer obstacles or a "
        "larger arena."
    )
