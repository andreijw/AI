"""Shared navigation test cases: free poses on a layout, and a grid built from a true map."""

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.mapping.grid import LIDAR_LOG_ODDS, OccupancyGrid
from rl_navigation.sim import Arena, RangeSensor, clearance, lidar


def free_poses(arena: Arena, count: int, seed: int, margin: float = 0.15) -> np.ndarray:
    """Up to ``count`` poses more than ``margin`` metres from every obstacle, drawn with their own
    seeded generator over the arena shrunk by 0.2 m."""
    rng = np.random.default_rng(seed)
    low, high = (0.2, 0.2), (arena.width - 0.2, arena.height - 0.2)
    poses = np.column_stack(
        [rng.uniform(low, high, (4 * count, 2)), rng.uniform(-np.pi, np.pi, 4 * count)]
    )
    return poses[clearance(arena, poses[:, :2]) > margin][:count]


def grid_from_mask(
    mask: ArrayLike, resolution: float, sensor: RangeSensor | None = None
) -> OccupancyGrid:
    """A lidar-model grid at origin (0, 0) that believes ``mask`` fully: occupied cells at
    ``l_max``, the others at ``l_min``, every cell observed."""
    model = LIDAR_LOG_ODDS
    log_odds = np.where(np.asarray(mask, dtype=bool), model.l_max, model.l_min)
    return OccupancyGrid(
        lidar() if sensor is None else sensor, np.shape(mask), resolution, model, log_odds=log_odds
    )
