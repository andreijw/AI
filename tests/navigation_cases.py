"""Shared navigation test cases: free poses on a layout, a grid built from a true map, and central
differences of a vector map."""

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.mapping.grid import LIDAR_LOG_ODDS, OccupancyGrid
from rl_navigation.sim import Arena, RangeSensor, clearance, lidar, wrap_angle


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


def fd_jacobian(f, point, eps=1e-6, angle_index=2):
    """Central differences of ``f`` at ``point``, one column per input. When ``angle_index`` is
    an index, that output entry is an angle: each difference is wrapped, so a step across +-pi
    reads as the small change it is. None wraps nothing (a measurement whose entries are all
    lengths)."""
    point = np.asarray(point, dtype=np.float64)
    columns = []
    for k in range(len(point)):
        step = np.zeros(len(point))
        step[k] = eps
        diff = np.asarray(f(point + step), dtype=np.float64) - np.asarray(
            f(point - step), dtype=np.float64
        )
        if angle_index is not None:
            diff[angle_index] = wrap_angle(diff[angle_index])
        columns.append(diff / (2 * eps))
    return np.stack(columns, axis=1)


# Two circles 2 m right of and 2 m above the pose (1, 2, 0): the landmark SLAM's hand case.
TWO_LANDMARK_POSE = np.array([1.0, 2.0, 0.0])
TWO_LANDMARKS = np.array([[3.0, 2.0, 0.3], [1.0, 4.0, 0.3]])

_QUARTER_TURN = np.array([[0.0, -1.0], [1.0, 0.0]])


def slam_null_space(pose, landmark_positions):
    """``(3 + 3L, 3)`` the landmark SLAM's unobservable directions at a linearization point: a
    global translation along x and along y (the pose and every landmark centre move, the heading
    and the radii stay), and a global rotation ``(J p, 1, J m_1, 0, ..., J m_L, 0)`` with ``J`` a
    quarter turn."""
    pose = np.asarray(pose, dtype=np.float64)
    positions = np.asarray(landmark_positions, dtype=np.float64).reshape(-1, 2)
    null = np.zeros((3 + 3 * len(positions), 3))
    null[:2, :2] = np.eye(2)
    null[:2, 2] = _QUARTER_TURN @ pose[:2]
    null[2, 2] = 1.0
    for j, position in enumerate(positions):
        rows = slice(3 + 3 * j, 5 + 3 * j)
        null[rows, :2] = np.eye(2)
        null[rows, 2] = _QUARTER_TURN @ position
    return null
