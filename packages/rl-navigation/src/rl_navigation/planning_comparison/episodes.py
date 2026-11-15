"""One comparison episode: a controller played from a seeded reset, and what it did.

:func:`play` resets the environment with ``seed`` (the layout comes from the environment's own
generator, so every controller sees the same layout for the same seed), steps it until it ends
or ``max_steps`` pass, and returns an :class:`EpisodeRecord`. An episode ends as the trainer's
evaluation ends one: on ``terminated`` (the goal or a collision), on ``truncated`` (the wrapper's
time limit) or after ``max_steps``.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_navigation.gym_env import MAX_EPISODE_STEPS
from rl_navigation.planning import (
    FollowerParams,
    Navigator,
    NavigatorParams,
    TrueMap,
    polyline_length,
)
from rl_navigation.planning_comparison.controllers import RESOLUTION, Controller, navigation_env
from rl_navigation.sim import Layout, clearance


@dataclass(frozen=True)
class EpisodeRecord:
    """What one episode did.

    Args:
        seed: The reset seed, which names the layout.
        success: Ended at the goal.
        collision: Ended in a collision.
        timeout: Neither: cut by the time limit.
        steps: Steps played.
        reward: The environment's reward, summed.
        time: Seconds driven (``steps * dt``).
        path_length: Metres driven by the robot's centre.
        shortest_length: The layout's shortest path in metres (:func:`shortest_length`), for SPL.
        min_clearance: The least distance in metres from the robot's centre to an obstacle or a
            wall, start and every step included.
        step_ms: Each step's control time in milliseconds, from the observation to the action,
            to the microsecond.
        ladder_level: The deepest margin-ladder level a plan used (0 full margin, 1 half, 2
            none), or None for a policy or a planner that found no path.
    """

    seed: int
    success: bool
    collision: bool
    timeout: bool
    steps: int
    reward: float
    time: float
    path_length: float
    shortest_length: float
    min_clearance: float
    step_ms: tuple[float, ...]
    ladder_level: int | None

    @property
    def plan_ms_max(self) -> float:
        """The slowest control step in milliseconds."""
        return max(self.step_ms)


def shortest_length(layout: Layout, robot_radius: float, resolution: float = RESOLUTION) -> float:
    """The length of the layout's shortest collision-free path for the robot's centre, as the
    planner finds it on the true map with no margin (inflated by the radius and half a cell):
    A*, then shortcut smoothing. The start-to-goal distance when there is none, so a success
    still counts.

    Args:
        layout: The layout.
        robot_radius: The robot's radius in metres.
        resolution: The grid's cell size in metres.
    """
    params = NavigatorParams(robot_radius=robot_radius, margin=0.0)
    # Speeds only shape the follower, which is not stepped: the path is all that is read.
    navigator = Navigator(layout.goal, params, FollowerParams(1.0, 1.0, 0.1))
    navigator.command(layout.start, TrueMap(layout.arena, resolution))
    if navigator.path is None:
        return float(np.hypot(*(layout.goal - layout.start[:2])))
    return polyline_length(navigator.path)


def play(
    env: Any,
    controller: Controller,
    seed: int,
    *,
    max_steps: int = MAX_EPISODE_STEPS,
    shortest_cache: dict[int, float] | None = None,
) -> EpisodeRecord:
    """Play one episode of ``controller`` from ``env.reset(seed=seed)``.

    Args:
        env: A :class:`~rl_navigation.gym_env.NavigationGymEnv` or a wrapper of one (any object
            with Gymnasium's ``reset`` and ``step`` over it).
        controller: What drives.
        seed: The reset seed.
        max_steps: Steps after which the episode is cut.
        shortest_cache: Shortest lengths by seed, shared by the rows that play the same layouts,
            so each is planned once; filled in here.

    Raises:
        TypeError: If ``env`` is not a navigation environment.
    """
    raw = navigation_env(env)
    observation, _ = env.reset(seed=seed)
    assert raw.world is not None and raw.layout is not None
    if shortest_cache is not None and seed in shortest_cache:
        shortest = shortest_cache[seed]
    else:
        shortest = shortest_length(raw.layout, raw.layout_params.robot_radius)
        if shortest_cache is not None:
            shortest_cache[seed] = shortest
    controller.reset(raw)
    positions = [raw.world.pose[:2].copy()]
    step_ms: list[float] = []
    reward, steps = 0.0, 0
    info: dict[str, Any] = {}
    while True:
        start = time.perf_counter()
        action = controller.act(observation)
        step_ms.append(round(1e3 * (time.perf_counter() - start), 3))
        observation, step_reward, terminated, truncated, info = env.step(action)
        reward += float(step_reward)
        steps += 1
        positions.append(raw.world.pose[:2].copy())
        if terminated or truncated or steps >= max_steps:
            break
    path = np.array(positions)
    success, collision = bool(info["is_success"]), bool(info["collision"])
    return EpisodeRecord(
        seed=seed,
        success=success,
        collision=collision,
        timeout=not success and not collision,
        steps=steps,
        reward=reward,
        time=steps * raw.dt,
        path_length=polyline_length(path),
        shortest_length=shortest,
        min_clearance=float(clearance(raw.world.arena, path).min()),
        step_ms=tuple(step_ms),
        ladder_level=controller.ladder_level,
    )
