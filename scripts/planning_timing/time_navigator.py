"""Time the navigator's control step on the navigation environment (TESTING.md ``PLAN-1b``).

Drives the raw navigation environment through layouts 10000 to 10199 (the planning item's tuning
seeds) on the true map and on an online map built from the readings already in each observation,
with each action kind at its tuned settings: continuous actions with the default ``Navigator`` and
``FollowerParams`` (margin 0.05 m, turn-in-place 25 degrees), discrete ones with margin 0.06 m and
turn-in-place 45 degrees through ``safe_discrete_action`` and ``Navigator.safety_grid``. A step is
timed from the pose to the action: for the online map, the map update, then
``Navigator.command`` (with any plan it makes), then the action mapping. The true map's grids are
built by ``Navigator.prepare`` before the first step and timed apart.

Prints, per map and action kind: the step's median, 99th percentile and maximum; and the mean
over every sliding window of 10 consecutive steps of an episode (one replan period), its 99th
percentile and its maximum. Run alone on the machine with ``OPENBLAS_NUM_THREADS=1``:

    python scripts/planning_timing/time_navigator.py
"""

import time

import numpy as np

from rl_navigation.gym_env import DISCRETE_ACTIONS, GOAL_AND_COMMAND, NavigationGymEnv
from rl_navigation.mapping import HC_SR04_LOG_ODDS, OccupancyGrid
from rl_navigation.planning import (
    TUNED_SETTINGS,
    FollowerParams,
    Navigator,
    NavigatorParams,
    OnlineMap,
    PlanningMap,
    TrueMap,
    continuous_action,
    safe_discrete_action,
)
from rl_navigation.planning_comparison import REPLAN_PERIOD, sliding_means

SEEDS = range(10000, 10200)


def run(seed: int, kind: str, actions: str) -> tuple[list[float], float]:
    """One episode's step times in ms, and the true map's preparation time (0 online)."""
    continuous = actions == "continuous"
    env = NavigationGymEnv(continuous=continuous)
    try:
        obs, _ = env.reset(seed=seed)
        assert env.layout is not None and env.world is not None
        world_map: PlanningMap
        if kind == "true":
            world_map = TrueMap(env.layout.arena, 0.05)
        else:
            world_map = OnlineMap(OccupancyGrid(env.sensor, (100, 100), 0.05, HC_SR04_LOG_ODDS))
        settings = TUNED_SETTINGS[actions]
        follower = FollowerParams(
            env.v_max, env.omega_max, env.dt, turn_in_place=settings.turn_in_place
        )
        navigator = Navigator(env.layout.goal, NavigatorParams(margin=settings.margin), follower)
        start = time.perf_counter()
        navigator.prepare(world_map)
        prepare = 1e3 * (time.perf_counter() - start) if kind == "true" else 0.0
        ticks = []
        for _ in range(400):
            start = time.perf_counter()
            pose = env.world.pose
            if isinstance(world_map, OnlineMap):
                world_map.update(pose, obs[GOAL_AND_COMMAND:] * env.sensor.max_range)
            command = navigator.command(pose, world_map)
            action: object
            if continuous:
                action = continuous_action(command, env.v_max, env.omega_max)
            else:
                grid = navigator.safety_grid(world_map)
                action = safe_discrete_action(
                    command, DISCRETE_ACTIONS, env.v_max, env.omega_max, pose, grid, env.dt
                )
            ticks.append(1e3 * (time.perf_counter() - start))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return ticks, prepare
    finally:
        env.close()


def main() -> None:
    for kind in ("true", "online"):
        for actions in TUNED_SETTINGS:
            steps: list[float] = []
            means: list[float] = []
            prepares: list[float] = []
            for seed in SEEDS:
                ticks, prepare = run(seed, kind, actions)
                steps += ticks
                means += sliding_means(ticks)
                prepares.append(prepare)
            print(
                f"{kind} {actions}: step median {np.median(steps):.2f} ms, "
                f"p99 {np.quantile(steps, 0.99):.1f}, max {max(steps):.1f}; "
                f"sliding {REPLAN_PERIOD}-step mean: median {np.median(means):.2f}, "
                f"p99 {np.quantile(means, 0.99):.1f}, max {max(means):.1f}"
                + (f"; prepare median {np.median(prepares):.1f} ms" if kind == "true" else "")
            )


if __name__ == "__main__":
    main()
