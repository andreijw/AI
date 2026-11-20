"""What drives the robot in a comparison episode: a trained policy, or the planner on a map.

A controller is reset with the raw :class:`~rl_navigation.gym_env.NavigationGymEnv` of each
episode, then gives one action per observation. :class:`PolicyController` hands the observation
to a :class:`Policy` (any object with ``select_action(observation, training=False)``, such as an
``rl_core`` agent, which this module never imports). :class:`PlannerController` runs a
:class:`~rl_navigation.planning.Navigator` on the true map (an oracle: it sees every obstacle) or
on an online log-odds map built from the readings already in the observation; both read the
robot's true pose, as the policies read the goal from it.
"""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from rl_navigation.gym_env import DISCRETE_ACTIONS, GOAL_AND_COMMAND, NavigationGymEnv
from rl_navigation.mapping import HC_SR04_LOG_ODDS, OccupancyGrid
from rl_navigation.planning import (
    DEFAULT_MAP_RANGE,
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

#: The planning grid's cell size in metres.
RESOLUTION = 0.05

MAP_SOURCES = ("true", "online")

#: Each planner row: its map source and action kind.
PLANNER_ROWS = {
    "P-true-c": ("true", "continuous"),
    "P-true-d": ("true", "discrete"),
    "P-online-c": ("online", "continuous"),
    "P-online-d": ("online", "discrete"),
}


class Policy(Protocol):
    """Anything that picks an action from an observation, greedily when ``training`` is False."""

    def select_action(self, observation: NDArray[Any], training: bool = False) -> Any: ...


class Controller(Protocol):
    """One action per observation, reset at the start of every episode."""

    def reset(self, env: NavigationGymEnv) -> None: ...

    def act(self, observation: Any) -> Any: ...

    @property
    def ladder_level(self) -> int | None: ...


def navigation_env(env: Any) -> NavigationGymEnv:
    """The raw navigation environment under ``env``: itself, its ``unwrapped`` (Gymnasium
    wrappers), or its ``env.unwrapped`` (the framework's wrapper).

    Raises:
        TypeError: If there is none.
    """
    for candidate in (
        env,
        getattr(env, "unwrapped", None),
        getattr(getattr(env, "env", None), "unwrapped", None),
    ):
        if isinstance(candidate, NavigationGymEnv):
            return candidate
    raise TypeError(f"expected a NavigationGymEnv or a wrapper of one, got {type(env).__name__}")


class PolicyController:
    """Plays ``policy`` greedily."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def reset(self, env: NavigationGymEnv) -> None:
        """Nothing to reset: the policy keeps no episode state here."""

    def act(self, observation: Any) -> Any:
        return self.policy.select_action(observation, training=False)

    @property
    def ladder_level(self) -> int | None:
        """None: a policy has no margin ladder."""
        return None


class PlannerController:
    """Plans and follows with a :class:`~rl_navigation.planning.Navigator`.

    At each reset it builds a navigator for the episode's goal with the environment's ``v_max``,
    ``omega_max``, ``dt`` and robot radius and the action kind's
    :data:`~rl_navigation.planning.TUNED_SETTINGS`, and a
    map: the episode's arena (``"true"``, prepared at every ladder level before the first step)
    or an empty HC-SR04 log-odds grid (``"online"``) that each step updates from the
    observation's readings (``observation[GOAL_AND_COMMAND:] * max_range``), so no scan is read
    beyond the one the policy would see. A step reads the true pose, updates the online map,
    asks for a command and maps it to an action: :func:`~rl_navigation.planning.continuous_action`,
    or :func:`~rl_navigation.planning.safe_discrete_action` on the navigator's safety grid.

    Args:
        map_source: ``"true"`` or ``"online"``.
        resolution: The grid's cell size in metres.
        map_range: The online map's farthest reading that marks a hit, in metres (farther ones
            only clear).

    Raises:
        ValueError: An unknown map source.
    """

    def __init__(
        self, map_source: str, resolution: float = RESOLUTION, map_range: float = DEFAULT_MAP_RANGE
    ) -> None:
        if map_source not in MAP_SOURCES:
            raise ValueError(
                f"map_source must be one of {', '.join(MAP_SOURCES)}, got {map_source!r}"
            )
        self.map_source = map_source
        self.resolution = resolution
        self.map_range = map_range
        self._env: NavigationGymEnv | None = None
        self._map: PlanningMap | None = None
        self._navigator: Navigator | None = None
        self._ladder: tuple[float, ...] = ()
        self._deepest: int | None = None

    @property
    def navigator(self) -> Navigator:
        """This episode's navigator.

        Raises:
            RuntimeError: Before the first reset.
        """
        if self._navigator is None:
            raise RuntimeError("PlannerController.reset has not been called")
        return self._navigator

    @property
    def world_map(self) -> PlanningMap:
        """This episode's map.

        Raises:
            RuntimeError: Before the first reset.
        """
        if self._map is None:
            raise RuntimeError("PlannerController.reset has not been called")
        return self._map

    @property
    def ladder_level(self) -> int | None:
        """The deepest margin-ladder level a plan of this episode used: 0 at the full margin, 1 at
        half, 2 at none; None when no plan was found."""
        return self._deepest

    def reset(self, env: NavigationGymEnv) -> None:
        if env.layout is None:
            raise RuntimeError("reset the environment before the controller")
        settings = TUNED_SETTINGS["continuous" if env.continuous else "discrete"]
        params = NavigatorParams(
            robot_radius=env.layout_params.robot_radius, margin=settings.margin
        )
        follower = FollowerParams(
            env.v_max, env.omega_max, env.dt, turn_in_place=settings.turn_in_place
        )
        world_map: PlanningMap
        if self.map_source == "true":
            world_map = TrueMap(env.layout.arena, self.resolution)
        else:
            shape = (
                round(env.layout_params.height / self.resolution),
                round(env.layout_params.width / self.resolution),
            )
            grid = OccupancyGrid(env.sensor, shape, self.resolution, HC_SR04_LOG_ODDS)
            world_map = OnlineMap(grid, self.map_range)
        self._env, self._map = env, world_map
        self._navigator = Navigator(env.layout.goal, params, follower)
        self._navigator.prepare(world_map)
        self._ladder = params.inflations(self.resolution)
        self._deepest = None

    def act(self, observation: Any) -> Any:
        env, world_map, navigator = self._env, self._map, self.navigator
        assert env is not None and env.world is not None and world_map is not None
        pose = env.world.pose
        if isinstance(world_map, OnlineMap):
            obs = np.asarray(observation, dtype=np.float64)
            if obs.shape != env.observation_space.shape:
                raise ValueError(
                    f"the online map reads the raw observation {env.observation_space.shape}, got "
                    f"{obs.shape}: play it without frame stacking or other observation settings"
                )
            world_map.update(pose, obs[GOAL_AND_COMMAND:] * env.sensor.max_range)
        command = navigator.command(pose, world_map)
        if navigator.inflation is not None:
            level = self._ladder.index(navigator.inflation)
            self._deepest = level if self._deepest is None else max(self._deepest, level)
        if env.continuous:
            return continuous_action(command, env.v_max, env.omega_max)
        grid = navigator.safety_grid(world_map)
        return safe_discrete_action(
            command, DISCRETE_ACTIONS, env.v_max, env.omega_max, pose, grid, env.dt
        )
