"""The framework's wrapper around the navigation environments: seeding, noise, domain
randomization and frame stacking for ``Navigation-v0`` and ``NavigationContinuous-v0``."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Unpack, cast

import gymnasium as gym
import numpy as np

from rl_core.environments import (
    BaseEnvWrapper,
    ObservationOptions,
    ParameterBounds,
    checked_randomization,
)
from rl_core.utils.config_values import check_positive_int, non_negative_int
from rl_navigation.gym_env import CONTINUOUS_ID, DISCRETE_ID, MAX_EPISODE_STEPS, NavigationGymEnv
from rl_navigation.sim import LayoutParams

# The layouts this wrapper's environments draw: the raw environment's default parameters, which
# set the default obstacle count and how many obstacles a layout can hold.
LAYOUT_PARAMS = LayoutParams()

# What each randomizable parameter may be: a slip or a probability is a fraction, a noise level
# and an obstacle count are not negative (the count's upper bound is the layout's capacity,
# checked by NavigationEnv's settings rule).
RANDOMIZATION_BOUNDS = MappingProxyType(
    {
        "wheel_slip": ParameterBounds(0.0, 1.0),
        "sensor_noise_std": ParameterBounds(0.0),
        "dropout_prob": ParameterBounds(0.0, 1.0),
        "num_obstacles": ParameterBounds(0.0),
    }
)


def obstacle_count(rng: np.random.Generator, low: float, high: float) -> int:
    """An obstacle count drawn uniformly from the integers ``round(low)`` to ``round(high)``.

    The range is a float range so that a curriculum can move it smoothly and a robustness sweep
    can pin it to ``(n, n)``; Python's ``round`` takes halves to the even integer.
    """
    return int(rng.integers(round(low), round(high) + 1))


class NavigationEnv(BaseEnvWrapper):
    """
    Wrapper around ``Navigation-v0`` (five discrete actions) and ``NavigationContinuous-v0`` (two
    actions in ``[-1, 1]``); see :class:`rl_navigation.gym_env.NavigationGymEnv` for the task.

    Seeding, observation noise, frame stacking and episode statistics come from
    :class:`rl_core.environments.BaseEnvWrapper`; this class adds:

    - **Action noise**: ``action_noise_std`` for the continuous id, Gaussian noise on the action
      after it is clipped to ``[-1, 1]``, then clipped again; ``action_noise_prob`` for the
      discrete id, the probability that the action is replaced by one of the other four, drawn
      uniformly. Each id refuses the other's.
    - **Domain randomization** at each reset, drawn from the wrapper's generator in the order the
      mapping gives: ``wheel_slip`` in ``[0, 1]``, ``sensor_noise_std`` and ``dropout_prob`` of
      the readings, and ``num_obstacles``, an integer drawn by :func:`obstacle_count` from a
      range no higher than the layout can hold.
    - **Sensor and arena**: ``sensor`` (``"hc_sr04"`` or ``"lidar"``), ``lidar_beams`` and the
      fixed ``num_obstacles``, passed to ``gym.make``.

    ``mask_velocity`` is refused: the observation has no velocity of the robot's state.
    """

    SCHEDULABLE_SETTINGS = (
        "obs_noise_std",
        "action_noise_std",
        "action_noise_prob",
        "domain_randomization",
    )
    RANDOMIZABLE_PARAMETERS = MappingProxyType({name: name for name in RANDOMIZATION_BOUNDS})

    def __init__(
        self,
        env_name: str = DISCRETE_ID,
        render_mode: str | None = None,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        action_noise_prob: float = 0.0,
        domain_randomization: Mapping[str, Sequence[float]] | None = None,
        sensor: str = "hc_sr04",
        lidar_beams: int = 64,
        num_obstacles: int = LAYOUT_PARAMS.num_obstacles,
        **observation: Unpack[ObservationOptions],
    ) -> None:
        """
        Args:
            env_name: ``"Navigation-v0"`` or ``"NavigationContinuous-v0"``.
            render_mode: None or ``"rgb_array"``; ``"human"`` shows the frames in a window
                through Gymnasium's HumanRendering (pygame-ce).
            max_episode_steps: Steps after which an episode is truncated (default: 400).
            seed: Random seed. Applied on the first reset; later resets continue the sequence,
                so runs are reproducible and episodes differ from one another.
            obs_noise_std: Standard deviation of Gaussian noise added to observations.
            action_noise_std: Gaussian action noise (continuous id only).
            action_noise_prob: Probability of replacing the action (discrete id only).
            domain_randomization: ``[min, max]`` ranges of ``wheel_slip``, ``sensor_noise_std``,
                ``dropout_prob`` and ``num_obstacles``.
            sensor: ``"hc_sr04"`` (five sonars) or ``"lidar"``.
            lidar_beams: Lidar beams (with ``sensor="lidar"`` only).
            num_obstacles: Obstacles per layout when it is not randomized.
            observation: The observation settings (``ObservationOptions``); see
                :class:`rl_core.environments.BaseEnvWrapper`.

        Raises:
            ValueError: An unknown id or sensor, the other id's action noise, a setting out of
                range, or more obstacles than the layout can hold.
        """
        super().__init__(
            env_name, render_mode, max_episode_steps, seed, obs_noise_std, **observation
        )
        if env_name not in (DISCRETE_ID, CONTINUOUS_ID):
            raise ValueError(
                f"NavigationEnv wraps {DISCRETE_ID} and {CONTINUOUS_ID}, got {env_name!r}"
            )
        self.action_noise_std = self.check_setting("action_noise_std", action_noise_std)
        self.action_noise_prob = self.check_setting("action_noise_prob", action_noise_prob)
        self.domain_randomization = self.check_setting("domain_randomization", domain_randomization)
        self._continuous = env_name == CONTINUOUS_ID
        foreign = "action_noise_prob" if self._continuous else "action_noise_std"
        if getattr(self, foreign):
            kind = "discrete" if self._continuous else "continuous"
            raise ValueError(
                f"{foreign} is for {kind} actions, and {env_name} has "
                f"{'continuous' if self._continuous else 'discrete'} ones: use "
                f"{'action_noise_std' if self._continuous else 'action_noise_prob'}"
            )
        self._make_env(
            sensor=sensor,
            lidar_beams=check_positive_int("lidar_beams", lidar_beams),
            num_obstacles=non_negative_int({"num_obstacles": num_obstacles}, "num_obstacles", 0),
        )

    @classmethod
    def _checked_setting(cls, name: str, value: Any) -> Any:
        """Also refuses an obstacle range whose top count the layout cannot hold, so the
        constructor, ``apply_settings``, a curriculum's start and a robustness sweep all refuse
        it before any episode is played."""
        if name == "domain_randomization":
            ranges = checked_randomization(value, cls.RANDOMIZABLE_PARAMETERS, RANDOMIZATION_BOUNDS)
            if "num_obstacles" in ranges:
                high = ranges["num_obstacles"][1]
                try:
                    dataclasses.replace(LAYOUT_PARAMS, num_obstacles=round(high))
                except ValueError as error:
                    raise ValueError(
                        f"domain_randomization num_obstacles reaches {high:g}, more than the "
                        f"layout holds: {error}"
                    ) from error
            return ranges
        return super()._checked_setting(name, value)

    @property
    def _navigation(self) -> NavigationGymEnv:
        """The raw navigation environment under any Gymnasium wrappers."""
        return cast(NavigationGymEnv, self.env.unwrapped)

    def _apply_domain_randomization(self) -> None:
        """Set this episode's draws on the raw environment, which its reset then reads."""
        for param, (low, high) in self.domain_randomization.items():
            if param == "num_obstacles":
                value: float = obstacle_count(self._np_random, low, high)
            else:
                value = float(self._np_random.uniform(low, high))
            setattr(self._navigation, param, value)

    def _prepare_action(self, action: Any) -> Any:
        """The agent's action after action noise: clipped, then perturbed and clipped again
        (continuous), or replaced by another action with ``action_noise_prob`` (discrete)."""
        if self._continuous:
            clipped = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
            if self.action_noise_std > 0:
                noise = self._np_random.normal(0.0, self.action_noise_std, clipped.shape)
                clipped = np.clip(clipped + noise, -1.0, 1.0)
            return clipped.astype(np.float32)
        chosen = int(action)
        if self.action_noise_prob > 0 and self._np_random.random() < self.action_noise_prob:
            count = int(cast(gym.spaces.Discrete, self.action_space).n)
            return (chosen + int(self._np_random.integers(1, count))) % count
        return chosen
