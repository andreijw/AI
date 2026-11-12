"""Pendulum continuous control environment wrapper for reinforcement learning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Unpack

import numpy as np

from rl_core.environments import (
    POSITIVE,
    BaseEnvWrapper,
    ObservationOptions,
    checked_randomization,
)


class PendulumEnv(BaseEnvWrapper):
    """
    Wrapper around Gymnasium Pendulum environments with noise injection and domain randomization.

    Provides a standardized interface for continuous control training pipelines. Seeding,
    observation noise and episode statistics come from
    :class:`rl_core.environments.BaseEnvWrapper`; this class adds the Pendulum-specific parts.

    Feature Compatibility:
    - **Observation noise injection**: Gaussian noise added to observation [cos(th), sin(th), th_dot]
    - **Action noise injection**: Gaussian perturbation added to continuous torque action,
      clipped to the valid torque bounds [-max_torque, max_torque].
    - **Domain randomization**: Physical parameter randomization on reset for gravity ('g'),
      mass ('m'), and length ('l').
    """

    VELOCITY_INDICES = MappingProxyType({"Pendulum-v1": (2,)})
    SCHEDULABLE_SETTINGS = ("obs_noise_std", "action_noise_std", "domain_randomization")

    RANDOMIZABLE_PARAMETERS = MappingProxyType(
        {"g": "g", "gravity": "g", "m": "m", "mass": "m", "l": "l", "length": "l"}
    )

    def __init__(
        self,
        env_name: str = "Pendulum-v1",
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        domain_randomization: Mapping[str, Sequence[float] | tuple[float, float]] | None = None,
        **observation: Unpack[ObservationOptions],
    ) -> None:
        """
        Initialize the Pendulum environment wrapper.

        Args:
            env_name: Name of the Gymnasium environment (default: "Pendulum-v1").
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            max_episode_steps: Maximum steps per episode (default: 200).
            seed: Random seed. Applied on the first reset; later resets continue the sequence,
                so runs are reproducible and episodes differ from one another.
            obs_noise_std: Standard deviation of Gaussian noise added to observations.
            observation: The observation settings (``ObservationOptions``); see
                :class:`rl_core.environments.BaseEnvWrapper`.
            action_noise_std: Standard deviation of Gaussian noise added to continuous actions.
            domain_randomization: Dictionary with parameter ranges for physical randomization.
                Supported keys: 'g' / 'gravity', 'm' / 'mass', 'l' / 'length'.
                Values should be 2-element tuples/lists [min, max].
        """
        super().__init__(
            env_name, render_mode, max_episode_steps, seed, obs_noise_std, **observation
        )

        self.action_noise_std = self.check_setting("action_noise_std", action_noise_std)
        self.domain_randomization = self.check_setting("domain_randomization", domain_randomization)

        self._make_env()

    @classmethod
    def _checked_setting(cls, name: str, value: Any) -> Any:
        if name == "domain_randomization":
            return checked_randomization(
                value,
                cls.RANDOMIZABLE_PARAMETERS,
                dict.fromkeys(cls.RANDOMIZABLE_PARAMETERS.values(), POSITIVE),
            )
        return super()._checked_setting(name, value)

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization to physical environment attributes."""
        if not self.domain_randomization:
            return

        base_env = self.env.unwrapped
        for param, (min_val, max_val) in self.domain_randomization.items():
            if hasattr(base_env, param):
                sample = float(self._np_random.uniform(min_val, max_val))
                setattr(base_env, param, sample)

    def _clip_to_action_space(self, action: np.ndarray) -> np.ndarray:
        """Clip an action to the torque limits of the action space, as float32."""
        low = getattr(self.action_space, "low", -2.0)
        high = getattr(self.action_space, "high", 2.0)
        return np.clip(action, low, high).astype(np.float32)

    def _apply_action_noise(self, action: np.ndarray) -> np.ndarray:
        """Add Gaussian perturbation to continuous action, clipped to bounds."""
        if self.action_noise_std > 0:
            noise = self._np_random.normal(0, self.action_noise_std, action.shape)
            return self._clip_to_action_space(action + noise)
        return action

    def _prepare_action(self, action: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """
        The continuous torque action (scalar or 1D array) as a float32 array, after action noise.
        """
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 0:
            action_arr = np.array([action_arr.item()], dtype=np.float32)
        # Agents return unclipped samples while training: clip to the torque limit first, so that
        # action noise still perturbs an out-of-range action instead of being clipped away.
        action_arr = self._clip_to_action_space(action_arr)
        return self._apply_action_noise(action_arr)
