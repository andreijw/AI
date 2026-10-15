"""Pendulum continuous control environment wrapper for reinforcement learning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from rl_core.seeding import WRAPPER_STREAM, derive_rng


class PendulumEnv:
    """
    Wrapper around Gymnasium Pendulum environments with noise injection and domain randomization.

    Provides a standardized interface for continuous control training pipelines,
    including observation noise, action noise, and physics domain randomization.

    Feature Compatibility:
    - **Observation noise injection**: Gaussian noise added to observation [cos(th), sin(th), th_dot]
    - **Action noise injection**: Gaussian perturbation added to continuous torque action,
      clipped to the valid torque bounds [-max_torque, max_torque].
    - **Domain randomization**: Physical parameter randomization on reset for gravity ('g'),
      mass ('m'), and length ('l').
    """

    SUPPORTED_DR_PARAMS: dict[str, str] = {
        "g": "g",
        "gravity": "g",
        "m": "m",
        "mass": "m",
        "l": "l",
        "length": "l",
    }

    def __init__(
        self,
        env_name: str = "Pendulum-v1",
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        seed: int | None = None,
        obs_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        domain_randomization: Mapping[str, Sequence[float] | tuple[float, float]] | None = None,
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
            action_noise_std: Standard deviation of Gaussian noise added to continuous actions.
            domain_randomization: Dictionary with parameter ranges for physical randomization.
                Supported keys: 'g' / 'gravity', 'm' / 'mass', 'l' / 'length'.
                Values should be 2-element tuples/lists [min, max].
        """
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.render_mode = render_mode

        # Generator for noise and domain randomization, on its own stream so it never repeats
        # the numbers the wrapped environment draws its start states from. Created first so an
        # invalid seed is rejected before any environment is opened.
        self._np_random = derive_rng(seed, WRAPPER_STREAM)
        # The constructor seed is applied on the first reset only; see reset().
        self._has_reset = False

        # Noise validation
        if not np.isfinite(obs_noise_std) or obs_noise_std < 0.0:
            raise ValueError(
                f"obs_noise_std must be a finite, non-negative number, got {obs_noise_std!r}"
            )
        if not np.isfinite(action_noise_std) or action_noise_std < 0.0:
            raise ValueError(
                f"action_noise_std must be a finite, non-negative number, got {action_noise_std!r}"
            )

        self.obs_noise_std = float(obs_noise_std)
        self.action_noise_std = float(action_noise_std)

        # Domain randomization validation
        self.domain_randomization = self._validate_domain_randomization(domain_randomization)

        # Create base environment
        self.env = gym.make(
            env_name,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

        if seed is not None:
            self.env.action_space.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata

        self._episode_steps = 0
        self._episode_reward = 0.0

    def _validate_domain_randomization(
        self,
        domain_randomization: Mapping[str, Sequence[float] | tuple[float, float]] | None,
    ) -> dict[str, tuple[float, float]]:
        """Validate and normalize domain randomization configuration."""
        if not domain_randomization:
            return {}
        if not isinstance(domain_randomization, Mapping):
            raise ValueError("domain_randomization must be a mapping of parameter ranges.")

        validated: dict[str, tuple[float, float]] = {}
        for param, value in domain_randomization.items():
            param_lower = param.lower()
            if param_lower not in self.SUPPORTED_DR_PARAMS:
                supported = sorted(set(self.SUPPORTED_DR_PARAMS.keys()))
                raise ValueError(
                    f"Unsupported domain randomization parameter '{param}'. "
                    f"Supported parameters: {supported}."
                )

            canon_key = self.SUPPORTED_DR_PARAMS[param_lower]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                val_tuple = tuple(value)
            else:
                val_tuple = value  # type: ignore[assignment]

            if not isinstance(val_tuple, tuple) or len(val_tuple) != 2:
                raise ValueError(
                    f"Domain randomization range for '{param}' must be a 2-element sequence "
                    f"[min, max], got: {value!r}"
                )

            min_val, max_val = val_tuple
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                raise ValueError(
                    f"Domain randomization range for '{param}' must contain numeric values, "
                    f"got: [{min_val!r}, {max_val!r}]"
                )
            if not (np.isfinite(min_val) and np.isfinite(max_val)):
                raise ValueError(
                    f"Domain randomization range for '{param}' must use finite values, "
                    f"got: [{min_val!r}, {max_val!r}]"
                )
            if min_val > max_val:
                raise ValueError(
                    f"Domain randomization range for '{param}' has min > max: "
                    f"[{min_val}, {max_val}]"
                )
            if min_val <= 0:
                raise ValueError(
                    f"Physical parameter '{param}' must be strictly positive, got min: {min_val}"
                )

            validated[canon_key] = (float(min_val), float(max_val))

        return validated

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization to physical environment attributes."""
        if not self.domain_randomization:
            return

        base_env = self.env.unwrapped
        for param, (min_val, max_val) in self.domain_randomization.items():
            if hasattr(base_env, param):
                sample = float(self._np_random.uniform(min_val, max_val))
                setattr(base_env, param, sample)

    def _add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observations."""
        if self.obs_noise_std > 0:
            noise = self._np_random.normal(0, self.obs_noise_std, obs.shape)
            return (obs + noise).astype(obs.dtype)
        return obs

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

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to an initial state.

        The constructor seed is applied on the first reset only. Later resets without a seed
        continue the same random sequences, so a seeded environment gives a reproducible series
        of different episodes. Passing a seed restarts that series from the given seed.

        Args:
            seed: Optional seed that restarts the start-state and noise sequences.
            options: Optional reset options dictionary.

        Returns:
            Tuple of (observation, info dict).
        """
        self._episode_steps = 0
        self._episode_reward = 0.0

        if seed is None and not self._has_reset:
            seed = self.seed

        obs, info = self.env.reset(seed=seed, options=options)
        self._has_reset = True

        if seed is not None:
            self._np_random = derive_rng(seed, WRAPPER_STREAM)
        self._apply_domain_randomization()
        obs = self._add_observation_noise(obs)

        return obs, info

    def step(
        self,
        action: float | Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Continuous torque action (scalar or 1D array).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 0:
            action_arr = np.array([action_arr.item()], dtype=np.float32)
        # Agents return unclipped samples while training: clip to the torque limit first, so that
        # action noise still perturbs an out-of-range action instead of being clipped away.
        action_arr = self._clip_to_action_space(action_arr)

        noisy_action = self._apply_action_noise(action_arr)

        obs, reward, terminated, truncated, info = self.env.step(noisy_action)

        self._episode_steps += 1
        self._episode_reward += float(reward)

        obs = self._add_observation_noise(obs)

        if terminated or truncated:
            info["episode"] = {
                "steps": self._episode_steps,
                "reward": self._episode_reward,
            }

        return obs, float(reward), terminated, truncated, info

    def wrap_env(self, wrapper: Any) -> None:
        """Inject a Gymnasium wrapper around the inner environment."""
        self.env = wrapper

    def close(self) -> None:
        """Close the environment and cleanup resources."""
        self.env.close()

    def render(self) -> Any:
        """Render the environment."""
        return self.env.render()

    @property
    def episode_steps(self) -> int:
        """Get current episode step count."""
        return self._episode_steps

    @property
    def episode_reward(self) -> float:
        """Get current episode cumulative reward."""
        return self._episode_reward
