"""Factory functions for creating Gymnasium Pendulum environments with noise and domain randomization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from rl_core.utils.config import read_env_settings

from .pendulum_env import PendulumEnv

# Each environment setting a Pendulum config may set, by its config name, with its default.
ENV_DEFAULTS: dict[str, Any] = {
    "name": "Pendulum-v1",
    "max_episode_steps": 200,
    "obs_noise_std": 0.0,
    "action_noise_std": 0.0,
    "domain_randomization": None,
}
# What both factories accept: the vectorized one also reads async_envs, and one environment section may
# serve both, so the single-environment factory accepts it and has no use for it.
ACCEPTED_SETTINGS: dict[str, Any] = {**ENV_DEFAULTS, "async_envs": True}


def make_env(
    env_name: str = "Pendulum-v1",
    render_mode: str | None = None,
    max_episode_steps: int = 200,
    seed: int | None = None,
    obs_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    domain_randomization: dict[str, tuple[float, float]] | None = None,
) -> PendulumEnv:
    """
    Create a Pendulum environment with optional continuous noise and domain randomization.

    Args:
        env_name: Name of the Gymnasium environment (default: "Pendulum-v1").
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        max_episode_steps: Maximum steps per episode (default: 200).
        seed: Random seed for reproducibility.
        obs_noise_std: Standard deviation of Gaussian observation noise.
        action_noise_std: Standard deviation of Gaussian action noise.
        domain_randomization: Physical parameter randomization ranges.

    Returns:
        PendulumEnv instance.
    """
    return PendulumEnv(
        env_name=env_name,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        seed=seed,
        obs_noise_std=obs_noise_std,
        action_noise_std=action_noise_std,
        domain_randomization=domain_randomization,
    )


def make_vec_env(
    num_envs: int = 4,
    env_name: str = "Pendulum-v1",
    render_mode: str | None = None,
    max_episode_steps: int = 200,
    seed: int | None = None,
    obs_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    domain_randomization: dict[str, tuple[float, float]] | None = None,
    async_envs: bool = True,
) -> gym.vector.VectorEnv:
    """
    Create vectorized Pendulum environments for parallel rollouts.

    Args:
        num_envs: Number of parallel environments.
        env_name: Name of Gymnasium environment.
        render_mode: Rendering mode.
        max_episode_steps: Maximum steps per episode.
        seed: Base random seed.
        obs_noise_std: Gaussian observation noise std.
        action_noise_std: Gaussian action noise std.
        domain_randomization: Physical parameter ranges.
        async_envs: If True, use AsyncVectorEnv; if False, use SyncVectorEnv.

    Returns:
        VectorEnv instance with num_envs parallel environments.
    """

    def _make_env(env_seed: int | None) -> Callable[[], PendulumEnv]:
        def _init() -> PendulumEnv:
            return make_env(
                env_name=env_name,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                seed=env_seed,
                obs_noise_std=obs_noise_std,
                action_noise_std=action_noise_std,
                domain_randomization=domain_randomization,
            )

        return _init

    env_fns = [_make_env(seed + i if seed is not None else None) for i in range(num_envs)]
    if async_envs:
        return AsyncVectorEnv(env_fns)  # type: ignore[arg-type]
    return SyncVectorEnv(env_fns)  # type: ignore[arg-type]


def make_env_from_config(config: dict[str, Any]) -> PendulumEnv:
    """Create a Pendulum environment from a configuration dictionary.

    Raises:
        ValueError: If the config sets a setting Pendulum does not support (see ACCEPTED_SETTINGS).
    """
    settings = read_env_settings(config, ACCEPTED_SETTINGS, "Pendulum-v1")
    del settings["async_envs"]
    return make_env(
        env_name=settings.pop("name"),
        render_mode=config.get("render_mode"),
        seed=config.get("seed"),
        **settings,
    )


def make_vec_env_from_config(config: dict[str, Any], num_envs: int = 4) -> gym.vector.VectorEnv:
    """Create vectorized Pendulum environments from a configuration dictionary.

    Raises:
        ValueError: If the config sets a setting Pendulum does not support (see ACCEPTED_SETTINGS).
    """
    settings = read_env_settings(config, ACCEPTED_SETTINGS, "Pendulum-v1")
    settings["async_envs"] = config.get("async_envs", True)  # null means synchronous, as before
    return make_vec_env(
        num_envs=num_envs,
        env_name=settings.pop("name"),
        render_mode=config.get("render_mode"),
        seed=config.get("seed"),
        **settings,
    )
