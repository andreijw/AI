"""Factory functions for creating Gymnasium environments with noise and domain randomization.

This module provides factory functions to create Gymnasium environments wrapped with
optional noise injection and domain randomization features. While the wrapper is generic,
certain features are environment-specific:
- Domain randomization is CartPole-specific (requires gravity, masscart, masspole, length attributes)
- Action noise currently only supports Discrete(2) action spaces
"""

from collections.abc import Callable
from typing import Any, Unpack

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from rl_core.environments import OBSERVATION_DEFAULTS, ObservationOptions
from rl_core.utils.config import read_env_settings
from rl_core.utils.config_values import async_envs_setting

from .cartpole_env import CartPoleEnv

# Each environment setting a CartPole config may set, by its config name, with its default.
ENV_DEFAULTS: dict[str, Any] = {
    "name": "CartPole-v1",
    "max_episode_steps": 500,
    "obs_noise_std": 0.0,
    **OBSERVATION_DEFAULTS,
    "action_noise_prob": 0.0,
    "domain_randomization": None,
}
# What both factories accept: the vectorized one also reads async_envs, and one environment section may
# serve both, so the single-environment factory accepts it and has no use for it.
ACCEPTED_SETTINGS: dict[str, Any] = {**ENV_DEFAULTS, "async_envs": True}


def make_env(
    env_name: str = "CartPole-v1",
    render_mode: str | None = None,
    max_episode_steps: int = 500,
    seed: int | None = None,
    obs_noise_std: float = 0.0,
    action_noise_prob: float = 0.0,
    domain_randomization: dict[str, tuple[float, float]] | None = None,
    **observation: Unpack[ObservationOptions],
) -> CartPoleEnv:
    """
    Create a Gymnasium environment with optional noise and domain randomization.

    Args:
        env_name: Name of the Gymnasium environment to create (default: "CartPole-v1")
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        max_episode_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        obs_noise_std: Standard deviation of Gaussian noise added to observations
        observation: The observation settings (``ObservationOptions``; see ``BaseEnvWrapper``).
        action_noise_prob: Probability of flipping the action (0.0 to 1.0)
                          Note: Only works with Discrete(2) action spaces
        domain_randomization: Dictionary with parameter ranges for randomization.
            Note: CartPole-specific. Supported keys: 'gravity', 'masscart', 'masspole', 'length'
            Values should be tuples of (min, max) for uniform sampling

    Returns:
        CartPoleEnv instance wrapping the specified Gymnasium environment

    Note:
        While env_name can be any Gymnasium environment ID, some features have requirements:
        - domain_randomization only works with CartPole environments
        - action_noise_prob only works with Discrete(2) action spaces
    """
    return CartPoleEnv(
        env_name=env_name,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        seed=seed,
        obs_noise_std=obs_noise_std,
        action_noise_prob=action_noise_prob,
        domain_randomization=domain_randomization,
        **observation,
    )


def make_vec_env(
    num_envs: int = 4,
    env_name: str = "CartPole-v1",
    render_mode: str | None = None,
    max_episode_steps: int = 500,
    seed: int | None = None,
    obs_noise_std: float = 0.0,
    action_noise_prob: float = 0.0,
    domain_randomization: dict[str, tuple[float, float]] | None = None,
    async_envs: bool = True,
    **observation: Unpack[ObservationOptions],
) -> gym.vector.VectorEnv:
    """
    Create vectorized CartPole environments for parallel training.

    Args:
        num_envs: Number of parallel environments
        env_name: Name of the Gymnasium environment to create (default: "CartPole-v1")
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        max_episode_steps: Maximum steps per episode
        seed: Base random seed (each env gets seed + i and continues its own
              sequence across automatic resets)
        obs_noise_std: Standard deviation of Gaussian noise added to observations
        observation: The observation settings (``ObservationOptions``; see ``BaseEnvWrapper``).
        action_noise_prob: Probability of flipping the action (0.0 to 1.0)
        domain_randomization: Dictionary with parameter ranges for randomization
        async_envs: If True, use AsyncVectorEnv (parallel processes).
                   If False, use SyncVectorEnv (sequential).

    Returns:
        VectorEnv instance with num_envs parallel environments

    Example:
        >>> # Create 4 parallel environments with async execution
        >>> vec_env = make_vec_env(num_envs=4, seed=42)
        >>> obs, info = vec_env.reset()
        >>> obs.shape  # (4, 4) - 4 envs, 4 observations each

        >>> # With domain randomization
        >>> vec_env = make_vec_env(
        ...     num_envs=8,
        ...     domain_randomization={'gravity': (8.0, 12.0), 'length': (0.3, 0.7)}
        ... )
    """

    def _make_env(env_seed: int | None) -> Callable[[], CartPoleEnv]:
        """Create a function that returns an environment with specific seed."""

        def _init() -> CartPoleEnv:
            return make_env(
                env_name=env_name,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                seed=env_seed,
                obs_noise_std=obs_noise_std,
                action_noise_prob=action_noise_prob,
                domain_randomization=domain_randomization,
                **observation,
            )

        return _init

    # Create list of environment factory functions
    env_fns = [_make_env(seed + i if seed is not None else None) for i in range(num_envs)]

    # Create vectorized environment
    if async_envs:
        return AsyncVectorEnv(env_fns)  # type: ignore[arg-type]
    else:
        return SyncVectorEnv(env_fns)  # type: ignore[arg-type]


def make_env_from_config(config: dict) -> CartPoleEnv:
    """
    Create a CartPole environment from a configuration dictionary.

    Args:
        config: Configuration dictionary with environment parameters

    Returns:
        CartPoleEnv instance

    Raises:
        ValueError: If domain_randomization ranges are malformed, or the config sets a setting
            CartPole does not support (see ACCEPTED_SETTINGS), or async_envs is not true, false
            or null.

    Example:
        >>> config = {
        ...     'name': 'CartPole-v1',
        ...     'render_mode': None,
        ...     'max_episode_steps': 500,
        ...     'seed': 42,
        ...     'obs_noise_std': 0.01,
        ...     'action_noise_prob': 0.05,
        ...     'domain_randomization': {
        ...         'gravity': [8.0, 12.0],
        ...         'length': [0.3, 0.7]
        ...     }
        ... }
        >>> env = make_env_from_config(config)
    """
    settings = read_env_settings(config, ACCEPTED_SETTINGS, "CartPole-v1")
    # Checked although unused here: one environment section may serve both factories.
    async_envs_setting(config, ACCEPTED_SETTINGS["async_envs"])
    del settings["async_envs"]
    # Checked here, by the wrapper's own rule, so a bad range fails before any worker starts.
    settings["domain_randomization"] = CartPoleEnv.check_setting(
        "domain_randomization", settings["domain_randomization"]
    )
    return make_env(
        env_name=settings.pop("name"),
        render_mode=config.get("render_mode"),
        seed=config.get("seed"),
        **settings,
    )


def make_vec_env_from_config(config: dict, num_envs: int = 4) -> gym.vector.VectorEnv:
    """
    Create vectorized CartPole environments from a configuration dictionary.

    Args:
        config: Configuration dictionary with environment parameters
        num_envs: Number of parallel environments

    Returns:
        VectorEnv instance

    Raises:
        ValueError: If domain_randomization ranges are malformed, or the config sets a setting
            CartPole does not support (see ACCEPTED_SETTINGS), or async_envs is not true, false
            or null.

    Example:
        >>> config = {
        ...     'name': 'CartPole-v1',
        ...     'max_episode_steps': 500,
        ...     'seed': 42,
        ...     'obs_noise_std': 0.01
        ... }
        >>> vec_env = make_vec_env_from_config(config, num_envs=8)
    """
    settings = read_env_settings(config, ACCEPTED_SETTINGS, "CartPole-v1")
    # Checked here, by the wrapper's own rule, so a bad range fails before any worker starts.
    settings["domain_randomization"] = CartPoleEnv.check_setting(
        "domain_randomization", settings["domain_randomization"]
    )
    settings["async_envs"] = async_envs_setting(config, ACCEPTED_SETTINGS["async_envs"])
    return make_vec_env(
        num_envs=num_envs,
        env_name=settings.pop("name"),
        render_mode=config.get("render_mode"),
        seed=config.get("seed"),
        **settings,
    )
