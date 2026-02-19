"""Factory functions for creating Gymnasium environments with noise and domain randomization.

This module provides factory functions to create Gymnasium environments wrapped with
optional noise injection and domain randomization features. While the wrapper is generic,
certain features are environment-specific:
- Domain randomization is CartPole-specific (requires gravity, masscart, masspole, length attributes)
- Action noise currently only supports Discrete(2) action spaces
"""

from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from .cartpole_env import CartPoleEnv


def make_env(
    env_name: str = "CartPole-v1",
    render_mode: Optional[str] = None,
    max_episode_steps: int = 500,
    seed: Optional[int] = None,
    obs_noise_std: float = 0.0,
    action_noise_prob: float = 0.0,
    domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
) -> CartPoleEnv:
    """
    Create a Gymnasium environment with optional noise and domain randomization.

    Args:
        env_name: Name of the Gymnasium environment to create (default: "CartPole-v1")
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        max_episode_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        obs_noise_std: Standard deviation of Gaussian noise added to observations
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
    )


def make_vec_env(
    num_envs: int = 4,
    env_name: str = "CartPole-v1",
    render_mode: Optional[str] = None,
    max_episode_steps: int = 500,
    seed: Optional[int] = None,
    obs_noise_std: float = 0.0,
    action_noise_prob: float = 0.0,
    domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
    async_envs: bool = True,
) -> gym.vector.VectorEnv:
    """
    Create vectorized CartPole environments for parallel training.

    Args:
        num_envs: Number of parallel environments
        env_name: Name of the Gymnasium environment to create (default: "CartPole-v1")
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        max_episode_steps: Maximum steps per episode
        seed: Base random seed (each env gets seed + i)
        obs_noise_std: Standard deviation of Gaussian noise added to observations
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

    def _make_env(env_seed: Optional[int]) -> Callable[[], CartPoleEnv]:
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
            )

        return _init

    # Create list of environment factory functions
    env_fns = [_make_env(seed + i if seed is not None else None) for i in range(num_envs)]

    # Create vectorized environment
    if async_envs:
        return AsyncVectorEnv(env_fns)
    else:
        return SyncVectorEnv(env_fns)


def make_env_from_config(config: Dict) -> CartPoleEnv:
    """
    Create a CartPole environment from a configuration dictionary.

    Args:
        config: Configuration dictionary with environment parameters

    Returns:
        CartPoleEnv instance

    Raises:
        ValueError: If domain_randomization ranges are malformed.

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
    # Extract domain randomization and convert lists to tuples if needed
    domain_rand = config.get("domain_randomization")
    if domain_rand:
        validated_domain_rand = {}
        for key, value in domain_rand.items():
            # Convert list to tuple if needed
            if isinstance(value, list):
                value = tuple(value)

            # Validate that it's a 2-element tuple of numbers
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(
                    f"Domain randomization range for '{key}' must be a 2-element list/tuple "
                    f"[min, max], got: {value}"
                )

            min_val, max_val = value
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                raise ValueError(
                    f"Domain randomization range for '{key}' must contain numeric values, "
                    f"got: [{min_val}, {max_val}]"
                )

            if min_val > max_val:
                raise ValueError(
                    f"Domain randomization range for '{key}' has min > max: [{min_val}, {max_val}]"
                )

            validated_domain_rand[key] = value

        domain_rand = validated_domain_rand

    return make_env(
        env_name=config.get("name", "CartPole-v1"),
        render_mode=config.get("render_mode"),
        max_episode_steps=config.get("max_episode_steps", 500),
        seed=config.get("seed"),
        obs_noise_std=config.get("obs_noise_std", 0.0),
        action_noise_prob=config.get("action_noise_prob", 0.0),
        domain_randomization=domain_rand,
    )


def make_vec_env_from_config(config: Dict, num_envs: int = 4) -> gym.vector.VectorEnv:
    """
    Create vectorized CartPole environments from a configuration dictionary.

    Args:
        config: Configuration dictionary with environment parameters
        num_envs: Number of parallel environments

    Returns:
        VectorEnv instance

    Example:
        >>> config = {
        ...     'name': 'CartPole-v1',
        ...     'max_episode_steps': 500,
        ...     'seed': 42,
        ...     'obs_noise_std': 0.01
        ... }
        >>> vec_env = make_vec_env_from_config(config, num_envs=8)
    """
    # Extract domain randomization and convert lists to tuples if needed
    domain_rand = config.get("domain_randomization")
    if domain_rand:
        domain_rand = {k: tuple(v) if isinstance(v, list) else v for k, v in domain_rand.items()}

    return make_vec_env(
        num_envs=num_envs,
        env_name=config.get("name", "CartPole-v1"),
        render_mode=config.get("render_mode"),
        max_episode_steps=config.get("max_episode_steps", 500),
        seed=config.get("seed"),
        obs_noise_std=config.get("obs_noise_std", 0.0),
        action_noise_prob=config.get("action_noise_prob", 0.0),
        domain_randomization=domain_rand,
        async_envs=config.get("async_envs", True),
    )
