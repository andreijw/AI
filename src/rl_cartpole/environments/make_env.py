"""Factory functions for creating CartPole environments."""

from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from .cartpole_env import CartPoleEnv


def make_env(
    render_mode: Optional[str] = None,
    max_episode_steps: int = 500,
    seed: Optional[int] = None,
    obs_noise_std: float = 0.0,
    action_noise_prob: float = 0.0,
    domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
) -> CartPoleEnv:
    """
    Create a single CartPole environment.

    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        max_episode_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        obs_noise_std: Standard deviation of Gaussian noise added to observations
        action_noise_prob: Probability of flipping the action (0.0 to 1.0)
        domain_randomization: Dictionary with parameter ranges for randomization.
            Supported keys: 'gravity', 'masscart', 'masspole', 'length'
            Values should be tuples of (min, max) for uniform sampling

    Returns:
        CartPoleEnv instance
    """
    return CartPoleEnv(
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        seed=seed,
        obs_noise_std=obs_noise_std,
        action_noise_prob=action_noise_prob,
        domain_randomization=domain_randomization,
    )


def make_vec_env(
    num_envs: int = 4,
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
        >>> obs = vec_env.reset()
        >>> obs.shape  # (4, 4) - 4 envs, 4 observations each

        >>> # With domain randomization
        >>> vec_env = make_vec_env(
        ...     num_envs=8,
        ...     domain_randomization={'gravity': (8.0, 12.0), 'length': (0.3, 0.7)}
        ... )
    """

    def _make_env(env_seed: int) -> Callable[[], CartPoleEnv]:
        """Create a function that returns an environment with specific seed."""

        def _init() -> CartPoleEnv:
            return make_env(
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

    Example:
        >>> config = {
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
        domain_rand = {k: tuple(v) if isinstance(v, list) else v for k, v in domain_rand.items()}

    return make_env(
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
        render_mode=config.get("render_mode"),
        max_episode_steps=config.get("max_episode_steps", 500),
        seed=config.get("seed"),
        obs_noise_std=config.get("obs_noise_std", 0.0),
        action_noise_prob=config.get("action_noise_prob", 0.0),
        domain_randomization=domain_rand,
        async_envs=config.get("async_envs", True),
    )
