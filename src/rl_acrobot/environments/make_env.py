"""Factory functions for creating Gymnasium Acrobot environments."""

from __future__ import annotations

from typing import Any

from rl_core.utils.config import config_get

from .acrobot_env import AcrobotEnv

# Environment settings other challenges support and Acrobot does not: rejected when set, so that a
# config copied from CartPole or Pendulum does not silently train without its noise.
UNSUPPORTED_SETTINGS = ("action_noise_prob", "action_noise_std", "domain_randomization")


def make_env(
    env_name: str = "Acrobot-v1",
    render_mode: str | None = None,
    max_episode_steps: int = 500,
    seed: int | None = None,
    obs_noise_std: float = 0.0,
) -> AcrobotEnv:
    """
    Create an Acrobot environment with optional observation noise.

    Args:
        env_name: Name of the Gymnasium environment (default: "Acrobot-v1").
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        max_episode_steps: Maximum steps per episode (default: 500).
        seed: Random seed for reproducibility.
        obs_noise_std: Standard deviation of Gaussian observation noise.

    Returns:
        AcrobotEnv instance.
    """
    return AcrobotEnv(
        env_name=env_name,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        seed=seed,
        obs_noise_std=obs_noise_std,
    )


def make_env_from_config(config: dict[str, Any]) -> AcrobotEnv:
    """
    Create an Acrobot environment from a configuration dictionary.

    Raises:
        ValueError: If the config sets action noise or domain randomization.
    """
    for name in UNSUPPORTED_SETTINGS:
        if config.get(name):
            raise ValueError(f"Acrobot does not support {name} (got {config[name]!r}); remove it")
    return make_env(
        env_name=config_get(config, "name", "Acrobot-v1"),
        render_mode=config.get("render_mode"),
        max_episode_steps=config_get(config, "max_episode_steps", 500),
        seed=config.get("seed"),
        obs_noise_std=config_get(config, "obs_noise_std", 0.0),
    )
