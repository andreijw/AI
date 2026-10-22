"""Factory functions for creating Gymnasium Acrobot environments."""

from __future__ import annotations

from typing import Any

from rl_core.utils.config import read_env_settings

from .acrobot_env import AcrobotEnv

# Each environment setting an Acrobot config may set, by its config name, with its default.
ENV_DEFAULTS: dict[str, Any] = {
    "name": "Acrobot-v1",
    "max_episode_steps": 500,
    "obs_noise_std": 0.0,
}


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
        ValueError: If the config sets a setting Acrobot does not support (see ENV_DEFAULTS).
    """
    settings = read_env_settings(config, ENV_DEFAULTS, "Acrobot-v1")
    return make_env(
        env_name=settings.pop("name"),
        render_mode=config.get("render_mode"),
        seed=config.get("seed"),
        **settings,
    )
