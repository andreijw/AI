"""Factory functions for creating the Gymnasium lunar lander environments, discrete and continuous."""

from __future__ import annotations

from typing import Any

from rl_core.environments import env_from_config

from .lunarlander_env import LunarLanderEnv

# Each environment setting a LunarLander config may set, by its config name, with its default.
ENV_DEFAULTS: dict[str, Any] = {
    "name": "LunarLander-v3",
    "max_episode_steps": 1000,
    "obs_noise_std": 0.0,
}

# The same for LunarLanderContinuous-v3, which shares the wrapper.
CONTINUOUS_ENV_DEFAULTS: dict[str, Any] = {**ENV_DEFAULTS, "name": "LunarLanderContinuous-v3"}


def make_env_from_config(config: dict[str, Any]) -> LunarLanderEnv:
    """
    Create a LunarLander-v3 environment from a configuration dictionary.

    Raises:
        ValueError: If the config sets a setting LunarLander does not support (see ENV_DEFAULTS).
        ImportError: If Box2D is not installed.
    """
    return env_from_config(LunarLanderEnv, config, ENV_DEFAULTS)


def make_continuous_env_from_config(config: dict[str, Any]) -> LunarLanderEnv:
    """
    Create a LunarLanderContinuous-v3 environment from a configuration dictionary.

    Raises:
        ValueError: If the config sets a setting it does not support (see CONTINUOUS_ENV_DEFAULTS).
        ImportError: If Box2D is not installed.
    """
    return env_from_config(LunarLanderEnv, config, CONTINUOUS_ENV_DEFAULTS)
