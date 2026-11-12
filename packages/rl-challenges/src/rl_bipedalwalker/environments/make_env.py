"""Factory function for creating the Gymnasium bipedal walker environment."""

from __future__ import annotations

from typing import Any

from rl_core.environments import OBSERVATION_DEFAULTS, env_from_config

from .bipedalwalker_env import BipedalWalkerEnv

# Each environment setting a BipedalWalker config may set, by its config name, with its default.
ENV_DEFAULTS: dict[str, Any] = {
    "name": "BipedalWalker-v3",
    "max_episode_steps": 1600,
    "obs_noise_std": 0.0,
    **OBSERVATION_DEFAULTS,
}


def make_env_from_config(config: dict[str, Any]) -> BipedalWalkerEnv:
    """
    Create a BipedalWalker-v3 environment from a configuration dictionary.

    Raises:
        ValueError: If the config sets a setting BipedalWalker does not support (see ENV_DEFAULTS),
            or ``mask_velocity``.
        ImportError: If Box2D is not installed.
    """
    return env_from_config(BipedalWalkerEnv, config, ENV_DEFAULTS)
