"""Factory functions for creating Gymnasium Pendulum environments with noise and domain randomization."""

from __future__ import annotations

from typing import Any

from rl_core.environments import OBSERVATION_DEFAULTS
from rl_core.utils import read_env_settings

from .pendulum_env import PendulumEnv

# Each environment setting a Pendulum config may set, by its config name, with its default.
ENV_DEFAULTS: dict[str, Any] = {
    "name": "Pendulum-v1",
    "max_episode_steps": 200,
    "obs_noise_std": 0.0,
    **OBSERVATION_DEFAULTS,
    "action_noise_std": 0.0,
    "domain_randomization": None,
}
# What make_env_from_config accepts: every setting, nothing else.
ACCEPTED_SETTINGS: dict[str, Any] = dict(ENV_DEFAULTS)


def make_env_from_config(config: dict[str, Any]) -> PendulumEnv:
    """Create a Pendulum environment from a configuration dictionary.

    Raises:
        ValueError: If the config sets a setting Pendulum does not support (see ACCEPTED_SETTINGS).
    """
    settings = read_env_settings(config, ACCEPTED_SETTINGS, "Pendulum-v1")
    return PendulumEnv(
        env_name=settings.pop("name"),
        render_mode=config.get("render_mode"),
        seed=config.get("seed"),
        **settings,
    )
