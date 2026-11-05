"""Building the navigation environments from a config's ``environment`` section."""

from __future__ import annotations

from typing import Any

from rl_core.environments import OBSERVATION_DEFAULTS, env_from_config
from rl_navigation.gym_env import CONTINUOUS_ID, DISCRETE_ID, MAX_EPISODE_STEPS

from .navigation_env import LAYOUT_PARAMS, NavigationEnv

# Each environment setting a Navigation-v0 config may set, by its config name, with its default.
ENV_DEFAULTS: dict[str, Any] = {
    "name": DISCRETE_ID,
    "max_episode_steps": MAX_EPISODE_STEPS,
    "obs_noise_std": 0.0,
    **OBSERVATION_DEFAULTS,
    "action_noise_prob": 0.0,
    "domain_randomization": None,
    "sensor": "hc_sr04",
    "lidar_beams": 64,
    "num_obstacles": LAYOUT_PARAMS.num_obstacles,
}

# The same for NavigationContinuous-v0, whose action noise is Gaussian.
CONTINUOUS_ENV_DEFAULTS: dict[str, Any] = {
    **{key: value for key, value in ENV_DEFAULTS.items() if key != "action_noise_prob"},
    "name": CONTINUOUS_ID,
    "action_noise_std": 0.0,
}


def make_env_from_config(config: dict[str, Any]) -> NavigationEnv:
    """
    Create a Navigation-v0 environment (five discrete actions) from a configuration dictionary.

    Raises:
        ValueError: If the config sets a setting it does not support (see ENV_DEFAULTS), or the
            wrapper refuses a value.
    """
    return env_from_config(NavigationEnv, config, ENV_DEFAULTS)


def make_continuous_env_from_config(config: dict[str, Any]) -> NavigationEnv:
    """
    Create a NavigationContinuous-v0 environment from a configuration dictionary.

    Raises:
        ValueError: If the config sets a setting it does not support (see
            CONTINUOUS_ENV_DEFAULTS), or the wrapper refuses a value.
    """
    return env_from_config(NavigationEnv, config, CONTINUOUS_ENV_DEFAULTS)
