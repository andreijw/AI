"""Factory functions for creating Gymnasium environments with noise and domain randomization.

This module provides factory functions to create Gymnasium environments wrapped with
optional noise injection and domain randomization features. While the wrapper is generic,
certain features are environment-specific:
- Domain randomization is CartPole-specific (requires gravity, masscart, masspole, length attributes)
- Action noise currently only supports Discrete(2) action spaces
"""

from typing import Any

from rl_core.environments import OBSERVATION_DEFAULTS
from rl_core.utils import read_env_settings

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
# What make_env_from_config accepts: every setting, nothing else.
ACCEPTED_SETTINGS: dict[str, Any] = dict(ENV_DEFAULTS)


def make_env_from_config(config: dict) -> CartPoleEnv:
    """
    Create a CartPole environment from a configuration dictionary.

    Args:
        config: Configuration dictionary with environment parameters

    Returns:
        CartPoleEnv instance

    Raises:
        ValueError: If domain_randomization ranges are malformed, or the config sets a setting
            CartPole does not support (see ACCEPTED_SETTINGS).

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
    # Checked here, by the wrapper's own rule, so a bad range fails before any worker starts.
    settings["domain_randomization"] = CartPoleEnv.check_setting(
        "domain_randomization", settings["domain_randomization"]
    )
    return CartPoleEnv(
        env_name=settings.pop("name"),
        render_mode=config.get("render_mode"),
        seed=config.get("seed"),
        **settings,
    )
