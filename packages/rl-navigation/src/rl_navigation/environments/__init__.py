"""The framework's navigation environments: the wrapper and its config factories.

Importing this package imports ``rl_core`` and registers the Gymnasium ids (through
``rl_navigation.gym_env``); the simulator alone is ``rl_navigation.sim``.
"""

from .make_env import make_continuous_env_from_config, make_env_from_config
from .navigation_env import NavigationEnv

__all__ = ["NavigationEnv", "make_continuous_env_from_config", "make_env_from_config"]
