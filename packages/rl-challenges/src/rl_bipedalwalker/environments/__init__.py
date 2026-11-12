"""Bipedal walker environment module."""

from .bipedalwalker_env import BipedalWalkerEnv
from .make_env import make_env_from_config

__all__ = ["BipedalWalkerEnv", "make_env_from_config"]
