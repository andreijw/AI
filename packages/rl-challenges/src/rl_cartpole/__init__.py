"""CartPole discrete control challenge package."""

from .environments import CartPoleEnv

#: The version of the whole rl-challenges distribution (all six challenge packages).
__version__ = "0.3.0"

__all__ = [
    "CartPoleEnv",
]
