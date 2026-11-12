"""CartPole discrete control challenge package."""

from .environments import CartPoleEnv, make_env, make_vec_env

#: The version of the whole rl-challenges distribution (all six challenge packages).
__version__ = "0.1.0"

__all__ = [
    "CartPoleEnv",
    "make_env",
    "make_vec_env",
]
