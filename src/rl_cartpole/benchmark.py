"""CartPole-v1 benchmark challenge implementation for the AI benchmark framework."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rl_cartpole.environments import make_env_from_config
from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_core.agents import (
    ActorCriticAgent,
    BaseAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
)
from rl_core.challenges import BenchmarkChallenge, register_challenge


@register_challenge
class CartPoleChallenge(BenchmarkChallenge):
    """CartPole-v1 discrete balance control benchmark challenge."""

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": RandomAgent,
        "reinforce": ReinforceAgent,
        "actor_critic": ActorCriticAgent,
        "ppo": PPOAgent,
        "dqn": DQNAgent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "cartpole_default.yaml",
        "reinforce": "cartpole_reinforce.yaml",
        "actor_critic": "cartpole_actor_critic.yaml",
        "ppo": "cartpole_ppo.yaml",
        "dqn": "cartpole_dqn.yaml",
    }

    @property
    def name(self) -> str:
        return "CartPole-v1"

    @property
    def key(self) -> str:
        return "cartpole"

    @property
    def solve_threshold(self) -> float | None:
        return 475.0

    @property
    def default_episodes(self) -> int:
        return 500

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> CartPoleEnv:
        return make_env_from_config({**env_config, "render_mode": render_mode})
