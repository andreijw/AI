"""Pendulum-v1 continuous control benchmark challenge implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rl_core.agents import (
    BaseAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DDPGAgent,
    TD3Agent,
)
from rl_core.challenges import (
    BenchmarkChallenge,
    ChallengeEnv,
    register_challenge,
    with_box_action_bounds,
)
from rl_pendulum.environments import make_env_from_config
from rl_pendulum.environments.pendulum_env import PendulumEnv


@register_challenge
class PendulumChallenge(BenchmarkChallenge):
    """Pendulum-v1 continuous torque control benchmark challenge."""

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": ContinuousRandomAgent,
        "actor_critic": ContinuousActorCriticAgent,
        "ppo": ContinuousPPOAgent,
        "ddpg": DDPGAgent,
        "td3": TD3Agent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "pendulum_default.yaml",
        "actor_critic": "pendulum_actor_critic.yaml",
        "ppo": "pendulum_ppo.yaml",
        "ddpg": "pendulum_ddpg.yaml",
        "td3": "pendulum_td3.yaml",
    }

    @property
    def name(self) -> str:
        return "Pendulum-v1"

    @property
    def key(self) -> str:
        return "pendulum"

    @property
    def solve_threshold(self) -> float | None:
        return -200.0

    @property
    def default_episodes(self) -> int:
        return 200

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> PendulumEnv:
        return make_env_from_config({**env_config, "render_mode": render_mode})

    def agent_config(self, agent_config: Mapping[str, Any], env: ChallengeEnv) -> dict[str, Any]:
        """The agent's config with the environment's torque bounds, unless it sets its own."""
        return with_box_action_bounds(agent_config, env, self.name)
