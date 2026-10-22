"""Pendulum-v1 continuous control benchmark challenge implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import gymnasium as gym

from rl_core.agents import (
    BaseAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DDPGAgent,
)
from rl_core.challenges import BenchmarkChallenge, ChallengeEnv, register_challenge
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
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "pendulum_default.yaml",
        "actor_critic": "pendulum_actor_critic.yaml",
        "ppo": "pendulum_ppo.yaml",
        "ddpg": "pendulum_ddpg.yaml",
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
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"Pendulum needs a Box action space, got {action_space}")
        config = dict(agent_config)
        config.setdefault("action_low", action_space.low)
        config.setdefault("action_high", action_space.high)
        return config
