"""Acrobot-v1 benchmark challenge implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rl_acrobot.environments import make_env_from_config
from rl_acrobot.environments.acrobot_env import AcrobotEnv
from rl_core.agents import BaseAgent, DQNAgent, PPOAgent, RandomAgent
from rl_core.challenges import BenchmarkChallenge, register_challenge


@register_challenge
class AcrobotChallenge(BenchmarkChallenge):
    """Acrobot-v1 discrete swing-up benchmark challenge."""

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": RandomAgent,
        "ppo": PPOAgent,
        "dqn": DQNAgent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "acrobot_default.yaml",
        "ppo": "acrobot_ppo.yaml",
        "dqn": "acrobot_dqn.yaml",
    }

    @property
    def name(self) -> str:
        return "Acrobot-v1"

    @property
    def key(self) -> str:
        return "acrobot"

    @property
    def solve_threshold(self) -> float | None:
        return -100.0

    @property
    def default_episodes(self) -> int:
        return 500

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> AcrobotEnv:
        return make_env_from_config({**env_config, "render_mode": render_mode})
