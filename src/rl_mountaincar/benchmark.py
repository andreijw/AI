"""MountainCar-v0 benchmark challenge implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rl_core.agents import BaseAgent, DQNAgent, RandomAgent
from rl_core.challenges import BenchmarkChallenge, register_challenge
from rl_mountaincar.environments import make_env_from_config
from rl_mountaincar.environments.mountaincar_env import MountainCarEnv


@register_challenge
class MountainCarChallenge(BenchmarkChallenge):
    """
    MountainCar-v0 discrete exploration benchmark challenge.

    PPO is not offered: without reward normalisation it never reached the flag (CHANGELOG.md,
    2026-10-23).
    """

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": RandomAgent,
        "dqn": DQNAgent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "mountaincar_default.yaml",
        "dqn": "mountaincar_dqn.yaml",
    }

    @property
    def name(self) -> str:
        return "MountainCar-v0"

    @property
    def key(self) -> str:
        return "mountaincar"

    @property
    def solve_threshold(self) -> float | None:
        return -110.0

    @property
    def default_episodes(self) -> int:
        return 500

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> MountainCarEnv:
        return make_env_from_config({**env_config, "render_mode": render_mode})
