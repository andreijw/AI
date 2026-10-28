"""LunarLander-v3 and LunarLanderContinuous-v3 benchmark challenge implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rl_core.agents import BaseAgent, ContinuousRandomAgent, RandomAgent
from rl_core.challenges import (
    BenchmarkChallenge,
    ChallengeEnv,
    register_challenge,
    with_box_action_bounds,
)
from rl_lunarlander.environments import make_continuous_env_from_config, make_env_from_config
from rl_lunarlander.environments.lunarlander_env import LunarLanderEnv


@register_challenge
class LunarLanderChallenge(BenchmarkChallenge):
    """LunarLander-v3: land between the flags with four discrete engine actions (needs Box2D)."""

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {"random": RandomAgent}
    CONFIG_MAP: ClassVar[dict[str, str]] = {"random": "lunarlander_default.yaml"}

    @property
    def name(self) -> str:
        return "LunarLander-v3"

    @property
    def key(self) -> str:
        return "lunarlander"

    @property
    def solve_threshold(self) -> float | None:
        return 200.0

    @property
    def default_episodes(self) -> int:
        return 500

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> LunarLanderEnv:
        return make_env_from_config({**env_config, "render_mode": render_mode})


@register_challenge
class LunarLanderContinuousChallenge(BenchmarkChallenge):
    """LunarLanderContinuous-v3: the same landing with two throttles in [-1, 1] (needs Box2D)."""

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {"random": ContinuousRandomAgent}
    CONFIG_MAP: ClassVar[dict[str, str]] = {"random": "lunarlander_continuous_default.yaml"}

    @property
    def name(self) -> str:
        return "LunarLanderContinuous-v3"

    @property
    def key(self) -> str:
        return "lunarlander_continuous"

    @property
    def solve_threshold(self) -> float | None:
        return 200.0

    @property
    def default_episodes(self) -> int:
        return 500

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> LunarLanderEnv:
        return make_continuous_env_from_config({**env_config, "render_mode": render_mode})

    def agent_config(self, agent_config: Mapping[str, Any], env: ChallengeEnv) -> dict[str, Any]:
        """The agent's config with the environment's throttle bounds, unless it sets its own."""
        return with_box_action_bounds(agent_config, env, self.name)
