"""BipedalWalker-v3 benchmark challenge implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rl_bipedalwalker.environments import BipedalWalkerEnv, make_env_from_config
from rl_core.agents import BaseAgent, ContinuousRandomAgent, SACAgent, TD3Agent
from rl_core.challenges import (
    BenchmarkChallenge,
    ChallengeEnv,
    register_challenge,
    with_box_action_bounds,
)


@register_challenge
class BipedalWalkerChallenge(BenchmarkChallenge):
    """BipedalWalker-v3: walk to the end of the course with four joint torques (needs Box2D)."""

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": ContinuousRandomAgent,
        "sac": SACAgent,
        "td3": TD3Agent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "bipedalwalker_default.yaml",
        "sac": "bipedalwalker_sac.yaml",
        "td3": "bipedalwalker_td3.yaml",
    }

    @property
    def name(self) -> str:
        return "BipedalWalker-v3"

    @property
    def key(self) -> str:
        return "bipedalwalker"

    @property
    def solve_threshold(self) -> float | None:
        return 300.0

    @property
    def default_episodes(self) -> int:
        return 500

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> BipedalWalkerEnv:
        return make_env_from_config({**env_config, "render_mode": render_mode})

    def agent_config(self, agent_config: Mapping[str, Any], env: ChallengeEnv) -> dict[str, Any]:
        """The agent's config with the environment's torque bounds, unless it sets its own."""
        return with_box_action_bounds(agent_config, env, self.name)
