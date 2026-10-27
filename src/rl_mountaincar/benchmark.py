"""MountainCar-v0 and MountainCarContinuous-v0 benchmark challenge implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rl_core.agents import (
    BaseAgent,
    ContinuousRandomAgent,
    DDPGAgent,
    DQNAgent,
    RandomAgent,
    TD3Agent,
)
from rl_core.challenges import (
    BenchmarkChallenge,
    ChallengeEnv,
    register_challenge,
    with_box_action_bounds,
)
from rl_mountaincar.environments import make_continuous_env_from_config, make_env_from_config
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


@register_challenge
class MountainCarContinuousChallenge(BenchmarkChallenge):
    """
    MountainCarContinuous-v0 continuous exploration benchmark challenge.

    The force costs reward and the flag pays +100, so standing still is a strong local optimum;
    its DDPG config explores with Ornstein-Uhlenbeck noise, as rl-baselines3-zoo's does.
    """

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": ContinuousRandomAgent,
        "ddpg": DDPGAgent,
        "td3": TD3Agent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "mountaincar_continuous_default.yaml",
        "ddpg": "mountaincar_continuous_ddpg.yaml",
        "td3": "mountaincar_continuous_td3.yaml",
    }

    @property
    def name(self) -> str:
        return "MountainCarContinuous-v0"

    @property
    def key(self) -> str:
        return "mountaincar_continuous"

    @property
    def solve_threshold(self) -> float | None:
        return 90.0

    @property
    def default_episodes(self) -> int:
        return 300

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> MountainCarEnv:
        return make_continuous_env_from_config({**env_config, "render_mode": render_mode})

    def agent_config(self, agent_config: Mapping[str, Any], env: ChallengeEnv) -> dict[str, Any]:
        """The agent's config with the environment's force bounds, unless it sets its own."""
        return with_box_action_bounds(agent_config, env, self.name)
