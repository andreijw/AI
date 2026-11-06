"""Navigation-v0 and NavigationContinuous-v0 benchmark challenges: reach a goal through random
obstacles with range sensors (``rl_navigation.gym_env``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from rl_core.agents import (
    BaseAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    TD3Agent,
)
from rl_core.challenges import (
    BenchmarkChallenge,
    ChallengeEnv,
    register_challenge,
    with_box_action_bounds,
)
from rl_navigation.environments import make_continuous_env_from_config, make_env_from_config
from rl_navigation.environments.navigation_env import NavigationEnv
from rl_navigation.gym_env import CONTINUOUS_ID, DISCRETE_ID

# Episodes for a config without a budget: 40,000 steps of 400-step episodes, the committed
# random-agent budget.
DEFAULT_EPISODES = 100


@register_challenge
class NavigationChallenge(BenchmarkChallenge):
    """Navigation-v0: drive to the goal with five discrete actions. No solve threshold yet."""

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": RandomAgent,
        "ppo": PPOAgent,
        "dqn": DQNAgent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "navigation_default.yaml",
        "ppo": "navigation_ppo.yaml",
        "dqn": "navigation_dqn.yaml",
    }

    @property
    def name(self) -> str:
        return DISCRETE_ID

    @property
    def key(self) -> str:
        return "navigation"

    @property
    def solve_threshold(self) -> float | None:
        return None

    @property
    def default_episodes(self) -> int:
        return DEFAULT_EPISODES

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> NavigationEnv:
        return make_env_from_config({**env_config, "render_mode": render_mode})


@register_challenge
class NavigationContinuousChallenge(BenchmarkChallenge):
    """NavigationContinuous-v0: the same task with speed and turn rate in [-1, 1]. No solve
    threshold yet."""

    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": ContinuousRandomAgent,
        "ppo": ContinuousPPOAgent,
        "td3": TD3Agent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "navigation_continuous_default.yaml",
        "ppo": "navigation_continuous_ppo.yaml",
        "td3": "navigation_continuous_td3.yaml",
    }

    @property
    def name(self) -> str:
        return CONTINUOUS_ID

    @property
    def key(self) -> str:
        return "navigation_continuous"

    @property
    def solve_threshold(self) -> float | None:
        return None

    @property
    def default_episodes(self) -> int:
        return DEFAULT_EPISODES

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> NavigationEnv:
        return make_continuous_env_from_config({**env_config, "render_mode": render_mode})

    def agent_config(self, agent_config: Mapping[str, Any], env: ChallengeEnv) -> dict[str, Any]:
        """The agent's config with the environment's action bounds, unless it sets its own."""
        return with_box_action_bounds(agent_config, env, self.name)
