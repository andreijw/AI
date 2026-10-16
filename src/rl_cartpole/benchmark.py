"""CartPole-v1 benchmark challenge implementation for the AI benchmark framework."""

from __future__ import annotations

from typing import Any, ClassVar

from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_core.agents import (
    ActorCriticAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
)
from rl_core.challenges import BenchmarkChallenge, register_challenge

AGENT_CLASSES: dict[str, Any] = {
    "random": RandomAgent,
    "reinforce": ReinforceAgent,
    "actor_critic": ActorCriticAgent,
    "ppo": PPOAgent,
    "dqn": DQNAgent,
}


@register_challenge
class CartPoleChallenge(BenchmarkChallenge):
    """CartPole-v1 discrete balance control benchmark challenge."""

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

    @property
    def supported_agents(self) -> list[str]:
        return sorted(AGENT_CLASSES)

    def setup_run(
        self, agent_type: str, seed: int, episodes: int | None, steps: int | None = None
    ) -> tuple[Any, Any, dict[str, Any]]:
        config = self.load_run_config(agent_type, seed, episodes, steps)

        env_cfg = config["environment"]
        env = CartPoleEnv(
            env_name=env_cfg.get("name", "CartPole-v1"),
            render_mode=None,
            max_episode_steps=env_cfg.get("max_episode_steps", 500),
            seed=seed,
            obs_noise_std=env_cfg.get("obs_noise_std", 0.0),
            action_noise_prob=env_cfg.get("action_noise_prob", 0.0),
            domain_randomization=env_cfg.get("domain_randomization"),
        )
        obs_dim = (
            int(env.observation_space.shape[0]) if env.observation_space.shape is not None else 4
        )
        action_dim = int(getattr(env.action_space, "n", 2))
        agent = AGENT_CLASSES[agent_type](
            observation_dim=obs_dim,
            action_dim=action_dim,
            config=config["agent"]["config"],
        )
        return env, agent, config["training"]
