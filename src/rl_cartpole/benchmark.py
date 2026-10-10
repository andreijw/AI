"""CartPole-v1 benchmark challenge implementation for the AI benchmark framework."""

from __future__ import annotations

import os
from typing import Any

from benchmark import BenchmarkChallenge, register_challenge
from rl_cartpole.agents import (
    ActorCriticAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
)
from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_cartpole.utils.config import load_config

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

    CONFIG_MAP: dict[str, str] = {
        "random": "cartpole_default.yaml",
        "reinforce": "cartpole_reinforce.yaml",
        "actor_critic": "cartpole_actor_critic.yaml",
        "ppo": "cartpole_ppo.yaml",
        "dqn": "cartpole_dqn.yaml",
    }

    def __init__(self, config_dir: str = "configs") -> None:
        self.config_dir = config_dir

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
        self, agent_type: str, seed: int, episodes: int
    ) -> tuple[Any, Any, dict[str, Any]]:
        config_file = self.CONFIG_MAP.get(agent_type, "cartpole_default.yaml")
        config_path = os.path.join(self.config_dir, config_file)
        config = load_config(config_path)

        config.setdefault("environment", {})["seed"] = seed
        config.setdefault("agent", {}).setdefault("config", {})["seed"] = seed
        config.setdefault("training", {})["num_episodes"] = episodes
        config["training"]["save_frequency"] = max(episodes + 1, 999999)

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
