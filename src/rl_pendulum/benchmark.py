"""Pendulum-v1 continuous control benchmark challenge implementation."""

from __future__ import annotations

from typing import Any, ClassVar

from benchmark import BenchmarkChallenge, register_challenge
from rl_core.agents import (
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
)
from rl_pendulum.environments.pendulum_env import PendulumEnv

AGENT_CLASSES: dict[str, Any] = {
    "random": ContinuousRandomAgent,
    "actor_critic": ContinuousActorCriticAgent,
    "ppo": ContinuousPPOAgent,
}


@register_challenge
class PendulumChallenge(BenchmarkChallenge):
    """Pendulum-v1 continuous torque control benchmark challenge."""

    CONFIG_MAP: ClassVar[dict[str, str]] = {
        "random": "pendulum_default.yaml",
        "actor_critic": "pendulum_actor_critic.yaml",
        "ppo": "pendulum_ppo.yaml",
    }

    def __init__(self, config_dir: str = "configs") -> None:
        self.config_dir = config_dir

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

    @property
    def supported_agents(self) -> list[str]:
        return sorted(AGENT_CLASSES)

    def setup_run(
        self, agent_type: str, seed: int, episodes: int | None
    ) -> tuple[Any, Any, dict[str, Any]]:
        config = self.load_run_config(agent_type, seed, episodes)

        env_cfg = config["environment"]
        env = PendulumEnv(
            env_name=env_cfg.get("name", "Pendulum-v1"),
            render_mode=None,
            max_episode_steps=env_cfg.get("max_episode_steps", 200),
            seed=seed,
            obs_noise_std=env_cfg.get("obs_noise_std", 0.0),
            action_noise_std=env_cfg.get("action_noise_std", 0.0),
            domain_randomization=env_cfg.get("domain_randomization"),
        )
        obs_dim = (
            int(env.observation_space.shape[0]) if env.observation_space.shape is not None else 3
        )
        action_dim = int(env.action_space.shape[0]) if env.action_space.shape is not None else 1

        agent_cfg = dict(config["agent"]["config"])
        if hasattr(env.action_space, "low") and "action_low" not in agent_cfg:
            agent_cfg["action_low"] = env.action_space.low
        if hasattr(env.action_space, "high") and "action_high" not in agent_cfg:
            agent_cfg["action_high"] = env.action_space.high

        agent = AGENT_CLASSES[agent_type](
            observation_dim=obs_dim,
            action_dim=action_dim,
            config=agent_cfg,
        )
        return env, agent, config["training"]
