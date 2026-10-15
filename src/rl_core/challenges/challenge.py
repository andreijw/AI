"""The benchmark challenge interface: one environment plus the agents that can run on it."""

from __future__ import annotations

import abc
import os
from collections.abc import Mapping
from typing import Any, ClassVar

from rl_core.utils.config import load_config


class BenchmarkChallenge(abc.ABC):
    """
    Abstract interface defining an AI / RL benchmark challenge.

    Allows benchmarking arbitrary environments and algorithms under identical conditions.

    A challenge built from YAML configs sets :attr:`CONFIG_MAP` (agent type to file name) and
    ``config_dir``, and loads them with :meth:`load_run_config`.
    """

    CONFIG_MAP: ClassVar[Mapping[str, str]] = {}

    def __init__(self, config_dir: str = "configs") -> None:
        """
        Args:
            config_dir: Directory holding the challenge's YAML configs.
        """
        self.config_dir = config_dir

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable challenge name (e.g. 'CartPole-v1', 'Pendulum-v1')."""
        ...

    @property
    @abc.abstractmethod
    def key(self) -> str:
        """CLI / lookup identifier key (e.g. 'cartpole', 'pendulum')."""
        ...

    @property
    @abc.abstractmethod
    def solve_threshold(self) -> float | None:
        """Reward score threshold defining task completion, or None if unbounded."""
        ...

    @property
    @abc.abstractmethod
    def default_episodes(self) -> int:
        """Training episodes for an agent whose config does not set ``training.num_episodes``."""
        ...

    @property
    @abc.abstractmethod
    def supported_agents(self) -> list[str]:
        """List of supported agent types for this challenge."""
        ...

    @abc.abstractmethod
    def setup_run(
        self, agent_type: str, seed: int, episodes: int | None
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Setup and return (env, agent, training_config) for a single benchmark run.

        Args:
            agent_type: Identifier of the agent to instantiate.
            seed: Random seed for environment and agent reproducibility.
            episodes: Training episodes for every agent, or None for this agent's own budget.

        Returns:
            Tuple of (initialized_env, initialized_agent, training_config_dict). The
            training config's ``num_episodes`` is the budget the run trains for.
        """
        ...

    def load_run_config(self, agent_type: str, seed: int, episodes: int | None) -> dict[str, Any]:
        """
        Load an agent's config from :attr:`config_dir` and prepare it for one benchmark run.

        The seed replaces the environment and agent seeds. The budget is ``episodes`` when
        given, else the config's ``training.num_episodes``, else :attr:`default_episodes`.
        Checkpointing is pushed past the end of the run.

        Raises:
            KeyError: If :attr:`CONFIG_MAP` has no config for ``agent_type``.
            ValueError: If the resulting budget is not positive.
        """
        if agent_type not in self.CONFIG_MAP:
            raise KeyError(
                f"Challenge '{self.name}' has no config for agent '{agent_type}' in CONFIG_MAP "
                f"(config_dir: {self.config_dir})"
            )
        config = load_config(os.path.join(self.config_dir, self.CONFIG_MAP[agent_type]))

        config.setdefault("environment", {})["seed"] = seed
        config.setdefault("agent", {}).setdefault("config", {})["seed"] = seed
        training = config.setdefault("training", {})
        if episodes is not None:
            training["num_episodes"] = episodes
        training.setdefault("num_episodes", self.default_episodes)
        if training["num_episodes"] <= 0:
            raise ValueError(
                f"{self.CONFIG_MAP[agent_type]}: training.num_episodes must be positive, "
                f"got {training['num_episodes']}"
            )
        training["save_frequency"] = max(training["num_episodes"] + 1, 999999)
        return config
