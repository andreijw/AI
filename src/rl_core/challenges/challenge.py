"""The benchmark challenge interface: one environment plus the agents that can run on it."""

from __future__ import annotations

import abc
import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

import gymnasium as gym

from rl_core.agents.base_agent import BaseAgent
from rl_core.training.budget import apply_budget_override
from rl_core.utils.config import load_config


class ChallengeEnv(Protocol):
    """The environment wrapper a challenge builds, as ``train.py``, ``record.py`` and ``Trainer`` use it."""

    env: gym.Env
    observation_space: gym.Space
    action_space: gym.Space

    def wrap_env(self, wrapper: gym.Env) -> None: ...

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]: ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


class BenchmarkChallenge(abc.ABC):
    """
    Abstract interface defining an AI / RL benchmark challenge.

    Allows benchmarking arbitrary environments and algorithms under identical conditions.

    A challenge built from YAML configs sets :attr:`CONFIG_MAP` (agent type to file name) and
    ``config_dir``, and loads them with :meth:`load_run_config`. ``train.py``, ``record.py`` and
    the benchmark build every environment and agent through :meth:`make_env` and
    :meth:`make_agent`, so a challenge is the only place that knows its wrapper and agent classes.

    A challenge lists its agents in :attr:`AGENT_CLASSES` (agent type to class); the defaults of
    :attr:`supported_agents` and :meth:`make_agent` read it, and :meth:`agent_config` is where a
    challenge adds what its environment implies (Pendulum's torque bounds).
    """

    CONFIG_MAP: ClassVar[Mapping[str, str]] = {}
    # Read-only, so a challenge that declares no table cannot add to one every such challenge shares.
    AGENT_CLASSES: ClassVar[Mapping[str, type[BaseAgent]]] = MappingProxyType({})

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
    def supported_agents(self) -> list[str]:
        """The agent types this challenge can build: the keys of :attr:`AGENT_CLASSES`, sorted."""
        return sorted(self.AGENT_CLASSES)

    @abc.abstractmethod
    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None) -> ChallengeEnv:
        """
        Build the challenge's environment from a config's ``environment`` section.

        A missing or ``null`` field takes the wrapper's default, whose environment id is
        :attr:`name`.

        Args:
            env_config: The ``environment`` section (``name``, ``seed``, ``max_episode_steps``,
                noise and domain randomization settings).
            render_mode: Gymnasium render mode, or None.
        """
        ...

    def make_agent(
        self, agent_type: str, agent_config: Mapping[str, Any], env: ChallengeEnv
    ) -> BaseAgent:
        """
        Build an agent of ``agent_type`` sized for ``env``, leaving ``agent_config`` unchanged.

        The agent's class comes from :attr:`AGENT_CLASSES`, in the backend the config's ``backend``
        key chooses (``BaseAgent.class_for_config``), its dimensions from the environment's
        flattened spaces and its config from :meth:`agent_config`.

        Raises:
            ValueError: If ``agent_type`` is not in :attr:`supported_agents`.
        """
        if agent_type not in self.AGENT_CLASSES:
            raise self.unsupported_agent_error(agent_type)
        config = self.agent_config(agent_config, env)
        return self.AGENT_CLASSES[agent_type].class_for_config(config)(
            observation_dim=gym.spaces.flatdim(env.observation_space),
            action_dim=gym.spaces.flatdim(env.action_space),
            config=config,
        )

    def check_agent_buildable(self, agent_type: str) -> None:
        """
        Resolve the class :meth:`setup_run` would build for ``agent_type`` from its config, without
        building an environment or an agent, so a run that cannot start fails before any other
        agent trains. A challenge with no config for ``agent_type`` has nothing to check here.

        Raises:
            ValueError: If the config names a backend this agent has no version in.
            ImportError: If the backend's library is not installed, naming how to install it.
        """
        if agent_type not in self.CONFIG_MAP or agent_type not in self.AGENT_CLASSES:
            return
        config = load_config(os.path.join(self.config_dir, self.CONFIG_MAP[agent_type]))
        self.AGENT_CLASSES[agent_type].class_for_config(config.get("agent", {}).get("config", {}))

    def agent_config(self, agent_config: Mapping[str, Any], env: ChallengeEnv) -> dict[str, Any]:
        """The config an agent is built with: a copy of ``agent_config`` by default; a challenge
        overrides this to add what its environment implies."""
        return dict(agent_config)

    def setup_run(
        self, agent_type: str, seed: int, episodes: int | None, steps: int | None = None
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Setup and return (env, agent, training_config) for a single benchmark run.

        Args:
            agent_type: Identifier of the agent to instantiate.
            seed: Random seed for environment and agent reproducibility.
            episodes: Training episodes for every agent, or None for this agent's own budget.
            steps: Training environment steps for every agent, or None.

        Returns:
            Tuple of (initialized_env, initialized_agent, training_config_dict). The
            training config's ``num_episodes`` is the budget the run trains for.
        """
        config = self.load_run_config(agent_type, seed, episodes, steps)
        env = self.make_env(config["environment"], render_mode=None)
        agent = self.make_agent(agent_type, config["agent"]["config"], env)
        return env, agent, config["training"]

    def setup_eval_env(self, agent_type: str, seed: int) -> ChallengeEnv | None:
        """
        A fresh environment for evaluating a benchmark run, apart from its training environment.

        Config-driven challenges (with ``agent_type`` in :attr:`CONFIG_MAP`) build it with
        :meth:`make_env` from the run's environment section. Others return None, and their runs
        evaluate on the training environment as before. A subclass that overrides
        :meth:`setup_run` to build its environment differently should override this too, or its
        evaluation environment will differ from its training one.
        """
        if agent_type not in self.CONFIG_MAP:
            return None
        config = self.load_run_config(agent_type, seed, None)
        return self.make_env(config["environment"], render_mode=None)

    def unsupported_agent_error(self, agent_type: str) -> ValueError:
        """The error for an agent type this challenge cannot build, naming the ones it can."""
        return ValueError(
            f"Agent type {agent_type!r} is not supported by challenge '{self.key}' "
            f"({self.name}). Supported: {sorted(self.supported_agents)}"
        )

    def load_run_config(
        self, agent_type: str, seed: int, episodes: int | None, steps: int | None = None
    ) -> dict[str, Any]:
        """
        Load an agent's config from :attr:`config_dir` and prepare it for one benchmark run.

        The seed replaces the environment and agent seeds. The budget is ``episodes`` or ``steps``
        when given (see :func:`~rl_core.training.budget.apply_budget_override`), else the config's
        ``training.total_steps`` or ``training.num_episodes``, else :attr:`default_episodes`.
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
        apply_budget_override(training, episodes, steps)
        training.setdefault("num_episodes", self.default_episodes)
        if training["num_episodes"] <= 0:
            raise ValueError(
                f"{self.CONFIG_MAP[agent_type]}: training.num_episodes must be positive, "
                f"got {training['num_episodes']}"
            )
        training["save_frequency"] = max(training["num_episodes"] + 1, 999999)
        return config


def with_box_action_bounds(
    agent_config: Mapping[str, Any], env: ChallengeEnv, owner: str
) -> dict[str, Any]:
    """
    A copy of ``agent_config`` with the environment's Box action bounds as ``action_low`` and
    ``action_high``, unless it sets its own: what a continuous challenge's
    :meth:`BenchmarkChallenge.agent_config` returns.

    Args:
        agent_config: The agent's config.
        env: The challenge's environment.
        owner: The challenge's environment id, named in the error.

    Raises:
        TypeError: If the environment's action space is not a Box.
    """
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"{owner} needs a Box action space, got {action_space}")
    config = dict(agent_config)
    config.setdefault("action_low", action_space.low)
    config.setdefault("action_high", action_space.high)
    return config
