"""Several training environments stepped in lockstep (``training.num_envs``).

The trainer asks a pool for every environment's step at once: one :meth:`EnvPool.reset` for the
environments that start an episode, then one :meth:`EnvPool.step` with the action of every one that
is playing. The calls are batched so that a pool that steps its environments elsewhere (worker
processes, or a simulator that steps many robots as one array) can implement the same interface;
:class:`InlinePool`, the one used today, steps them one after another in this process.

A pool never resets an environment on its own: the trainer decides when each one starts an episode
(after the curriculum's settings, and only while the budget allows), so the observation a step
returns is always the true last one of a finished episode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol

import numpy as np

from ..environments import ResumableEnv
from ..utils.config_values import positive_int
from .curriculum import require_schedulable

NUM_ENVS_KEY = "num_envs"


class ResetResult(NamedTuple):
    """One environment's start of an episode."""

    env_index: int
    observation: Any
    info: dict[str, Any]


class StepResult(NamedTuple):
    """One environment's step, as ``env.step`` returned it."""

    env_index: int
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class EnvPool(Protocol):
    """Training environments the trainer resets and steps together."""

    @property
    def num_envs(self) -> int: ...

    def reset(
        self, indices: Sequence[int], settings: Mapping[str, Any] | None
    ) -> list[ResetResult]: ...

    def step(self, actions: Mapping[int, Any]) -> list[StepResult]: ...


class InlinePool:
    """
    The environments themselves, stepped one after another in this process.

    An environment not named in a :meth:`step` call is not touched, so an environment that has
    finished its last episode draws nothing more from its generators.
    """

    def __init__(self, envs: Sequence[Any]) -> None:
        self.envs = tuple(envs)

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset(
        self, indices: Sequence[int], settings: Mapping[str, Any] | None = None
    ) -> list[ResetResult]:
        """Start an episode in each of ``indices``, in ascending order, giving each the
        curriculum's ``settings`` first when there are any."""
        results = []
        for index in sorted(indices):
            env = self.envs[index]
            if settings is not None:
                require_schedulable(env).apply_settings(settings)
            observation, info = env.reset()
            results.append(ResetResult(index, observation, info))
        return results

    def step(self, actions: Mapping[int, Any]) -> list[StepResult]:
        """Step each environment named in ``actions`` with its action, in ascending order."""
        results = []
        for index in sorted(actions):
            observation, reward, terminated, truncated, info = self.envs[index].step(actions[index])
            results.append(StepResult(index, observation, reward, terminated, truncated, info))
        return results


def num_envs_of(training: Mapping[str, Any]) -> int:
    """``training.num_envs``: a positive integer, 1 when unset."""
    return positive_int(training, NUM_ENVS_KEY, 1)


def single_env_training(training: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a training section without ``num_envs``, for a trainer that only evaluates on one
    environment (``record.py``, the robustness sweep) from a config that trains several."""
    return {key: value for key, value in training.items() if key != NUM_ENVS_KEY}


def env_state_key(index: int, key: str) -> str:
    """The trainer-file key of environment ``index``'s generator state ``key``: environment 0 keeps
    the keys a one-environment run has always written."""
    return key if index == 0 else f"env{index}_{key}"


def rng_states(envs: Sequence[Any]) -> dict[str, np.ndarray]:
    """Every resumable environment's generators, under :func:`env_state_key`."""
    return {
        env_state_key(index, key): value
        for index, env in enumerate(envs)
        if isinstance(env, ResumableEnv)
        for key, value in env.rng_state().items()
    }


def split_rng_states(envs: Sequence[Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The saved ``state`` of each environment, keyed as its own ``rng_state()``."""
    split = []
    for index, env in enumerate(envs):
        own = env.rng_state() if isinstance(env, ResumableEnv) else {}
        split.append(
            {
                key: state[env_state_key(index, key)]
                for key in own
                if env_state_key(index, key) in state
            }
        )
    return split
