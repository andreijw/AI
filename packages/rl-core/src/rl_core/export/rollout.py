"""The observations an agent meets when it acts greedily, for checking an export on real inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from rl_core.agents.base_agent import BaseAgent
from rl_core.agents.inference import InferencePolicy
from rl_core.challenges.challenge import ChallengeEnv
from rl_core.export.verify import ATOL, RTOL, near_ties

__all__ = ["Agreement", "GreedyRollout", "greedy_rollout", "policy_disagreements"]


@dataclass(frozen=True)
class GreedyRollout:
    """Each step's raw observation (before any normalisation), shape (steps, input), and the
    action the agent took on it in evaluation mode."""

    observations: npt.NDArray[np.float32]
    actions: list[int | npt.NDArray[np.floating]]


def greedy_rollout(
    env: ChallengeEnv, agent: BaseAgent, episodes: int, seed: int | None = None
) -> GreedyRollout:
    """
    Play ``episodes`` greedy episodes (``select_action(..., training=False)``, no learning, the
    normalisation statistics only read) and record what the agent saw and did. The first reset
    takes ``seed``; later ones continue the environment's generator.
    """
    observations: list[npt.NDArray[np.float32]] = []
    actions: list[int | npt.NDArray[np.floating]] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        done = False
        while not done:
            raw = np.asarray(obs, dtype=np.float32).ravel()
            action = agent.select_action(agent.normalize_observation(obs, training=False), False)
            observations.append(raw)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return GreedyRollout(np.array(observations, dtype=np.float32), actions)


class Agreement(NamedTuple):
    """Steps on which the exported policy acts unlike the agent (``mismatches``), and discrete steps
    where it does only because its two best outputs are within the tolerance (``near_ties``: the
    agent's float64 sums may round differently, as a folded dueling network's do)."""

    mismatches: int
    near_ties: int


def policy_disagreements(policy: InferencePolicy, rollout: GreedyRollout) -> Agreement:
    """Compare the exported policy, in NumPy, with the agent's actions over the rollout: a
    different discrete action is a mismatch unless it is a near-tie, and a continuous action is
    one outside the verification tolerance."""
    mismatches = ties = 0
    for obs, taken in zip(rollout.observations, rollout.actions, strict=True):
        exact = obs.astype(np.float64)
        ours = policy.act(exact)
        if policy.discrete:
            if ours != taken:
                tied = bool(near_ties(policy.outputs(exact))[0])
                ties += int(tied)
                mismatches += int(not tied)
        else:
            expected = np.asarray(taken, dtype=np.float64)
            mismatches += int(not np.allclose(ours, expected, rtol=RTOL, atol=ATOL))
    return Agreement(mismatches, ties)
