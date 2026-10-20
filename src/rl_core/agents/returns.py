"""Discounted returns, advantage estimates and episode-end handling, shared by the agents that learn
from episodes."""

from typing import Any

import numpy as np


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the discounted return G_t = Σ_{k≥t} γ^(k-t) r_k for each step of one episode.

    Args:
        rewards: 1-D array of per-step rewards, in the order they were received.
        gamma: Discount factor. Not validated here; agents validate it at construction.

    Returns:
        float64 array of the same length as ``rewards``; empty if ``rewards`` is empty.
    """
    n_steps = len(rewards)
    returns = np.empty(n_steps, dtype=np.float64)
    cumulative = 0.0
    for t in range(n_steps - 1, -1, -1):
        cumulative = float(rewards[t]) + gamma * cumulative
        returns[t] = cumulative
    return returns


def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> np.ndarray:
    """
    Generalised advantage estimates (Schulman et al., 2016) for a rollout of one or more episodes.

    ``A_t = δ_t + γλ·(1 − done_t)·A_{t+1}`` with ``δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)``,
    computed from the last step back. A step with ``done_t`` ends its episode: nothing is
    bootstrapped across it and the trace restarts.

    Args:
        rewards: Per-step rewards, shape (T,).
        values: V(s_t) for each step, shape (T,).
        dones: True on the last step of each episode, shape (T,).
        gamma: Discount factor.
        lam: GAE λ in [0, 1]; 1 gives the Monte-Carlo return minus V(s_t).
        last_value: V of the state after the final step, used only if that step is not done.

    Returns:
        float64 array of advantages, shape (T,).
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        next_value = values[t + 1] if t + 1 < n_steps else last_value
        done_mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * done_mask - values[t]
        last_gae = delta + gamma * lam * done_mask * last_gae
        advantages[t] = last_gae
    return advantages


def episode_end(
    n_steps: int,
    dones: Any = None,
    terminated: Any = None,
    next_observations: Any = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Where an episode's advantage estimate ends, and the state to bootstrap from if it was cut.

    The episode's last step always ends the estimate, so that no episode in a rollout borrows from
    the next. It was cut, not ended, when its final ``terminated`` flag (or, without those flags,
    its final ``done``) is False: a time limit, or the Trainer's ``max_steps_per_episode``. A cut
    episode should then add ``gamma * V(next state)`` to its last reward, as Stable-Baselines3 does
    for timeouts; without a known next state the cut is treated as a true end.

    Args:
        n_steps: Number of steps in the episode.
        dones: Per-step done flags, shape (n_steps,), or None for all False.
        terminated: Per-step terminated flags; only the last one is read. None to rely on ``dones``.
        next_observations: Per-step next states; only the last one is read. None if unknown.

    Returns:
        A new bool array of shape (n_steps,) with the last step set True, and the float64 next
        state to bootstrap from (None for a true end or an unknown next state). The inputs are not
        modified.

    Raises:
        ValueError: If ``dones`` does not hold one flag per step.
    """
    ends = np.zeros(n_steps, dtype=bool) if dones is None else np.array(dones, dtype=bool)
    if ends.shape != (n_steps,):
        raise ValueError(f"dones must hold one flag per step ({n_steps}), got shape {ends.shape}")
    if n_steps == 0:
        return ends, None
    if terminated is not None and len(terminated) > 0:
        ends[-1] = bool(np.asarray(terminated, dtype=bool)[-1])
    bootstrap_obs = None
    if not ends[-1] and next_observations is not None and len(next_observations) > 0:
        bootstrap_obs = np.asarray(next_observations[-1], dtype=np.float64)
    ends[-1] = True
    return ends, bootstrap_obs
