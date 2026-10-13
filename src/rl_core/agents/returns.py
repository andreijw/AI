"""Discounted returns, shared by every agent that learns from whole episodes."""

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
