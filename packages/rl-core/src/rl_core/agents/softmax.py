"""The numerically stable softmax shared by the discrete policy agents."""

import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Softmax over the last axis, shifted by each row's maximum so ``exp`` never overflows.

    Subtracting the maximum leaves the result unchanged mathematically and keeps every exponent
    at or below zero, so logits of ±1000 give finite probabilities instead of inf / inf.

    Args:
        logits: Array of shape (action_dim,) for one observation, or (batch, action_dim) for a
            batch; each row is normalised on its own.

    Returns:
        Array of the same shape and float dtype as ``logits`` whose rows sum to one.
        ``logits`` is not modified.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)
