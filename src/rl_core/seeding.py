"""Seeded random generators for components that must not share a stream."""

import numpy as np

# Stream used by environment wrappers for observation noise, action noise and domain
# randomization, kept apart from the stream the wrapped environment draws start states from.
WRAPPER_STREAM = 1


def derive_rng(seed: int | None, stream: int) -> np.random.Generator:
    """
    Create a generator for one named stream of a seeded component.

    Two components built from the same seed with ``np.random.default_rng(seed)`` draw the very
    same numbers. Giving each its own stream keeps them reproducible but independent.

    Args:
        seed: Non-negative seed shared by the component, or None for an entropy-seeded
            (non-reproducible) generator.
        stream: Non-negative identifier of the stream; different values give different
            sequences for the same seed.

    Returns:
        A generator that is the same for equal (seed, stream) and differs from
        ``np.random.default_rng(seed)``.

    Raises:
        ValueError: If seed or stream is not a non-negative integer (booleans are rejected).
    """
    for name, value in (("seed", seed), ("stream", stream)):
        if value is None and name == "seed":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    if seed is None:
        return np.random.default_rng()
    # A spawn key, unlike extra entropy words, cannot collide with a plain seed: entropy [1, 0]
    # is zero-padded to the same state as entropy 1, so default_rng([stream, seed]) would make
    # seed 0 on stream 1 identical to default_rng(1).
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(int(stream),)))
