"""Seeded random generators for components that must not share a stream."""

import json

import numpy as np

# Stream used by environment wrappers for observation noise, action noise and domain
# randomization, kept apart from the stream the wrapped environment draws start states from.
WRAPPER_STREAM = 1

# An agent's generators, one per role. Committed configs give an agent the same seed as its
# environment, whose start states Gymnasium draws exactly like np.random.default_rng(seed), so every
# agent generator comes from one of these streams (and differs from WRAPPER_STREAM).
# WEIGHTS: weight initialisation. EXPLORATION: action sampling, action noise, epsilon-greedy, random
# agents' training actions. REPLAY: replay-buffer sampling and minibatch order. TARGET_NOISE: TD3's
# target-policy smoothing noise (13 and 20 are the evaluation streams below).
AGENT_WEIGHTS_STREAM = 10
AGENT_EXPLORATION_STREAM = 11
AGENT_REPLAY_STREAM = 12
AGENT_TARGET_NOISE_STREAM = 14


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


# Stream for the seed of a run's fixed evaluation start states (see eval_seed_for).
EVAL_STREAM = 20
# Stream a random agent draws its evaluation actions from, apart from its training actions.
AGENT_EVAL_STREAM = 13


def eval_seed_for(seed: int | None) -> int:
    """
    The first reset seed of a run's evaluation episodes; episode i of every evaluation resets with
    ``eval_seed_for(seed) + i``, so all evaluations in a run play the same start states.

    The value is drawn from the run's seed on its own stream and lies in [2**31, 2**32), far above the
    small seeds runs train with, so evaluation start states do not repeat a training sequence. A run
    without a seed gets an entropy-drawn value: fixed within that run, not reproducible across runs.
    """
    return int(derive_rng(seed, EVAL_STREAM).integers(2**31, 2**32))


def generator_state(generator: np.random.Generator) -> np.ndarray:
    """
    The generator's state as a checkpoint array: a 0-d string array of its bit generator's state
    in JSON, which ``np.load`` reads back without pickle.
    """
    return np.array(json.dumps(generator.bit_generator.state))


def restore_generator(generator: np.random.Generator, state: np.ndarray) -> None:
    """
    Put ``generator`` in the state :func:`generator_state` saved, in place, so every reference to it
    continues the saved sequence.

    Raises:
        ValueError: If the state belongs to another kind of bit generator.
    """
    generator.bit_generator.state = json.loads(str(np.asarray(state).item()))


def check_generator_state(generator: np.random.Generator, state: np.ndarray, name: str) -> None:
    """
    Check that ``state`` (from :func:`generator_state`) can be restored into ``generator``, without
    changing anything, so a caller can check every state before restoring any.

    Raises:
        ValueError: Naming ``name``, if ``state`` is not a generator state or is one of another kind
            of bit generator.
    """
    try:
        kind = json.loads(str(np.asarray(state).item()))["bit_generator"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{name} is not a generator state: {error}") from error
    expected = type(generator.bit_generator).__name__
    if kind != expected:
        raise ValueError(f"{name} holds a {kind} state, the generator is {expected}")
