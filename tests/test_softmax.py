"""Tests for the shared numerically stable softmax."""

import warnings

import numpy as np
import pytest

from rl_core.agents.softmax import stable_softmax


@pytest.mark.parametrize(
    "logits,expected",
    [
        # exp(0) = 1, exp(ln 2) = 2: [1, 2] / 3
        ([0.0, np.log(2.0)], [1 / 3, 2 / 3]),
        # exp gives [1, 2, 3]: [1, 2, 3] / 6
        ([0.0, np.log(2.0), np.log(3.0)], [1 / 6, 2 / 6, 3 / 6]),
        # adding a constant to every logit changes nothing: still [1, 2] / 3
        ([5.0, 5.0 + np.log(2.0)], [1 / 3, 2 / 3]),
        # equal logits: uniform
        ([0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]),
    ],
)
def test_stable_softmax_matches_hand_computed_values(logits, expected):
    np.testing.assert_allclose(stable_softmax(np.array(logits)), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "logits,expected",
    [
        ([1000.0, -1000.0], [1.0, 0.0]),  # exp(1000) overflows without the shift
        ([1000.0, 1000.0], [0.5, 0.5]),  # inf / inf without it
        ([-1000.0, -1000.0], [0.5, 0.5]),  # 0 / 0 without it
    ],
)
def test_stable_softmax_is_finite_for_extreme_logits(logits, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an overflow or invalid-value warning is a failure
        probs = stable_softmax(np.array(logits))

    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-12)


def test_stable_softmax_keeps_shape_and_dtype():
    probs = stable_softmax(np.array([0.1, -0.2, 0.3]))
    assert probs.shape == (3,)
    assert probs.dtype == np.float64


def test_stable_softmax_batch_normalises_each_row_on_its_own():
    # A maximum taken over the whole batch (1000) would leave row 2 as exp(-1000) / exp(-1000)
    # = 0 / 0; each row must be shifted by its own maximum.
    logits = np.array([[1000.0, 999.0], [0.0, np.log(3.0)]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probs = stable_softmax(logits)

    # Row 1: exp of [1, 0] after the shift, so [e, 1] / (e + 1); row 2: [1, 3] / 4.
    e = np.e
    np.testing.assert_allclose(
        probs, [[e / (e + 1), 1 / (e + 1)], [0.25, 0.75]], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], rtol=0, atol=1e-15)


def test_stable_softmax_batch_rows_match_the_single_row_result():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=5.0, size=(6, 4))

    batched = stable_softmax(logits)

    assert batched.shape == logits.shape
    for row, row_logits in zip(batched, logits, strict=True):
        np.testing.assert_allclose(row, stable_softmax(row_logits), rtol=0, atol=1e-15)


def test_stable_softmax_does_not_modify_its_input():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
    original = logits.copy()

    stable_softmax(logits)

    assert np.array_equal(logits, original)
