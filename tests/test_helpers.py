"""Tests for the shared numerical helpers in tests/helpers.py."""

import numpy as np
import pytest

from tests.helpers import assert_gradients_match, numerical_grad


def test_numerical_grad_matches_known_derivative():
    # loss = sum(w ** 2)  =>  d loss / d w = 2 * w
    w = np.array([[1.0, -2.0], [0.5, 3.0]])

    grad = numerical_grad(lambda: float(np.sum(w**2)), w)

    np.testing.assert_allclose(grad, 2 * w, rtol=1e-6, atol=1e-8)


def test_numerical_grad_restores_the_parameter():
    w = np.array([0.3, -1.7, 2.0])
    before = w.copy()

    numerical_grad(lambda: float(np.sum(np.sin(w))), w)

    np.testing.assert_array_equal(w, before)


def test_numerical_grad_rejects_non_float64_parameter():
    w = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="float64"):
        numerical_grad(lambda: float(np.sum(w**2)), w)


def test_assert_gradients_match_accepts_correct_gradients():
    w = np.array([1.0, -2.0, 0.5])
    b = np.array([0.25])

    def loss() -> float:
        return float(np.sum(w**3) + 4.0 * b[0])

    assert_gradients_match(loss, {"w": w, "b": b}, {"w": 3 * w**2, "b": np.array([4.0])})


def test_assert_gradients_match_names_the_parameter_with_a_wrong_sign():
    w = np.array([1.0, -2.0, 0.5])
    b = np.array([0.25])

    def loss() -> float:
        return float(np.sum(w**3) + 4.0 * b[0])

    with pytest.raises(AssertionError, match="'w'"):
        assert_gradients_match(loss, {"w": w, "b": b}, {"w": -3 * w**2, "b": np.array([4.0])})


def test_assert_gradients_match_reports_a_parameter_without_a_gradient():
    w = np.array([1.0])

    with pytest.raises(AssertionError, match="no analytic gradient"):
        assert_gradients_match(lambda: float(w[0] ** 2), {"w": w}, {})
