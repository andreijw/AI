"""The linear Kalman update the EKF calls, in Joseph form: hand-computed cases."""

import numpy as np

from rl_navigation.estimation import kalman_update


def test_one_coordinate_measured_by_hand():
    # S = 4 + 1 = 5, K = (0.8, 0), x' = x + K * 2, P'_00 = (1 - 0.8)^2 4 + 0.8^2 1 = 0.8.
    x, P = kalman_update(
        np.array([1.0, 2.0]),
        np.diag([4.0, 1.0]),
        np.array([2.0]),
        np.array([[1.0, 0.0]]),
        np.eye(1),
    )
    np.testing.assert_allclose(
        x, [2.6, 2.0], rtol=1e-12, err_msg="x' = x + K nu with K = (0.8, 0)."
    )
    np.testing.assert_allclose(
        P, np.diag([0.8, 1.0]), rtol=1e-12, err_msg="P'_00 = (1 - K)^2 P + K^2 R."
    )


def test_a_sum_of_correlated_states_by_hand():
    # H = (1, 1): S = 2 + 1 + 1 + 3 + 1 = 8, P H^T = (3, 4), K = (3, 4) / 8, P' = P - (3, 4)(3, 4)^T / 8.
    x, P = kalman_update(
        np.zeros(2), np.array([[2.0, 1.0], [1.0, 3.0]]), np.ones(1), np.ones((1, 2)), np.eye(1)
    )
    np.testing.assert_allclose(x, [0.375, 0.5], rtol=1e-12, err_msg="x' = K for a unit innovation.")
    np.testing.assert_allclose(
        P, [[0.875, -0.5], [-0.5, 1.0]], rtol=1e-12, err_msg="P' = P - K S K^T."
    )


def test_joseph_form_keeps_a_near_perfect_measurement_positive_definite():
    """With R = 1e-17 the gain rounds to exactly 1: the standard form (I - K H) P gives P' = 0,
    which Cholesky refuses, while the Joseph form keeps K R K^T = R."""
    _, P = kalman_update(np.zeros(1), np.eye(1), np.zeros(1), np.eye(1), np.full((1, 1), 1e-17))
    assert P[0, 0] == 1e-17, f"Joseph form should leave P' = R = 1e-17, got {P[0, 0]!r}."
    np.linalg.cholesky(P)


def test_the_updated_covariance_is_exactly_symmetric():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    P = A @ A.T + np.eye(3)
    H = rng.standard_normal((5, 3))
    _, updated = kalman_update(np.zeros(3), P, rng.standard_normal(5), H, 0.01 * np.eye(5))
    np.testing.assert_array_equal(updated, updated.T, err_msg="P' should be symmetrised.")
