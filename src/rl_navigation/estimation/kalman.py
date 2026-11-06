"""The linear Kalman filter's measurement update, in Joseph form."""

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.sim.geometry import FloatArray


def symmetrized(matrix: FloatArray) -> FloatArray:
    """``(M + M^T) / 2``: a covariance with the rounding asymmetry of its products removed."""
    return (matrix + matrix.T) / 2.0


def kalman_update(
    x: ArrayLike, P: ArrayLike, innovation: ArrayLike, H: ArrayLike, R: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Correct a Gaussian ``(x, P)`` with a linear measurement ``z = H x + v``, ``v ~ N(0, R)``.

    ``S = H P H^T + R``, ``K = P H^T S^-1`` (solved, never inverted), ``x' = x + K innovation``
    and the Joseph form ``P' = (I - K H) P (I - K H)^T + K R K^T``, symmetrised. Unlike
    ``(I - K H) P`` it stays positive semi-definite when rounding makes ``K H`` nearly the
    identity (a near-perfect measurement).

    Args:
        x: ``(D,)`` prior mean.
        P: ``(D, D)`` prior covariance.
        innovation: ``(M,)`` measurement minus its prediction, ``z - H x`` for a linear model.
        H: ``(M, D)`` measurement matrix.
        R: ``(M, M)`` measurement noise covariance.

    Returns:
        ``(x', P')``, new arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T  # S is symmetric: (S^-1 H P)^T = P H^T S^-1
    updated = x + K @ np.asarray(innovation, dtype=np.float64)
    I_KH = np.eye(len(x)) - K @ H
    covariance = I_KH @ P @ I_KH.T + K @ R @ K.T
    return updated, symmetrized(covariance)
